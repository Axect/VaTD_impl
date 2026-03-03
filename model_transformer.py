"""
Lattice-GPT: Autoregressive Transformer for lattice spin systems.

Replaces the PixelCNN backbone with a causal Transformer while keeping
the same VaTD training objective (variational free energy minimization
via REINFORCE).  Designed to test the architecture independence of the
low-rank phenomenon observed in DiscretePixelCNN.

Architecture (v0.20.1 — Tier 1 engineering fixes)
--------------------------------------------------
  Flatten 16×16 → 256-token sequence (raster order)
  Token embedding: nn.Embedding(q, d_model)
  Position embedding: optional absolute + periodic 2D relative bias
  Temperature conditioning: AdaLN in every block
  Manual QKV + F.scaled_dot_product_attention (FlashAttention)
  Proper projected KV-cache for O(L²) sampling with O(d²) per step
  Output: nn.Linear(d_model, q) → per-position logits

Interface is identical to DiscretePixelCNN:
  sample(batch_size, T) → [B, 1, H, W]
  log_prob(sample, T)   → [B, 1]
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention as _sdpa


# ──────────────────────────────────────────────────────────────
# Adaptive Layer Normalization (AdaLN)
# ──────────────────────────────────────────────────────────────


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization conditioned on temperature.

    Given a conditioning vector c (from temperature MLP), produces
    per-channel scale (γ) and shift (β):

        AdaLN(x, c) = γ(c) ⊙ LayerNorm(x) + β(c)

    Zero-initialized so that at init AdaLN ≈ LayerNorm (identity modulation).
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(cond_dim, 2 * d_model)
        # Zero-init so modulation starts as identity (scale=1, shift=0)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
            cond: [B, cond_dim]
        Returns:
            [B, L, d_model]
        """
        scale_shift = self.proj(cond)  # [B, 2*d_model]
        scale, shift = scale_shift.chunk(2, dim=-1)  # each [B, d_model]
        # Unsqueeze for broadcasting over sequence length
        scale = scale.unsqueeze(1)  # [B, 1, d_model]
        shift = shift.unsqueeze(1)  # [B, 1, d_model]
        return (1.0 + scale) * self.norm(x) + shift


# ──────────────────────────────────────────────────────────────
# Causal Transformer Block (with KV-cache support)
# ──────────────────────────────────────────────────────────────


class CausalTransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with AdaLN and causal masking.

    AdaLN → MHSA (causal) → residual → AdaLN → FFN → residual

    Uses manual QKV projections + F.scaled_dot_product_attention for
    FlashAttention support and correct projected KV-cache.

    KV-cache stores post-projection K, V as [B, n_heads, L, head_dim]
    so incremental decoding only projects the new token (O(d²) per step
    instead of O(L·d²) with the old nn.MHA approach, which re-projected
    all cached states at every step — a correctness bug, not just perf).

    When ``attn_bias`` is provided, SDPA falls back to the math kernel
    (no FlashAttention); without bias, FlashAttention is used via
    ``is_causal=True``.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, cond_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.adaln1 = AdaLN(d_model, cond_dim)
        # Manual QKV projections (replaces nn.MultiheadAttention)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout

        self.adaln2 = AdaLN(d_model, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def _reshape_heads(self, t, B):
        """Reshape [B, L, d_model] → [B, n_heads, L, head_dim]."""
        return t.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, cond, past_kv=None, attn_bias=None):
        """
        Args:
            x: [B, L, d_model] (full sequence) or [B, 1, d_model] (incremental)
            cond: [B, cond_dim]
            past_kv: optional tuple (past_key, past_value),
                     each [B, n_heads, kv_len, head_dim]
            attn_bias: optional additive bias for attention scores.
                Full-sequence: [n_heads, L, L]
                Incremental:   [n_heads, 1, kv_len]

        Returns:
            output: [B, L, d_model] or [B, 1, d_model]
            new_kv: tuple (key, value), each [B, n_heads, kv_len, head_dim]
        """
        B = x.shape[0]

        # Self-attention with causal mask
        h = self.adaln1(x, cond)

        # Project Q, K, V and reshape to multi-head format
        q = self._reshape_heads(self.W_q(h), B)  # [B, n_heads, L_q, head_dim]
        k = self._reshape_heads(self.W_k(h), B)
        v = self._reshape_heads(self.W_v(h), B)

        # Concatenate with cached K, V (incremental decoding)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # [B, n_heads, kv_len, head_dim]
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v)  # Store projected K, V

        # Compute attention
        drop_p = self.attn_dropout if self.training else 0.0

        if past_kv is None and attn_bias is None:
            # Full sequence, no bias → FlashAttention via is_causal
            attn_out = _sdpa(q, k, v, is_causal=True, dropout_p=drop_p)
        else:
            # Build explicit mask when bias is present or doing incremental
            if past_kv is None:
                # Full sequence with bias: causal mask + bias
                L = q.shape[2]
                causal = torch.full(
                    (L, L), float('-inf'), device=q.device, dtype=q.dtype
                )
                causal = torch.triu(causal, diagonal=1)
                mask = causal.unsqueeze(0) + attn_bias  # [n_heads, L, L]
            else:
                # Incremental: single query → all keys causal by construction
                mask = attn_bias  # [n_heads, 1, kv_len] or None

            attn_out = _sdpa(q, k, v, attn_mask=mask, dropout_p=drop_p)

        # Reshape back: [B, n_heads, L, head_dim] → [B, L, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        attn_out = self.W_o(attn_out)

        x = x + attn_out

        # Feed-forward
        h = self.adaln2(x, cond)
        h = self.ffn(h)
        x = x + h

        return x, new_kv


# ──────────────────────────────────────────────────────────────
# Causal Transformer Backbone
# ──────────────────────────────────────────────────────────────


class CausalTransformerBackbone(nn.Module):
    """
    Stack of CausalTransformerBlocks with a final LayerNorm.

    Exposes `self.blocks` (nn.ModuleList) for activation hooks
    in analyze_rank.py.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_ff, cond_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, cond, past_kvs=None, attn_bias=None):
        """
        Args:
            x: [B, L, d_model] or [B, 1, d_model]
            cond: [B, cond_dim]
            past_kvs: optional list of (key, value) tuples, one per block
            attn_bias: optional additive attention bias, passed to each block

        Returns:
            x: [B, L, d_model] or [B, 1, d_model]
            new_kvs: list of (key, value) tuples
        """
        new_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, kv = block(x, cond, past_kv=past_kv, attn_bias=attn_bias)
            new_kvs.append(kv)
        return self.final_norm(x), new_kvs


# ──────────────────────────────────────────────────────────────
# LatticeGPT: Full Model
# ──────────────────────────────────────────────────────────────


class LatticeGPT(nn.Module):
    """
    Autoregressive Transformer for lattice spin systems.

    Flattens the 2D lattice into a 1D sequence (raster order),
    applies a causal Transformer, and predicts per-site categorical
    distributions.  Same interface as DiscretePixelCNN.

    Args:
        hparams: dict with keys matching net_config in YAML configs.
        device: torch device string.
    """

    def __init__(self, hparams: dict, device: str = "cpu"):
        super().__init__()
        self.hparams = hparams
        self.device = device

        # ── Lattice configuration ──
        size = hparams["size"]
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        H, W = self.size
        self.seq_len = H * W  # 256 for 16×16

        self.channel = 1
        self.category = hparams.get("category", 2)

        self.fix_first = hparams.get("fix_first", 1)
        self.batch_size = hparams["batch_size"]
        self.num_beta = hparams["num_beta"]
        self.beta_min = hparams["beta_min"]
        self.beta_max = hparams["beta_max"]

        # ── State mapping (same as DiscretePixelCNN) ──
        if self.category == 2:
            self.mapping = lambda x: 2 * x - 1        # {0,1} → {-1,+1}
            self.reverse_mapping = lambda x: torch.div(x + 1, 2, rounding_mode="trunc")
        else:
            self.mapping = lambda x: x                 # Potts: keep as-is
            self.reverse_mapping = lambda x: x

        # ── Training settings (curriculum, logit temp, etc.) ──
        self.curriculum_enabled = hparams.get("curriculum_enabled", False)
        self.curriculum_warmup_epochs = hparams.get("curriculum_warmup_epochs", 50)
        self.curriculum_start_beta_max = hparams.get(
            "curriculum_start_beta_max", self.beta_min * 1.5
        )
        self.phase1_epochs = hparams.get("phase1_epochs", 50)
        self.phase1_beta_max = hparams.get("phase1_beta_max", 0.35)
        self.phase2_epochs = hparams.get("phase2_epochs", 100)

        self.logit_temp_scale = hparams.get("logit_temp_scale", False)
        self.temp_scale_min = hparams.get("temp_scale_min", 0.1)
        self.temp_scale_max = hparams.get("temp_scale_max", 2.0)

        # ── Transformer hyperparameters ──
        self._d_model = hparams.get("d_model", 192)
        n_heads = hparams.get("n_heads", 6)
        n_layers = hparams.get("n_layers", 6)
        d_ff = hparams.get("d_ff", 768)
        cond_dim = hparams.get("cond_dim", 64)
        dropout = hparams.get("dropout", 0.0)

        # ── Token embedding ──
        self.tok_embed = nn.Embedding(self.category, self._d_model)

        # ── 2D Relative Position Bias (optional, periodic for PBC) ──
        self.use_rel_pos_bias = hparams.get("use_rel_pos_bias", False)
        self.drop_abs_pos_when_rel = hparams.get("drop_abs_pos_when_rel", False)

        # ── Position embeddings ──
        if self.use_rel_pos_bias and self.drop_abs_pos_when_rel:
            # Drop absolute position embeddings (zero buffer, non-learnable)
            self.register_buffer(
                "pos_embed", torch.zeros(self.seq_len, self._d_model)
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(self.seq_len, self._d_model) * 0.02
            )

        if self.use_rel_pos_bias:
            # Periodic distances for PBC lattice topology
            positions = torch.arange(self.seq_len)
            row = positions // W
            col = positions % W
            # Absolute row/col differences
            abs_dr = (row.unsqueeze(0) - row.unsqueeze(1)).abs()  # [L, L]
            abs_dc = (col.unsqueeze(0) - col.unsqueeze(1)).abs()  # [L, L]
            # Periodic wrapping: min(d, size - d)
            periodic_dr = torch.min(abs_dr, H - abs_dr)  # range [0, H//2]
            periodic_dc = torch.min(abs_dc, W - abs_dc)  # range [0, W//2]
            # Flat index into compact bias table
            max_dc = W // 2 + 1
            rel_pos_index = periodic_dr * max_dc + periodic_dc  # [L, L]
            self.register_buffer("rel_pos_index", rel_pos_index)

            num_rel_pos = (H // 2 + 1) * (W // 2 + 1)
            self.rel_pos_bias_table = nn.Parameter(
                torch.zeros(n_heads, num_rel_pos)
            )
            nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        # ── Temperature conditioning MLP ──
        # Maps scalar T → cond_dim vector
        self.temp_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # ── Transformer backbone ──
        self.backbone = CausalTransformerBackbone(
            d_model=self._d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            cond_dim=cond_dim,
            dropout=dropout,
        )

        # ── Output head ──
        self.output_head = nn.Linear(self._d_model, self.category)

        self.to(device)

    def _get_rel_pos_bias(self, query_len, kv_len=None):
        """
        Compute relative position bias for attention.

        Args:
            query_len: number of query positions (L for full, 1 for incremental)
            kv_len: number of key/value positions (defaults to query_len)

        Returns:
            [n_heads, query_len, kv_len] additive bias tensor
        """
        if not self.use_rel_pos_bias:
            return None

        if kv_len is None:
            kv_len = query_len

        if query_len == self.seq_len and kv_len == self.seq_len:
            # Full sequence: use pre-computed index [L, L]
            idx = self.rel_pos_index  # [L, L]
        else:
            # Incremental or partial: compute the relevant slice
            # For incremental decoding at position `pos` (query_len=1, kv_len=pos+1),
            # we need the row `pos` of the full index table, columns 0..pos
            # But we don't know `pos` here; caller should pass the right slice.
            # Fallback: compute from full table
            # query positions: last `query_len` positions of a kv_len-length sequence
            start_q = kv_len - query_len
            idx = self.rel_pos_index[start_q:kv_len, :kv_len]  # [query_len, kv_len]

        # Gather bias: [n_heads, query_len, kv_len]
        bias = self.rel_pos_bias_table[:, idx.reshape(-1)]  # [n_heads, query_len*kv_len]
        bias = bias.reshape(-1, query_len, kv_len)  # [n_heads, query_len, kv_len]
        return bias

    def _compute_temp_scale(self, T):
        """
        Temperature-dependent logit scaling (same as DiscretePixelCNN).

        Args:
            T: [B] or [B, 1] temperature tensor.
        Returns:
            scale: [B, 1, 1] for broadcasting with logits [B, L, q],
                   or float 1.0 if disabled.
        """
        if not self.logit_temp_scale:
            return 1.0

        if T.dim() == 1:
            T = T.unsqueeze(1)

        scale = 1.0 / T
        scale = scale.clamp(min=self.temp_scale_min, max=self.temp_scale_max)
        return scale.unsqueeze(-1)  # [B, 1, 1]

    def _get_cond(self, T):
        """
        Compute conditioning vector from temperature.

        Args:
            T: [B] or [B, 1] temperature tensor.
        Returns:
            [B, cond_dim]
        """
        if T.dim() == 1:
            T = T.unsqueeze(1)  # [B, 1]
        return self.temp_mlp(T)

    def _forward_logits(self, tokens, T):
        """
        Compute per-position logits from a token sequence (training path).

        Shift-right: position i receives token i-1, so logits at position i
        predict token i from context 0..i-1.

        Args:
            tokens: [B, L] integer token indices in {0, ..., q-1}.
            T: [B] temperature tensor.

        Returns:
            logits: [B, L, q] unnormalized log-probabilities.
        """
        B, L = tokens.shape

        # Shift right: prepend a learned "start" token (use zeros)
        # Position 0 gets zero embedding, position i gets token i-1
        tok_emb = self.tok_embed(tokens)  # [B, L, d_model]
        # Shift: drop last, prepend zeros
        zero_start = torch.zeros(B, 1, tok_emb.shape[-1], device=tok_emb.device)
        tok_emb = torch.cat([zero_start, tok_emb[:, :-1, :]], dim=1)  # [B, L, d_model]

        # Add positional embedding
        x = tok_emb + self.pos_embed[:L]  # [B, L, d_model]

        # Temperature conditioning
        cond = self._get_cond(T)  # [B, cond_dim]

        # 2D relative position bias (if enabled)
        attn_bias = self._get_rel_pos_bias(L)

        # Transformer backbone (no KV-cache for training)
        x, _ = self.backbone(x, cond, attn_bias=attn_bias)  # [B, L, d_model]

        # Output logits
        logits = self.output_head(x)  # [B, L, q]

        # Temperature-dependent scaling
        temp_scale = self._compute_temp_scale(T)
        if isinstance(temp_scale, torch.Tensor):
            logits = logits * temp_scale

        return logits

    @torch.no_grad()
    def sample(self, batch_size=None, T=None):
        """
        Autoregressive sampling with KV-cache for efficiency.

        Uses incremental decoding: at each step, only the new token is
        processed through the Transformer, with KV-cache accumulating
        past key/value states.  This is O(L²) total vs O(L³) naive.

        Args:
            batch_size: number of samples.
            T: [B] temperature tensor.

        Returns:
            [B, 1, H, W] samples in physical representation ({-1,+1} or {0,...,q-1}).
        """
        if batch_size is None:
            batch_size = self.batch_size

        H, W = self.size
        L = self.seq_len

        tokens = torch.zeros(batch_size, L, dtype=torch.long, device=self.device)

        # Fix first spin if configured
        if self.fix_first is not None:
            if self.category == 2:
                # fix_first is in physical space {-1,+1}, map to {0,1}
                tokens[:, 0] = (self.fix_first + 1) // 2
            else:
                tokens[:, 0] = self.fix_first

        if T is None:
            T = torch.ones(batch_size, device=self.device)
        T = T.to(self.device)

        cond = self._get_cond(T)
        temp_scale = self._compute_temp_scale(T)

        start_pos = 1 if self.fix_first is not None else 0

        # ── Prefill: process positions 0..start_pos as a batch ──
        # Position 0 always gets zero embedding (shift-right)
        if start_pos > 0:
            # Positions 0..start_pos-1 are fixed; build their shifted input
            # For pos 0: zero embedding, for pos k: embed(token[k-1])
            prefill_len = start_pos
            tok_emb = self.tok_embed(tokens[:, :prefill_len])  # [B, prefill_len, d]
            zero_start = torch.zeros(batch_size, 1, self._d_model, device=self.device)
            if prefill_len > 1:
                shifted = torch.cat([zero_start, tok_emb[:, :-1, :]], dim=1)
            else:
                shifted = zero_start  # Only 1 fixed token
            x = shifted + self.pos_embed[:prefill_len]
            prefill_bias = self._get_rel_pos_bias(prefill_len)
            _, past_kvs = self.backbone(x, cond, attn_bias=prefill_bias)
        else:
            past_kvs = None

        # ── Incremental decoding: one token at a time with KV-cache ──
        for i in range(start_pos, L):
            # Input for position i: embedding of token[i-1] (shift-right)
            if i == 0:
                # First position: zero embedding
                x_i = torch.zeros(batch_size, 1, self._d_model, device=self.device)
            else:
                x_i = self.tok_embed(tokens[:, i-1]).unsqueeze(1)  # [B, 1, d_model]

            x_i = x_i + self.pos_embed[i:i+1]  # [B, 1, d_model]

            # Incremental relative position bias: query=position i, keys=0..i
            incr_bias = self._get_rel_pos_bias(query_len=1, kv_len=i + 1)

            # Run through backbone with KV-cache
            out_i, past_kvs = self.backbone(x_i, cond, past_kvs=past_kvs, attn_bias=incr_bias)
            logits_i = self.output_head(out_i.squeeze(1))  # [B, q]

            if isinstance(temp_scale, torch.Tensor):
                logits_i = logits_i * temp_scale.squeeze(-1)  # [B, q]

            probs = F.softmax(logits_i, dim=-1)
            tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)

        # Reshape to [B, 1, H, W] and apply state mapping
        sample = tokens.reshape(batch_size, 1, H, W).float()
        sample = self.mapping(sample)

        return sample

    def log_prob(self, sample, T=None):
        """
        Compute log probability of a sample.

        Args:
            sample: [B, 1, H, W] in physical representation ({-1,+1} or {0,...,q-1}).
            T: [B] temperature tensor.

        Returns:
            [B, 1] total log probability (sum over all sites, excluding fix_first).
        """
        B = sample.shape[0]

        # Map to internal representation {0, ..., q-1}
        internal = self.reverse_mapping(sample)
        tokens = internal.reshape(B, -1).long()  # [B, L]

        if self.fix_first is not None:
            expected = (self.fix_first if self.category > 2
                        else (self.fix_first + 1) // 2)
            assert (
                tokens[:, 0] == expected
            ).all(), "First token doesn't match fix_first"

        if T is None:
            T = torch.ones(B, device=self.device)
        T = T.to(self.device)

        # Forward pass: get logits for all positions (parallel, no KV-cache)
        logits = self._forward_logits(tokens, T)  # [B, L, q]

        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [B, L, q]

        # Gather log prob of actual tokens
        log_prob_selected = log_probs.gather(
            2, tokens.unsqueeze(-1)
        ).squeeze(-1)  # [B, L]

        # Exclude fix_first position
        if self.fix_first is not None:
            log_prob_selected = log_prob_selected[:, 1:]

        # Sum over sequence
        return log_prob_selected.sum(dim=-1, keepdim=True)  # [B, 1]

    def use_pytorch_mhc(self):
        """No-op for API compatibility with DiscretePixelCNN."""
        pass


# Backward-compatible alias (configs saved as model_transformer.SpinGPT)
SpinGPT = LatticeGPT
