"""
Spin-GPT: Autoregressive Transformer for lattice spin systems.

Replaces the PixelCNN backbone with a causal Transformer while keeping
the same VaTD training objective (variational free energy minimization
via REINFORCE).  Designed to test the architecture independence of the
low-rank phenomenon observed in DiscretePixelCNN.

Architecture
------------
  Flatten 16×16 → 256-token sequence (raster order)
  Token embedding: nn.Embedding(q, d_model)
  Position embedding: learnable nn.Parameter([L², d_model])
  Temperature conditioning: AdaLN in every block
  Causal self-attention with KV-cache for O(L²) sampling
  Output: nn.Linear(d_model, q) → per-position logits

Interface is identical to DiscretePixelCNN:
  sample(batch_size, T) → [B, 1, H, W]
  log_prob(sample, T)   → [B, 1]
"""

import torch
from torch import nn
import torch.nn.functional as F
import math


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

    Supports KV-cache for incremental decoding: when past_kv is provided,
    only the new query token is computed and the cache is extended.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, cond_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.adaln1 = AdaLN(d_model, cond_dim)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.adaln2 = AdaLN(d_model, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, cond, past_kv=None):
        """
        Args:
            x: [B, L, d_model] (full sequence) or [B, 1, d_model] (incremental)
            cond: [B, cond_dim]
            past_kv: optional tuple (past_key, past_value), each [B, n_kv, d_model]

        Returns:
            output: [B, L, d_model] or [B, 1, d_model]
            new_kv: tuple (key, value) for the full sequence so far
        """
        # Self-attention with causal mask
        h = self.adaln1(x, cond)

        if past_kv is None:
            # Full sequence: standard causal attention
            L = h.shape[1]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                L, device=h.device, dtype=h.dtype
            )
            attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, is_causal=True)
            new_kv = (h, h)  # Store full KV for future incremental steps
        else:
            # Incremental: h is [B, 1, d_model], attend to all past + current
            past_k, past_v = past_kv
            full_k = torch.cat([past_k, h], dim=1)  # [B, n_kv+1, d_model]
            full_v = torch.cat([past_v, h], dim=1)
            # No mask needed: query is single token, attends to all past (causal by construction)
            attn_out, _ = self.attn(h, full_k, full_v)
            new_kv = (full_k, full_v)

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

    def forward(self, x, cond, past_kvs=None):
        """
        Args:
            x: [B, L, d_model] or [B, 1, d_model]
            cond: [B, cond_dim]
            past_kvs: optional list of (key, value) tuples, one per block

        Returns:
            x: [B, L, d_model] or [B, 1, d_model]
            new_kvs: list of (key, value) tuples
        """
        new_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, kv = block(x, cond, past_kv=past_kv)
            new_kvs.append(kv)
        return self.final_norm(x), new_kvs


# ──────────────────────────────────────────────────────────────
# SpinGPT: Full Model
# ──────────────────────────────────────────────────────────────


class SpinGPT(nn.Module):
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

        # ── Token and position embeddings ──
        self.tok_embed = nn.Embedding(self.category, self._d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(self.seq_len, self._d_model) * 0.02
        )

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

        # Transformer backbone (no KV-cache for training)
        x, _ = self.backbone(x, cond)  # [B, L, d_model]

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
            _, past_kvs = self.backbone(x, cond)
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

            # Run through backbone with KV-cache
            out_i, past_kvs = self.backbone(x_i, cond, past_kvs=past_kvs)
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
