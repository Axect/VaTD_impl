"""
LatticeFLA: Flash Linear Attention-based autoregressive model for lattice spin systems.

Replaces the Transformer backbone in LatticeGPT with a Flash Linear Attention (FLA)
layer while keeping the same VaTD training objective (variational free energy
minimization via REINFORCE).  Designed as an additional architecture witness for the
PRL paper's eRank universality claim, and to enable O(L) sampling complexity vs O(L²)
for standard Transformers.

Architecture
------------
  Flatten 16×16 → 256-token sequence (raster order)
  Token embedding:        nn.Embedding(q, d_model)
  Position embedding:     nn.Parameter(seq_len, d_model)
  Temperature MLP:        Linear(1, cond_dim) → GELU → Linear(cond_dim, cond_dim)
  FLABackbone:            N × [AdaLN → FLA layer → Residual] + final LayerNorm
  Output head:            Linear(d_model, q)

Training:  parallel chunk mode via FLABlock.forward() — used by log_prob()
Sampling:  recurrent step via FLABlock.forward(use_cache=True) — used by sample()

Supported FLA layers (via `fla_layer` hparam):
  "gla"             — Gated Linear Attention (fla.layers.GatedLinearAttention)
  "gated_deltanet"  — Gated DeltaNet       (fla.layers.GatedDeltaNet)

Interface is identical to LatticeGPT / LatticeMamba:
  sample(batch_size, T)  → [B, 1, H, W]
  log_prob(sample, T)    → [B, 1]
  use_pytorch_mhc()      → no-op
"""

import torch
from torch import nn
import torch.nn.functional as F


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
            x:    [B, L, d_model]
            cond: [B, cond_dim]
        Returns:
            [B, L, d_model]
        """
        scale_shift = self.proj(cond)               # [B, 2*d_model]
        scale, shift = scale_shift.chunk(2, dim=-1)  # each [B, d_model]
        # Unsqueeze for broadcasting over sequence length
        scale = scale.unsqueeze(1)                   # [B, 1, d_model]
        shift = shift.unsqueeze(1)                   # [B, 1, d_model]
        return (1.0 + scale) * self.norm(x) + shift


# ──────────────────────────────────────────────────────────────
# FLABlock: Single block with AdaLN + FLA layer + Residual
# ──────────────────────────────────────────────────────────────


class FLABlock(nn.Module):
    """
    Single FLA block: AdaLN → FLA layer → Residual.

    Wraps any Flash Linear Attention layer (GLA, GatedDeltaNet, etc.) with
    AdaLN temperature conditioning and a manual residual connection.

    Each layer computes:
        x = x + fla_layer(AdaLN(x, cond))

    Exposes ``self.layer`` (the fla layer) for activation hooks in analyze_rank.py.
    The forward method returns a tuple (output, cache) so that analyze_rank.py's
    existing hook code (which handles tuples via ``out[0] if isinstance(out, tuple) else out``)
    works correctly.
    """

    def __init__(self, fla_layer: nn.Module, d_model: int, cond_dim: int):
        super().__init__()
        self.adaln = AdaLN(d_model, cond_dim)
        self.layer = fla_layer

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Args:
            x:               [B, L, d_model]
            cond:            [B, cond_dim]
            past_key_values: fla Cache object or None
            use_cache:       bool — whether to return updated cache
        Returns:
            output:          [B, L, d_model]
            cache:           updated Cache (or None when use_cache=False)
        """
        h = self.adaln(x, cond)
        out, _, cache = self.layer(h, past_key_values=past_key_values, use_cache=use_cache)
        return x + out, cache


# ──────────────────────────────────────────────────────────────
# FLABackbone: Stack of FLABlocks with temperature conditioning
# ──────────────────────────────────────────────────────────────


class FLABackbone(nn.Module):
    """
    Stack of FLABlocks with per-block AdaLN conditioning and final LayerNorm.

    Exposes ``self.blocks`` (nn.ModuleList of FLABlock) for activation hooks
    in analyze_rank.py.

    Two operating modes:
      forward() — parallel chunk mode (training / log_prob)
      step()    — recurrent mode with fla Cache (sampling)

    Note on fla Cache routing: fla layers use ``layer_idx`` internally to select
    which slot in the Cache object to read/write.  All FLABlocks share the same
    Cache object; each FLABlock's underlying fla layer writes to its own slot
    keyed by its ``layer_idx``.
    """

    def __init__(
        self,
        fla_layer_cls,
        fla_layer_kwargs: dict,
        n_layers: int,
        d_model: int,
        cond_dim: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            FLABlock(
                fla_layer_cls(layer_idx=i, **fla_layer_kwargs),
                d_model,
                cond_dim,
            )
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parallel chunk mode — used during training (log_prob).

        Args:
            x:    [B, L, d_model]
            cond: [B, cond_dim]
        Returns:
            [B, L, d_model]
        """
        for block in self.blocks:
            x, _ = block(x, cond, use_cache=False)
        return self.final_norm(x)

    def step(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        past_key_values,
    ):
        """
        Recurrent step mode — used during sampling (one token at a time).

        Args:
            x:               [B, 1, d_model]  single-token input (sequence dim kept as 1)
            cond:            [B, cond_dim]
            past_key_values: fla Cache object
        Returns:
            output:          [B, d_model]  (squeezed, with final_norm applied)
            past_key_values: updated Cache object
        """
        for block in self.blocks:
            x, past_key_values = block(
                x, cond, past_key_values=past_key_values, use_cache=True
            )
        # x is [B, 1, d_model]; apply final_norm and squeeze the sequence dim
        return self.final_norm(x).squeeze(1), past_key_values


# ──────────────────────────────────────────────────────────────
# Layer factory helpers
# ──────────────────────────────────────────────────────────────


def _create_fla_layer(hparams: dict):
    """
    Create the FLA layer class and kwargs dict from hparams.

    Supported values of hparams['fla_layer']:
      "gla"             — GatedLinearAttention
      "gated_deltanet"  — GatedDeltaNet

    Returns:
        cls:    the layer class (not instantiated)
        kwargs: dict of constructor kwargs (excluding layer_idx)
    """
    layer_type = hparams.get("fla_layer", "gla")

    if layer_type == "gla":
        from fla.layers import GatedLinearAttention
        cls = GatedLinearAttention
        kwargs = dict(
            mode="chunk",
            hidden_size=hparams["d_model"],
            num_heads=hparams["num_heads"],
            expand_k=hparams.get("expand_k", 1.0),
            expand_v=hparams.get("expand_v", 1.0),
            use_short_conv=hparams.get("use_short_conv", True),
            conv_size=hparams.get("conv_size", 4),
            use_output_gate=hparams.get("use_output_gate", True),
            gate_low_rank_dim=hparams.get("gate_low_rank_dim", 16),
        )

    elif layer_type == "gated_deltanet":
        from fla.layers import GatedDeltaNet
        cls = GatedDeltaNet
        kwargs = dict(
            mode="chunk",
            hidden_size=hparams["d_model"],
            head_dim=hparams.get("head_dim", 32),
            num_heads=hparams["num_heads"],
            expand_v=hparams.get("expand_v", 1.0),
            use_gate=hparams.get("use_gate", True),
            use_short_conv=hparams.get("use_short_conv", True),
            conv_size=hparams.get("conv_size", 4),
        )

    else:
        raise ValueError(
            f"Unknown fla_layer type '{layer_type}'. "
            "Supported: 'gla', 'gated_deltanet'."
        )

    return cls, kwargs


# ──────────────────────────────────────────────────────────────
# LatticeFLA: Full Model
# ──────────────────────────────────────────────────────────────


class LatticeFLA(nn.Module):
    """
    Flash Linear Attention-based autoregressive model for lattice spin systems.

    Flattens the 2D lattice into a 1D sequence (raster order),
    applies a FLA backbone with AdaLN temperature conditioning,
    and predicts per-site categorical distributions.

    Same interface as LatticeGPT / LatticeMamba.

    Args:
        hparams: dict with keys matching net_config in YAML configs.
        device:  torch device string.
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

        # ── State mapping (same as DiscretePixelCNN / LatticeGPT / LatticeMamba) ──
        if self.category == 2:
            self.mapping = lambda x: 2 * x - 1         # {0,1} → {-1,+1}
            self.reverse_mapping = lambda x: torch.div(x + 1, 2, rounding_mode="trunc")
        else:
            self.mapping = lambda x: x                  # Potts: keep as-is
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

        # ── FLA / model hyperparameters ──
        self._d_model = hparams.get("d_model", 128)
        n_layers = hparams.get("n_layers", 4)
        cond_dim = hparams.get("cond_dim", 64)

        # ── Token embedding ──
        self.tok_embed = nn.Embedding(self.category, self._d_model)

        # ── Circular embedding (optional, for clock model) ──
        self.circular_embedding = hparams.get("circular_embedding", False)
        if self.circular_embedding:
            self.register_buffer(
                "_circ_angles",
                2.0 * torch.pi * torch.arange(self.category, dtype=torch.float32) / self.category
            )
            self.circ_proj = nn.Linear(2, self._d_model)

        # ── Learnable absolute positional embedding ──
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

        # ── FLA backbone ──
        fla_layer_cls, fla_layer_kwargs = _create_fla_layer(hparams)
        self.backbone = FLABackbone(
            fla_layer_cls=fla_layer_cls,
            fla_layer_kwargs=fla_layer_kwargs,
            n_layers=n_layers,
            d_model=self._d_model,
            cond_dim=cond_dim,
        )

        # ── Output head ──
        self.output_head = nn.Linear(self._d_model, self.category)

        self.to(device)

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _embed_tokens(self, tokens):
        """Embed tokens with optional circular structure for clock model."""
        if self.circular_embedding:
            angles = self._circ_angles[tokens]
            circ = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            return self.circ_proj(circ)
        return self.tok_embed(tokens)

    def _compute_temp_scale(self, T: torch.Tensor):
        """
        Temperature-dependent logit scaling (same as LatticeGPT / LatticeMamba).

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

    def _get_cond(self, T: torch.Tensor) -> torch.Tensor:
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

    # ──────────────────────────────────────────────────────────
    # Forward methods
    # ──────────────────────────────────────────────────────────

    def _forward_logits(self, tokens: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute per-position logits from a token sequence (parallel / training path).

        Shift-right: position i receives token i-1, so logits at position i
        predict token i from context 0..i-1.

        Args:
            tokens: [B, L] integer token indices in {0, ..., q-1}.
            T:      [B] temperature tensor.
        Returns:
            logits: [B, L, q] unnormalized log-probabilities.
        """
        B, L = tokens.shape

        # Embed tokens
        tok_emb = self._embed_tokens(tokens)  # [B, L, d_model]

        # Shift right: position 0 gets zero embedding, position i gets embed(token[i-1])
        zero_start = torch.zeros(B, 1, tok_emb.shape[-1], device=tok_emb.device)
        tok_emb = torch.cat([zero_start, tok_emb[:, :-1, :]], dim=1)  # [B, L, d_model]

        # Add positional embedding
        x = tok_emb + self.pos_embed[:L]  # [B, L, d_model]

        # Temperature conditioning
        cond = self._get_cond(T)  # [B, cond_dim]

        # FLA backbone — parallel chunk mode
        x = self.backbone(x, cond)  # [B, L, d_model]

        # Output logits
        logits = self.output_head(x)  # [B, L, q]

        # Temperature-dependent scaling
        temp_scale = self._compute_temp_scale(T)
        if isinstance(temp_scale, torch.Tensor):
            logits = logits * temp_scale

        return logits

    @torch.no_grad()
    def sample(self, batch_size=None, T=None) -> torch.Tensor:
        """
        Autoregressive sampling using FLA recurrent step mode.

        At each position, a single token is processed via backbone.step(),
        which maintains the FLA layer's recurrent state via a shared Cache object.
        This is O(L) total (constant per step) vs O(L²) for Transformers.

        Args:
            batch_size: number of samples.
            T:          [B] temperature tensor.
        Returns:
            [B, 1, H, W] samples in physical representation ({-1,+1} or {0,...,q-1}).
        """
        from fla.models.utils import Cache

        if batch_size is None:
            batch_size = self.batch_size

        H, W = self.size
        L = self.seq_len

        tokens = torch.zeros(batch_size, L, dtype=torch.long, device=self.device)

        # Fix first spin if configured
        if self.fix_first is not None:
            if self.category == 2:
                # fix_first is in physical space {-1,+1}, map to internal {0,1}
                tokens[:, 0] = (self.fix_first + 1) // 2
            else:
                tokens[:, 0] = self.fix_first

        if T is None:
            T = torch.ones(batch_size, device=self.device)
        T = T.to(self.device)

        cond = self._get_cond(T)               # [B, cond_dim]
        temp_scale = self._compute_temp_scale(T)

        start_pos = 1 if self.fix_first is not None else 0

        # Initialize fla Cache — shared across all FLABlocks.
        # Each block's fla layer uses its own layer_idx to route to the correct slot.
        past_key_values = Cache()

        # ── Prefill: process positions 0..start_pos-1 to warm up the FLA state ──
        # These positions are fixed; we step through them to update the cache
        # but discard the outputs (we already know what those tokens are).
        for i in range(start_pos):
            # Input for position i: shift-right means we feed embed(token[i-1])
            # For i==0: zero embedding (no previous token)
            if i == 0:
                x_i = torch.zeros(batch_size, 1, self._d_model, device=self.device)
            else:
                x_i = self._embed_tokens(tokens[:, i - 1]).unsqueeze(1)  # [B, 1, d_model]

            x_i = x_i + self.pos_embed[i]  # [B, 1, d_model] — broadcast add

            # Step through backbone, updating cache; discard output
            _, past_key_values = self.backbone.step(x_i, cond, past_key_values)

        # ── Autoregressive decoding: sample one token per position ──
        for i in range(start_pos, L):
            # Input for position i: embed(token[i-1]) shifted right
            if i == 0:
                x_i = torch.zeros(batch_size, 1, self._d_model, device=self.device)
            else:
                x_i = self._embed_tokens(tokens[:, i - 1]).unsqueeze(1)  # [B, 1, d_model]

            x_i = x_i + self.pos_embed[i]  # [B, 1, d_model]

            # Recurrent step through backbone; out_i is [B, d_model] after squeeze
            out_i, past_key_values = self.backbone.step(x_i, cond, past_key_values)

            # Project to logits and sample
            logits_i = self.output_head(out_i)  # [B, q]

            if isinstance(temp_scale, torch.Tensor):
                logits_i = logits_i * temp_scale.squeeze(-1)  # [B, q]

            probs = F.softmax(logits_i, dim=-1)
            tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)

        # Reshape to [B, 1, H, W] and apply state mapping
        sample = tokens.reshape(batch_size, 1, H, W).float()
        sample = self.mapping(sample)

        return sample

    def log_prob(self, sample: torch.Tensor, T=None) -> torch.Tensor:
        """
        Compute log probability of a sample.

        Args:
            sample: [B, 1, H, W] in physical representation ({-1,+1} or {0,...,q-1}).
            T:      [B] temperature tensor.
        Returns:
            [B, 1] total log probability (sum over all sites, excluding fix_first).
        """
        B = sample.shape[0]

        # Map to internal representation {0, ..., q-1}
        internal = self.reverse_mapping(sample)
        tokens = internal.reshape(B, -1).long()  # [B, L]

        if self.fix_first is not None:
            expected = (
                self.fix_first if self.category > 2
                else (self.fix_first + 1) // 2
            )
            assert (
                tokens[:, 0] == expected
            ).all(), "First token doesn't match fix_first"

        if T is None:
            T = torch.ones(B, device=self.device)
        T = T.to(self.device)

        # Forward pass: get logits for all positions (parallel chunk, no recurrence)
        logits = self._forward_logits(tokens, T)  # [B, L, q]

        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [B, L, q]

        # Gather log prob of the actual tokens at each position
        log_prob_selected = log_probs.gather(
            2, tokens.unsqueeze(-1)
        ).squeeze(-1)  # [B, L]

        # Exclude fix_first position (its probability is 1 by construction)
        if self.fix_first is not None:
            log_prob_selected = log_prob_selected[:, 1:]

        # Sum over sequence
        return log_prob_selected.sum(dim=-1, keepdim=True)  # [B, 1]

    def use_pytorch_mhc(self):
        """No-op for API compatibility with DiscretePixelCNN."""
        pass


# Backward-compatible alias (configs saved as model_fla.SpinFLA)
SpinFLA = LatticeFLA
