"""
LatticeMamba: Mamba-based autoregressive model for lattice spin systems.

Replaces the Transformer backbone in LatticeGPT with a Mamba SSM while keeping
the same VaTD training objective (variational free energy minimization via
REINFORCE).  Designed as the third architecture witness for the PRL paper's
eRank universality claim, and to enable scaling to L=32/64 lattices where
Transformer attention is infeasible.

Architecture
------------
  Flatten 16×16 → 256-token sequence (raster order)
  Token embedding:        nn.Embedding(q, d_model)
  Position embedding:     nn.Parameter(seq_len, d_model)
  Temperature MLP:        Linear(1, cond_dim) → GELU → Linear(cond_dim, cond_dim)
  MambaBackbone:          N × [AdaLN → MambaBlock → Residual] + final LayerNorm
  Output head:            Linear(d_model, q)

Training:  parallel scan via MambaBlock.forward() — used by log_prob()
Sampling:  recurrent step via MambaBlock.step()  — used by sample()
Default d_state = 32 (per mitigations.md: start with 32, not 16)

Interface is identical to LatticeGPT / DiscretePixelCNN:
  sample(batch_size, T)  → [B, 1, H, W]
  log_prob(sample, T)    → [B, 1]
  use_pytorch_mhc()      → no-op

mambapy version: pinned to 1.2.0 (see requirements.txt)
"""

import torch
from torch import nn
import torch.nn.functional as F

from mambapy.mamba import MambaBlock, MambaConfig


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
        scale_shift = self.proj(cond)              # [B, 2*d_model]
        scale, shift = scale_shift.chunk(2, dim=-1) # each [B, d_model]
        # Unsqueeze for broadcasting over sequence length
        scale = scale.unsqueeze(1)                 # [B, 1, d_model]
        shift = shift.unsqueeze(1)                 # [B, 1, d_model]
        return (1.0 + scale) * self.norm(x) + shift


# ──────────────────────────────────────────────────────────────
# Mamba Backbone with AdaLN temperature conditioning
# ──────────────────────────────────────────────────────────────


class MambaBackbone(nn.Module):
    """
    Stack of MambaBlocks with per-block AdaLN conditioning and manual residuals.

    Each layer computes:
        x = x + MambaBlock(AdaLN(x, cond))

    This bypasses mambapy's ResidualBlock (which uses an internal RMSNorm),
    giving us full control over normalization and residual structure.

    Exposes ``self.blocks`` (nn.ModuleList of MambaBlock) and ``self.adalns``
    (nn.ModuleList of AdaLN) for activation hooks in analyze_rank.py.
    """

    def __init__(self, config: MambaConfig, cond_dim: int):
        super().__init__()
        n_layers = config.n_layers
        self.blocks = nn.ModuleList([
            MambaBlock(config) for _ in range(n_layers)
        ])
        self.adalns = nn.ModuleList([
            AdaLN(config.d_model, cond_dim) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parallel scan mode — used during training (log_prob).

        Args:
            x:    [B, L, d_model]
            cond: [B, cond_dim]
        Returns:
            [B, L, d_model]
        """
        for adaln, block in zip(self.adalns, self.blocks):
            x = x + block(adaln(x, cond))  # manual residual
        return self.final_norm(x)

    def step(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        caches: list,
    ):
        """
        Recurrent step mode — used during sampling (one token at a time).

        Args:
            x:      [B, d_model]  single-token input
            cond:   [B, cond_dim]
            caches: list of (h, inputs) tuples, one per layer
                      h:      [B, d_inner, d_state]
                      inputs: [B, d_inner, d_conv-1]
        Returns:
            x:         [B, d_model]
            new_caches: updated list of (h, inputs) tuples
        """
        new_caches = []
        for i, (adaln, block) in enumerate(zip(self.adalns, self.blocks)):
            # AdaLN expects [B, L, d_model]; unsqueeze L=1, then squeeze back
            x_seq = adaln(x.unsqueeze(1), cond).squeeze(1)  # [B, d_model]
            out, new_cache = block.step(x_seq, caches[i])   # out: [B, d_model]
            x = x + out                                       # manual residual
            new_caches.append(new_cache)
        # Final norm is NOT applied per-step; it is applied after all positions
        # in sample() via a single pass (see LatticeMamba.sample).
        # Actually, applying final_norm per-step is correct for consistency:
        # the norm is position-wise so it can be applied token-by-token.
        return self.final_norm(x.unsqueeze(1)).squeeze(1), new_caches


# ──────────────────────────────────────────────────────────────
# LatticeMamba: Full Model
# ──────────────────────────────────────────────────────────────


class LatticeMamba(nn.Module):
    """
    Mamba-based autoregressive model for lattice spin systems.

    Flattens the 2D lattice into a 1D sequence (raster order),
    applies a Mamba SSM backbone with AdaLN temperature conditioning,
    and predicts per-site categorical distributions.

    Same interface as LatticeGPT / DiscretePixelCNN.

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

        # ── State mapping (same as DiscretePixelCNN / LatticeGPT) ──
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

        # ── Mamba / model hyperparameters ──
        self._d_model = hparams.get("d_model", 128)
        n_layers = hparams.get("n_layers", 4)
        d_state = hparams.get("d_state", 32)          # default 32 per mitigations.md
        expand_factor = hparams.get("expand_factor", 2)
        d_conv = hparams.get("d_conv", 4)
        cond_dim = hparams.get("cond_dim", 64)
        pscan = hparams.get("pscan", True)

        # Store derived dimensions for cache initialization
        self._n_layers = n_layers
        self._d_state = d_state
        self._d_inner = expand_factor * self._d_model  # ED
        self._d_conv = d_conv

        # Build MambaConfig
        # pscan=False uses sequential scan (O(1) memory per step vs O(L) for parallel)
        # Recommended for ≤8GB GPUs with d_state≥32
        mamba_cfg = MambaConfig(
            d_model=self._d_model,
            n_layers=n_layers,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            pscan=pscan,
        )

        # ── Token embedding ──
        self.tok_embed = nn.Embedding(self.category, self._d_model)

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

        # ── Mamba backbone ──
        self.backbone = MambaBackbone(mamba_cfg, cond_dim)

        # ── Output head ──
        self.output_head = nn.Linear(self._d_model, self.category)

        self.to(device)

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _compute_temp_scale(self, T: torch.Tensor):
        """
        Temperature-dependent logit scaling (same as LatticeGPT).

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

    def _init_caches(self, batch_size: int) -> list:
        """
        Initialize Mamba recurrent caches for autoregressive sampling.

        Returns list of (h, inputs) tuples, one per layer:
          h:      None  (mambapy interprets None as zero initial state)
          inputs: [B, d_inner, d_conv-1]  zero-initialized conv buffer
        """
        caches = []
        for _ in range(self._n_layers):
            h = None  # mambapy's ssm_step handles None → zeros internally
            inputs = torch.zeros(
                batch_size, self._d_inner, self._d_conv - 1,
                device=self.device
            )
            caches.append((h, inputs))
        return caches

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
        tok_emb = self.tok_embed(tokens)  # [B, L, d_model]

        # Shift right: position 0 gets zero embedding, position i gets embed(token[i-1])
        zero_start = torch.zeros(B, 1, tok_emb.shape[-1], device=tok_emb.device)
        tok_emb = torch.cat([zero_start, tok_emb[:, :-1, :]], dim=1)  # [B, L, d_model]

        # Add positional embedding
        x = tok_emb + self.pos_embed[:L]  # [B, L, d_model]

        # Temperature conditioning
        cond = self._get_cond(T)  # [B, cond_dim]

        # Mamba backbone — parallel scan mode
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
        Autoregressive sampling using Mamba recurrent step mode.

        At each position, a single token is processed via MambaBlock.step(),
        which maintains the SSM hidden state h and the conv input buffer.
        This is O(L) total (constant per step) vs O(L²) for Transformers.

        Args:
            batch_size: number of samples.
            T:          [B] temperature tensor.
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

        # Initialize recurrent caches (zero state)
        caches = self._init_caches(batch_size)

        # ── Prefill: process positions 0..start_pos-1 to warm up the SSM state ──
        # These positions are fixed; we step through them to update the caches
        # but discard the outputs (we already know what those tokens are).
        for i in range(start_pos):
            # Input for position i: shift-right means we feed embed(token[i-1])
            # For i==0: zero embedding (no previous token)
            if i == 0:
                x_i = torch.zeros(batch_size, self._d_model, device=self.device)
            else:
                x_i = self.tok_embed(tokens[:, i - 1])  # [B, d_model]

            x_i = x_i + self.pos_embed[i]  # [B, d_model] — broadcast add

            # Step through backbone, updating caches; discard output
            _, caches = self.backbone.step(x_i, cond, caches)

        # ── Autoregressive decoding: sample one token per position ──
        for i in range(start_pos, L):
            # Input for position i: embed(token[i-1]) shifted right
            if i == 0:
                x_i = torch.zeros(batch_size, self._d_model, device=self.device)
            else:
                x_i = self.tok_embed(tokens[:, i - 1])  # [B, d_model]

            x_i = x_i + self.pos_embed[i]  # [B, d_model]

            # Recurrent step through backbone
            out_i, caches = self.backbone.step(x_i, cond, caches)  # [B, d_model]

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

        # Forward pass: get logits for all positions (parallel scan, no recurrence)
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


# Backward-compatible alias (configs saved as model_mamba.SpinMamba)
SpinMamba = LatticeMamba
