"""
Discrete Flow Matching model for 2D Ising Model.

This module implements a CTMC-based discrete flow matching approach for
learning the Boltzmann distribution of the 2D Ising model.

Key components:
- SinusoidalTimeEmbedding: Time embedding for flow matching
- ResConv2D: ResNet backbone without autoregressive masking
- DiscreteFlowMatcher: Main model with sample() and log_prob() interface

References:
- Discrete Flow Matching (Gat et al., NeurIPS 2024)
- Flow Matching with General Discrete Paths (2024)
"""

import torch
from torch import nn
import torch.nn.functional as F
import einops
import numpy as np
import math
from typing import Tuple, Optional


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for continuous time t in [0, T_max].

    Uses the same formulation as Transformer positional encodings,
    adapted for scalar time values.

    Args:
        dim: Embedding dimension (must be even)
        max_period: Maximum period for the sinusoidal functions
    """

    def __init__(self, dim: int = 64, max_period: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.dim = dim
        self.max_period = max_period

        # Precompute frequency bands
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time embedding.

        Args:
            t: Time values of shape (B,) or (B, 1)

        Returns:
            Embedding of shape (B, dim)
        """
        if t.dim() == 2:
            t = t.squeeze(-1)

        # (B, half_dim)
        args = t.unsqueeze(-1) * self.freqs.to(t.device)

        # (B, dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


class ResConv2D(nn.Module):
    """
    ResNet backbone for flow matching without autoregressive masking.

    Unlike MaskedResConv2D, this allows full receptive field over all spatial
    positions, which is appropriate for flow matching where we predict
    all positions simultaneously.

    Architecture:
    - First conv: (in_channels + time_channels) -> 2*hidden_channels
    - Residual blocks with 1x1 -> 3x3 -> 1x1 convolutions
    - FC layers (1x1 convs)
    - Final output: category * channel logits

    Args:
        channel: Number of data channels (1 for Ising)
        category: Number of categories per channel (2 for binary spin)
        time_dim: Dimension of time embedding
        hidden_channels: Width of residual blocks
        hidden_conv_layers: Number of residual blocks
        hidden_kernel_size: Kernel size for spatial convolutions
        hidden_width: Width of FC layers
        hidden_fc_layers: Number of FC layers
        augment_channels: Additional input channels (e.g., temperature)
    """

    def __init__(
        self,
        channel: int = 1,
        category: int = 2,
        time_dim: int = 64,
        hidden_channels: int = 64,
        hidden_conv_layers: int = 5,
        hidden_kernel_size: int = 3,
        hidden_width: int = 128,
        hidden_fc_layers: int = 2,
        augment_channels: int = 1,  # Temperature
    ):
        super().__init__()

        self.channel = channel
        self.category = category
        self.time_dim = time_dim

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_channels * 2),
            nn.GELU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )

        # Input channels: data (as probabilities) + temperature
        # For flow matching, input is probability simplex (B, 2, H, W) + T channel
        in_channels = category * channel + augment_channels

        self.first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * hidden_channels,
            kernel_size=7,
            padding=3,
        )

        # Residual blocks
        hidden_convs = []
        for _ in range(hidden_conv_layers):
            hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(2 * hidden_channels, hidden_channels, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=hidden_kernel_size,
                        padding=(hidden_kernel_size - 1) // 2,
                    ),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, 2 * hidden_channels, kernel_size=1),
                )
            )
        self.hidden_convs = nn.ModuleList(hidden_convs)

        # FC layers
        self.first_fc = nn.Conv2d(2 * hidden_channels, hidden_width, kernel_size=1)

        hidden_fcs = []
        for _ in range(hidden_fc_layers):
            hidden_fcs.append(nn.Conv2d(hidden_width, hidden_width, kernel_size=1))
        self.hidden_fcs = nn.ModuleList(hidden_fcs)

        # Final output
        self.final_fc = nn.Conv2d(hidden_width, category * channel, kernel_size=1)

        # Initialize biases to zero for symmetry
        self._init_zero_bias()

    def _init_zero_bias(self):
        """Initialize all biases to 0 for spin symmetry."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)
               Contains probability simplex + temperature channel
            t_emb: Time embedding of shape (B, time_dim)

        Returns:
            Logits of shape (B, category, channel, H, W)
        """
        B, _, H, W = x.shape

        # Process time embedding
        t_proj = self.time_mlp(t_emb)  # (B, 2*hidden_channels)
        t_proj = t_proj.unsqueeze(-1).unsqueeze(-1)  # (B, 2*hidden_channels, 1, 1)

        # First convolution
        h = self.first_conv(x)
        h = F.gelu(h)

        # Add time embedding (broadcast over spatial dimensions)
        h = h + t_proj

        # Residual blocks
        for layer in self.hidden_convs:
            h = h + F.gelu(layer(h))

        # FC layers
        h = self.first_fc(h)
        h = F.gelu(h)

        for layer in self.hidden_fcs:
            h = layer(h)
            h = F.gelu(h)

        # Final output
        out = self.final_fc(h)  # (B, category*channel, H, W)

        # Reshape to (B, category, channel, H, W)
        out = out.reshape(B, self.category, self.channel, H, W)

        return out


class DiscreteFlowMatcher(nn.Module):
    """
    Discrete Flow Matching model for Ising spin systems.

    Implements a CTMC-based flow matching approach that learns to transform
    a uniform distribution into the target Boltzmann distribution.

    Training: Cross-entropy denoising + REINFORCE for energy guidance
    Sampling: Euler integration on probability simplex
    Partition Function: Thermodynamic integration or ELBO

    Args:
        hparams: Dictionary containing model hyperparameters
        device: Device to place the model on
    """

    def __init__(self, hparams: dict, device: str = "cpu"):
        super().__init__()
        self.hparams = hparams
        self.device = device

        # Lattice configuration
        self.channel = 1  # Single channel
        self.category = 2  # Binary spin {0, 1} mapped to {-1, +1}

        size = hparams.get("size", 16)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

        self.fix_first = hparams.get("fix_first", 1)

        # Training parameters
        self.batch_size = hparams["batch_size"]
        self.num_beta = hparams["num_beta"]
        self.beta_min = hparams["beta_min"]
        self.beta_max = hparams["beta_max"]

        # Flow matching parameters
        self.num_flow_steps = hparams.get("num_flow_steps", 80)
        self.t_max = hparams.get("t_max", 9.0)  # Maximum integration time
        self.t_min = hparams.get("t_min", 0.01)  # Avoid t=0 singularity
        self.mh_model_init_prob = hparams.get("mh_model_init_prob", 0.0)

        # Curriculum learning (same as PixelCNN)
        self.curriculum_enabled = hparams.get("curriculum_enabled", False)
        self.curriculum_warmup_epochs = hparams.get("curriculum_warmup_epochs", 50)
        self.curriculum_start_beta_max = hparams.get(
            "curriculum_start_beta_max", self.beta_min * 1.5
        )
        self.phase1_epochs = hparams.get("phase1_epochs", 50)
        self.phase1_beta_max = hparams.get("phase1_beta_max", 0.35)
        self.phase2_epochs = hparams.get("phase2_epochs", 100)

        # Temperature scaling (optional, for compatibility)
        self.logit_temp_scale = hparams.get("logit_temp_scale", False)
        self.temp_ref = hparams.get("temp_ref", 2.27)
        self.temp_scale_power = hparams.get("temp_scale_power", 0.5)
        self.temp_scale_min = hparams.get("temp_scale_min", 0.1)
        self.temp_scale_max = hparams.get("temp_scale_max", 10.0)

        # Spin mappings
        self.mapping = lambda x: 2 * x - 1  # {0,1} -> {-1,+1}
        self.reverse_mapping = lambda x: torch.div(x + 1, 2, rounding_mode="trunc")

        # Time embedding
        time_dim = hparams.get("time_dim", 64)
        self.time_embedding = SinusoidalTimeEmbedding(dim=time_dim)

        # Main network
        self.net = ResConv2D(
            channel=self.channel,
            category=self.category,
            time_dim=time_dim,
            hidden_channels=hparams.get("hidden_channels", 64),
            hidden_conv_layers=hparams.get("hidden_conv_layers", 5),
            hidden_kernel_size=hparams.get("hidden_kernel_size", 3),
            hidden_width=hparams.get("hidden_width", 128),
            hidden_fc_layers=hparams.get("hidden_fc_layers", 2),
            augment_channels=1,  # Temperature
        )

    def to(self, *args, **kwargs):
        """Override to() to update self.device."""
        self = super().to(*args, **kwargs)
        if args and isinstance(args[0], (torch.device, str)):
            self.device = args[0]
        elif "device" in kwargs:
            self.device = kwargs["device"]
        return self

    def _compute_temp_scale(self, T: torch.Tensor) -> torch.Tensor:
        """Compute temperature-dependent scaling (for compatibility)."""
        if not self.logit_temp_scale:
            return 1.0

        if T.dim() == 1:
            T = T.unsqueeze(1)

        scale = (self.temp_ref / T) ** self.temp_scale_power
        scale = scale.clamp(min=self.temp_scale_min, max=self.temp_scale_max)
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return scale

    def apply_dirichlet_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Dirichlet noising along the probability path.

        The probability path interpolates from uniform at t=0 to
        delta function at t=infinity:

            p_t(y | x) = Dir(y; alpha = 1 + t * onehot(x))

        Args:
            x: Clean samples of shape (B, 1, H, W) in {-1, +1}
            t: Time values of shape (B,) in [0, t_max]

        Returns:
            Noisy samples on probability simplex (B, 2, H, W)
        """
        B, C, H, W = x.shape

        # Convert {-1, +1} to {0, 1}
        x_idx = self.reverse_mapping(x).long()  # (B, 1, H, W)

        # Create one-hot encoding: (B, 2, H, W)
        x_onehot = F.one_hot(x_idx.squeeze(1), num_classes=2)  # (B, H, W, 2)
        x_onehot = x_onehot.permute(0, 3, 1, 2).float()  # (B, 2, H, W)

        # Dirichlet parameters: alpha = 1 + t * onehot(x)
        t_exp = t.view(B, 1, 1, 1)  # (B, 1, 1, 1)
        alpha = 1.0 + t_exp * x_onehot  # (B, 2, H, W)

        # Sample from Dirichlet for each spatial position
        # Flatten spatial dimensions for sampling
        alpha_flat = alpha.permute(0, 2, 3, 1).reshape(-1, 2)  # (B*H*W, 2)

        # Dirichlet sampling
        x_noisy_flat = torch.distributions.Dirichlet(alpha_flat).sample()

        # Reshape back
        x_noisy = x_noisy_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)  # (B, 2, H, W)

        return x_noisy

    def velocity_field(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        T: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the probability velocity field for Discrete Flow Matching.

        For Dirichlet probability path p_t(y|x) = Dir(y; 1 + t·e_x),
        the conditional velocity is:
            u_t(y|x) = (e_x - y) / (2 + t)

        The network predicts the target distribution, and we compute
        the properly scaled velocity.

        Args:
            x: Current state on probability simplex (B, 2, H, W)
            t: Time values (B,)
            T: Temperature values (B,)

        Returns:
            Velocity field (B, 2, H, W)
        """
        B, _, H, W = x.shape

        # Time embedding
        t_emb = self.time_embedding(t)  # (B, time_dim)

        # Temperature as additional channel
        T_channel = T.view(B, 1, 1, 1).expand(B, 1, H, W)  # (B, 1, H, W)

        # Concatenate: (B, 3, H, W) = (prob_0, prob_1, T)
        x_input = torch.cat([x, T_channel], dim=1)

        # Network forward pass -> logits (B, 2, 1, H, W)
        logits = self.net(x_input, t_emb)  # (B, 2, 1, H, W)
        logits = logits.squeeze(2)  # (B, 2, H, W)

        # Apply temperature scaling if enabled
        if self.logit_temp_scale:
            temp_scale = self._compute_temp_scale(T)
            logits = logits * temp_scale.squeeze(-1)  # Adjust shape

        # Convert logits to velocity with proper time scaling
        # For Dirichlet path: u_t(y|x) = (e_x - y) / (2 + t)
        target_prob = F.softmax(logits, dim=1)

        # Time scaling factor: 1 / (2 + t)
        time_scale = 1.0 / (2.0 + t.view(B, 1, 1, 1))

        velocity = (target_prob - x) * time_scale

        return velocity

    def compute_energy_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        T: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy-guided velocity using mean-field Boltzmann approximation.

        For Ising model, the local field at position (i,j) is:
            h_{i,j} = J * sum of neighbor spins

        The Boltzmann probability for spin +1 given local field h:
            p(s=+1) = sigmoid(2 * beta * h)

        The energy velocity points from current state toward Boltzmann equilibrium,
        with proper time scaling for Dirichlet flow matching.

        Args:
            x: Current state on probability simplex (B, 2, H, W)
               x[:, 0] = p(spin=-1), x[:, 1] = p(spin=+1)
            t: Time values (B,) for proper scaling
            T: Temperature values (B,)

        Returns:
            Energy-guided velocity (B, 2, H, W)
        """
        B, _, H, W = x.shape
        beta = (1.0 / T).view(B, 1, 1, 1)

        # Expected spin from current probabilities: <s> = 2*p(+1) - 1
        # Range: [-1, +1]
        expected_spin = 2 * x[:, 1:2] - 1  # (B, 1, H, W)

        # Local field = sum of neighbor expected spins (Ising coupling J=1)
        # Using convolution with periodic boundary conditions
        kernel = torch.tensor(
            [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]],
            dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)

        # Circular padding for periodic boundaries
        padded = F.pad(expected_spin, (1, 1, 1, 1), mode='circular')
        local_field = F.conv2d(padded, kernel)  # (B, 1, H, W)

        # Boltzmann probability for spin +1: p(+1) = sigmoid(2β * h)
        # Energy for spin s at field h is E = -s*h
        # So p(+1) ∝ exp(β*h), p(-1) ∝ exp(-β*h)
        # p(+1) = exp(βh) / (exp(βh) + exp(-βh)) = sigmoid(2βh)
        p_boltz_plus = torch.sigmoid(2 * beta * local_field)  # (B, 1, H, W)

        # Target Boltzmann distribution
        p_boltz = torch.cat([1 - p_boltz_plus, p_boltz_plus], dim=1)  # (B, 2, H, W)

        # Velocity toward Boltzmann equilibrium with time scaling
        time_scale = 1.0 / (2.0 + t.view(B, 1, 1, 1))
        v_energy = (p_boltz - x) * time_scale

        return v_energy

    def velocity_field_with_energy(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        T: torch.Tensor,
        energy_weight: float = 1.0,
        time_dependent_weight: bool = True
    ) -> torch.Tensor:
        """
        Combined velocity: learned + energy-guided.

        v_total = v_learned + λ(t) * v_energy

        Where λ(t) increases with time (stronger guidance near the end).

        Args:
            x: Current state on probability simplex (B, 2, H, W)
            t: Time values (B,)
            T: Temperature values (B,)
            energy_weight: Base weight for energy velocity
            time_dependent_weight: If True, λ(t) = energy_weight * (t/t_max)

        Returns:
            Combined velocity (B, 2, H, W)
        """
        # Learned velocity component (already includes 1/(2+t) scaling)
        v_learned = self.velocity_field(x, t, T)

        # Energy-guided velocity component (also includes 1/(2+t) scaling)
        v_energy = self.compute_energy_velocity(x, t, T)

        # Time-dependent weighting (stronger guidance at later times)
        if time_dependent_weight:
            # λ(t) starts at 0 and increases to energy_weight
            lambda_t = energy_weight * (t / self.t_max).view(-1, 1, 1, 1)
        else:
            lambda_t = energy_weight

        # Combine velocities
        v_total = v_learned + lambda_t * v_energy

        return v_total

    def sample_with_energy_guidance(
        self,
        batch_size: Optional[int] = None,
        T: Optional[torch.Tensor] = None,
        energy_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Generate samples using DDPM-style denoising with energy guidance.

        Combines learned denoising with mean-field Boltzmann guidance
        for better convergence to the target distribution.

        Args:
            batch_size: Number of samples to generate
            T: Temperature tensor of shape (B,) or (B, 1)
            energy_weight: Weight for energy-guided correction

        Returns:
            Samples of shape (B, 1, H, W) in {-1, +1}
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        H, W = self.size

        # Handle temperature
        if T is None:
            T = torch.ones(batch_size, device=self.device) * self.temp_ref
        else:
            T = T.to(self.device)
            if T.dim() == 2:
                T = T.squeeze(-1)
            if T.shape[0] == 1 and batch_size > 1:
                T = T.expand(batch_size)

        beta = 1.0 / T  # (B,)

        # Initialize with random Dirichlet samples
        alpha_init = torch.ones(batch_size, H, W, 2, device=self.device)
        x = torch.distributions.Dirichlet(alpha_init).sample()
        x = x.permute(0, 3, 1, 2)  # (B, 2, H, W)

        # Time schedule: from t_max down to t_min (cosine)
        steps = torch.linspace(0, 1, self.num_flow_steps + 1, device=self.device)
        t_schedule = self.t_min + (self.t_max - self.t_min) * (1 - torch.cos(steps * math.pi / 2))
        t_schedule = t_schedule.flip(0)

        # Neighbor kernel for local field computation
        kernel = torch.tensor(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            dtype=x.dtype, device=self.device
        ).view(1, 1, 3, 3)

        # DDPM-style denoising with energy guidance
        with torch.no_grad():
            for i in range(self.num_flow_steps):
                t_current = t_schedule[i]
                t_next = t_schedule[i + 1]
                t_batch = torch.full((batch_size,), t_current.item(), device=self.device)

                # 1. Predict clean state from learned denoiser
                logits = self.denoise(x, t_batch, T)
                x_clean_pred = F.softmax(logits, dim=1)

                # 2. Apply energy-based correction (mean-field Boltzmann)
                # Compute local field from current expected spins
                expected_spin = 2 * x[:, 1:2] - 1
                padded = F.pad(expected_spin, (1, 1, 1, 1), mode='circular')
                local_field = F.conv2d(padded, kernel)

                # Boltzmann probability
                p_boltz_plus = torch.sigmoid(2 * beta.view(-1, 1, 1, 1) * local_field)
                p_boltz = torch.cat([1 - p_boltz_plus, p_boltz_plus], dim=1)

                # Blend learned prediction with energy guidance
                # More energy guidance at later steps (lower noise)
                progress = i / self.num_flow_steps
                blend_weight = energy_weight * progress
                x_guided = (1 - blend_weight) * x_clean_pred + blend_weight * p_boltz

                if t_next > self.t_min:
                    # Re-noise using Dirichlet with soft predictions
                    t_val = t_next.item()
                    effective_t = min(t_val, 20.0)
                    alpha = 1.0 + effective_t * x_guided
                    alpha = alpha.permute(0, 2, 3, 1)
                    x = torch.distributions.Dirichlet(alpha).sample()
                    x = x.permute(0, 3, 1, 2)
                else:
                    x = x_guided

        # Convert to discrete spins by SAMPLING from predicted distribution
        samples = (torch.rand_like(x[:, 1]) < x[:, 1]).float()
        samples = samples.unsqueeze(1)

        if self.fix_first is not None:
            samples[:, 0, 0, 0] = float(self.fix_first)

        samples = self.mapping(samples)

        return samples

    def improve_samples_with_energy(
        self,
        samples: torch.Tensor,
        T: torch.Tensor,
        energy_fn,
        n_steps: int = 10,
        use_checkerboard: bool = True
    ) -> torch.Tensor:
        """
        Improve samples by Metropolis-Hastings spin flips toward Boltzmann distribution.

        Uses checkerboard updates for faster mixing: update all even/odd sites
        in parallel since they don't share neighbors.

        Args:
            samples: Input samples (B, 1, H, W) in {-1, +1}
            T: Temperature values (B,)
            energy_fn: Energy function (unused, we compute locally)
            n_steps: Number of MH sweeps (each sweep updates all sites)
            use_checkerboard: If True, use parallel checkerboard updates

        Returns:
            Improved samples (B, 1, H, W) in {-1, +1}
        """
        B, C, H, W = samples.shape
        improved = samples.clone()
        beta = 1.0 / T.view(B, 1, 1)  # (B, 1, 1)

        if use_checkerboard:
            # Create checkerboard masks
            row_idx = torch.arange(H, device=samples.device).view(1, H, 1)
            col_idx = torch.arange(W, device=samples.device).view(1, 1, W)
            even_mask = ((row_idx + col_idx) % 2 == 0).expand(B, H, W)
            odd_mask = ~even_mask

            # Fixed first position mask
            if self.fix_first is not None:
                fixed_mask = torch.zeros(B, H, W, dtype=torch.bool, device=samples.device)
                fixed_mask[:, 0, 0] = True
            else:
                fixed_mask = torch.zeros(B, H, W, dtype=torch.bool, device=samples.device)

            for _ in range(n_steps):
                for mask in [even_mask, odd_mask]:
                    # Compute local field for all sites
                    spins = improved[:, 0]  # (B, H, W)
                    neighbors = (
                        torch.roll(spins, 1, dims=1) +
                        torch.roll(spins, -1, dims=1) +
                        torch.roll(spins, 1, dims=2) +
                        torch.roll(spins, -1, dims=2)
                    )  # (B, H, W)

                    # Energy change if we flip: ΔE = 2 * s * h
                    delta_E = 2 * spins * neighbors  # (B, H, W)

                    # Metropolis acceptance probability
                    accept_prob = torch.where(
                        delta_E < 0,
                        torch.ones_like(delta_E),
                        torch.exp(-beta * delta_E)
                    )

                    # Random acceptance
                    accept = torch.rand_like(accept_prob) < accept_prob

                    # Apply mask and fixed position constraint
                    accept = accept & mask & ~fixed_mask

                    # Flip accepted spins
                    improved[:, 0] = torch.where(accept, -spins, spins)
        else:
            # Original single-site update
            for _ in range(n_steps * H * W):
                i = torch.randint(0, H, (B,), device=samples.device)
                j = torch.randint(0, W, (B,), device=samples.device)

                if self.fix_first is not None:
                    mask_fixed = (i == 0) & (j == 0)
                else:
                    mask_fixed = torch.zeros(B, dtype=torch.bool, device=samples.device)

                batch_idx = torch.arange(B, device=samples.device)
                current_spin = improved[batch_idx, 0, i, j]

                neighbors = (
                    improved[batch_idx, 0, (i - 1) % H, j] +
                    improved[batch_idx, 0, (i + 1) % H, j] +
                    improved[batch_idx, 0, i, (j - 1) % W] +
                    improved[batch_idx, 0, i, (j + 1) % W]
                )

                delta_E = 2 * current_spin * neighbors
                accept_prob = torch.where(
                    delta_E < 0,
                    torch.ones_like(delta_E),
                    torch.exp(-beta * delta_E)
                )
                accept = torch.rand(B, device=samples.device) < accept_prob
                accept = accept & ~mask_fixed

                new_spin = torch.where(accept, -current_spin, current_spin)
                improved[batch_idx, 0, i, j] = new_spin

        return improved

    def wolff_cluster_update(
        self,
        samples: torch.Tensor,
        T: torch.Tensor,
        n_clusters: int = 10
    ) -> torch.Tensor:
        """
        Improve samples using Wolff cluster algorithm.

        Wolff algorithm eliminates critical slowing down by flipping entire
        clusters of aligned spins. The cluster is grown with probability
        p_add = 1 - exp(-2β) for same-spin neighbors.

        This is O(1) in correlation time at Tc, unlike Metropolis which is O(ξ^z).

        Args:
            samples: Input samples (B, 1, H, W) in {-1, +1}
            T: Temperature values (B,)
            n_clusters: Number of cluster flips per sample

        Returns:
            Improved samples (B, 1, H, W) in {-1, +1}
        """
        B, C, H, W = samples.shape
        device = samples.device
        improved = samples.clone()

        # Bond probability: p_add = 1 - exp(-2β)
        beta = 1.0 / T  # (B,)
        p_add = 1.0 - torch.exp(-2.0 * beta)  # (B,)

        # Direction offsets for neighbors (up, down, left, right)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for _ in range(n_clusters):
            # Process each batch element (cluster algorithm is inherently sequential per sample)
            for b in range(B):
                spins = improved[b, 0]  # (H, W)
                p = p_add[b].item()

                # Random seed position
                seed_i = torch.randint(0, H, (1,), device=device).item()
                seed_j = torch.randint(0, W, (1,), device=device).item()

                # Skip if seed is fixed position
                if self.fix_first is not None and seed_i == 0 and seed_j == 0:
                    continue

                seed_spin = spins[seed_i, seed_j].item()

                # BFS to grow cluster
                cluster = set()
                cluster.add((seed_i, seed_j))
                queue = [(seed_i, seed_j)]

                while queue:
                    ci, cj = queue.pop(0)

                    for di, dj in directions:
                        ni, nj = (ci + di) % H, (cj + dj) % W

                        # Skip if already in cluster
                        if (ni, nj) in cluster:
                            continue

                        # Skip fixed position
                        if self.fix_first is not None and ni == 0 and nj == 0:
                            continue

                        # Check if same spin and accept with probability p_add
                        if spins[ni, nj].item() == seed_spin:
                            if torch.rand(1, device=device).item() < p:
                                cluster.add((ni, nj))
                                queue.append((ni, nj))

                # Flip entire cluster
                for (fi, fj) in cluster:
                    improved[b, 0, fi, fj] = -improved[b, 0, fi, fj]

        return improved

    def wolff_cluster_update_vectorized(
        self,
        samples: torch.Tensor,
        T: torch.Tensor,
        n_clusters: int = 10
    ) -> torch.Tensor:
        """
        Vectorized Wolff cluster algorithm using bond percolation.

        Correct implementation: pre-compute all bond activations, then
        grow clusters through activated bonds only.

        Args:
            samples: Input samples (B, 1, H, W) in {-1, +1}
            T: Temperature values (B,)
            n_clusters: Number of cluster flips per sample

        Returns:
            Improved samples (B, 1, H, W) in {-1, +1}
        """
        B, C, H, W = samples.shape
        device = samples.device
        improved = samples.clone()

        # Bond probability: p_add = 1 - exp(-2β)
        beta = 1.0 / T.view(B, 1, 1)  # (B, 1, 1)
        p_add = 1.0 - torch.exp(-2.0 * beta)  # (B, 1, 1)

        # Fixed position mask
        if self.fix_first is not None:
            fixed_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
            fixed_mask[:, 0, 0] = True
        else:
            fixed_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

        for _ in range(n_clusters):
            spins = improved[:, 0]  # (B, H, W)

            # Pre-compute bond activations (key fix!)
            # For each direction, bond is active if:
            # 1. Same spin on both ends
            # 2. Random < p_add (computed once per bond)

            # Right bonds: between (i,j) and (i, j+1)
            same_right = (spins == torch.roll(spins, -1, dims=2))
            bond_right = same_right & (torch.rand(B, H, W, device=device) < p_add)

            # Down bonds: between (i,j) and (i+1, j)
            same_down = (spins == torch.roll(spins, -1, dims=1))
            bond_down = same_down & (torch.rand(B, H, W, device=device) < p_add)

            # Random seed positions for each batch
            seed_i = torch.randint(0, H, (B,), device=device)
            seed_j = torch.randint(0, W, (B,), device=device)

            # Initialize cluster
            batch_idx = torch.arange(B, device=device)
            in_cluster = torch.zeros(B, H, W, dtype=torch.bool, device=device)
            in_cluster[batch_idx, seed_i, seed_j] = True

            # Iterative expansion through active bonds only
            frontier = in_cluster.clone()
            max_iterations = H + W

            for _ in range(max_iterations):
                if not frontier.any():
                    break

                # Expand through active bonds
                # Right: frontier can reach right neighbor if bond_right is active
                expand_right = frontier & bond_right
                reach_right = torch.roll(expand_right, -1, dims=2)

                # Left: right neighbor's frontier can reach us if their bond_right (our left) is active
                expand_left = torch.roll(frontier, -1, dims=2) & torch.roll(bond_right, -1, dims=2)
                reach_left = torch.roll(expand_left, 1, dims=2)

                # Down: frontier can reach down neighbor if bond_down is active
                expand_down = frontier & bond_down
                reach_down = torch.roll(expand_down, -1, dims=1)

                # Up: down neighbor's frontier can reach us if their bond_down (our up) is active
                expand_up = torch.roll(frontier, -1, dims=1) & torch.roll(bond_down, -1, dims=1)
                reach_up = torch.roll(expand_up, 1, dims=1)

                # New members: reachable and not in cluster yet
                new_members = (reach_right | reach_left | reach_down | reach_up) & ~in_cluster & ~fixed_mask

                in_cluster = in_cluster | new_members
                frontier = new_members

            # Flip entire cluster
            improved[:, 0] = torch.where(in_cluster, -spins, spins)

        return improved

    def swendsen_wang_update(
        self,
        samples: torch.Tensor,
        T: torch.Tensor,
        n_sweeps: int = 1
    ) -> torch.Tensor:
        """
        Swendsen-Wang cluster algorithm - fully parallel version.

        Unlike Wolff which flips one cluster at a time, SW identifies ALL
        clusters simultaneously and flips each with 50% probability.

        This is more efficient for GPU as it's fully parallelizable.

        Args:
            samples: Input samples (B, 1, H, W) in {-1, +1}
            T: Temperature values (B,)
            n_sweeps: Number of full SW sweeps

        Returns:
            Improved samples (B, 1, H, W) in {-1, +1}
        """
        B, C, H, W = samples.shape
        device = samples.device
        improved = samples.clone()

        # Bond probability
        beta = 1.0 / T.view(B, 1, 1)
        p_bond = 1.0 - torch.exp(-2.0 * beta)

        for _ in range(n_sweeps):
            spins = improved[:, 0]  # (B, H, W)

            # Activate bonds between same-spin neighbors
            same_right = (spins == torch.roll(spins, -1, dims=2))
            same_down = (spins == torch.roll(spins, -1, dims=1))

            bond_right = same_right & (torch.rand(B, H, W, device=device) < p_bond)
            bond_down = same_down & (torch.rand(B, H, W, device=device) < p_bond)

            # Find connected components using iterative label propagation
            # Initialize each site with unique label
            labels = torch.arange(H * W, device=device).view(1, H, W).expand(B, -1, -1).float()
            labels = labels + torch.arange(B, device=device).view(B, 1, 1) * H * W

            # Iterative min-propagation through active bonds
            for _ in range(H + W):
                old_labels = labels.clone()

                # Propagate through right bonds
                right_labels = torch.roll(labels, -1, dims=2)
                labels = torch.where(bond_right, torch.minimum(labels, right_labels), labels)
                left_labels = torch.roll(labels, 1, dims=2)
                bond_left = torch.roll(bond_right, 1, dims=2)
                labels = torch.where(bond_left, torch.minimum(labels, left_labels), labels)

                # Propagate through down bonds
                down_labels = torch.roll(labels, -1, dims=1)
                labels = torch.where(bond_down, torch.minimum(labels, down_labels), labels)
                up_labels = torch.roll(labels, 1, dims=1)
                bond_up = torch.roll(bond_down, 1, dims=1)
                labels = torch.where(bond_up, torch.minimum(labels, up_labels), labels)

                if (labels == old_labels).all():
                    break

            # Generate random flip decision for each unique label
            # Use label as random seed for consistent flip decision within cluster
            flip_random = torch.fmod(labels * 2654435761, 2**32) / 2**32  # Hash-based random
            flip_mask = flip_random < 0.5

            # Don't flip fixed position's cluster
            if self.fix_first is not None:
                fixed_label = labels[:, 0, 0].view(B, 1, 1)
                flip_mask = flip_mask & (labels != fixed_label)

            # Flip selected clusters
            improved[:, 0] = torch.where(flip_mask, -spins, spins)

        return improved

    def mcmc_update(
        self,
        samples: torch.Tensor,
        T: torch.Tensor,
        energy_fn,
        n_sw_sweeps: int = 10,
        n_mh_steps: int = 0
    ) -> torch.Tensor:
        """
        MCMC update for optimal equilibration.

        Uses Swendsen-Wang (SW) cluster algorithm as the primary engine.
        SW is theoretically sufficient to reach Boltzmann distribution
        and eliminates critical slowing down at Tc.

        Args:
            samples: Input samples (B, 1, H, W) in {-1, +1}
            T: Temperature values (B,)
            energy_fn: Energy function for MH acceptance (if used)
            n_sw_sweeps: Number of Swendsen-Wang sweeps (Recommended: 5-10)
            n_mh_steps: Number of MH sweeps (Optional refinement, 0 to skip)

        Returns:
            Improved samples (B, 1, H, W) in {-1, +1}
        """
        # Step 1: Swendsen-Wang for fast global equilibration
        improved = self.swendsen_wang_update(samples, T, n_sweeps=n_sw_sweeps)

        # Step 2: Optional Metropolis-Hastings for local refinement
        if n_mh_steps > 0:
            improved = self.improve_samples_with_energy(
                improved, T, energy_fn, n_steps=n_mh_steps, use_checkerboard=True
            )

        return improved

    def denoise(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        T: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict clean state from noisy state.

        Used for cross-entropy training loss.

        Args:
            x_noisy: Noisy state on probability simplex (B, 2, H, W)
            t: Time values (B,)
            T: Temperature values (B,)

        Returns:
            Logits for clean state (B, 2, H, W)
        """
        B, _, H, W = x_noisy.shape

        # Time embedding
        t_emb = self.time_embedding(t)

        # Temperature channel
        T_channel = T.view(B, 1, 1, 1).expand(B, 1, H, W)

        # Input
        x_input = torch.cat([x_noisy, T_channel], dim=1)

        # Network forward
        logits = self.net(x_input, t_emb).squeeze(2)  # (B, 2, H, W)

        return logits

    def sample(
        self,
        batch_size: Optional[int] = None,
        T: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate samples. Defaults to ODE integration which is more stable.
        """
        return self.sample_ode(batch_size, T)

    def sample_ode(
        self,
        batch_size: Optional[int] = None,
        T: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate samples via Euler ODE integration (original method).

        Note: This method may have convergence issues due to diminishing
        velocity at large t. Use sample() for better results.

        Args:
            batch_size: Number of samples to generate
            T: Temperature tensor of shape (B,) or (B, 1)

        Returns:
            Samples of shape (B, 1, H, W) in {-1, +1}
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        H, W = self.size

        # Handle temperature
        if T is None:
            T = torch.ones(batch_size, device=self.device) * self.temp_ref
        else:
            T = T.to(self.device)
            if T.dim() == 2:
                T = T.squeeze(-1)
            if T.shape[0] == 1 and batch_size > 1:
                T = T.expand(batch_size)

        # Initialize with random samples from Dirichlet(1, 1) = Uniform on simplex
        alpha = torch.ones(batch_size, H, W, 2, device=self.device)
        x = torch.distributions.Dirichlet(alpha).sample()
        x = x.permute(0, 3, 1, 2)

        # Time grid
        dt = (self.t_max - self.t_min) / self.num_flow_steps

        # Euler integration
        for step in range(self.num_flow_steps):
            t = self.t_min + step * dt
            t_batch = torch.full((batch_size,), t, device=self.device)

            with torch.no_grad():
                v = self.velocity_field(x, t_batch, T)

            x = x + v * dt
            x = F.softmax(torch.log(x.clamp(min=1e-8)), dim=1)

        # Convert to discrete spins
        samples = (x[:, 1] > 0.5).float()
        samples = samples.unsqueeze(1)

        if self.fix_first is not None:
            samples[:, 0, 0, 0] = float(self.fix_first)

        samples = self.mapping(samples)

        return samples

    def log_prob(
        self,
        sample: torch.Tensor,
        T: Optional[torch.Tensor] = None,
        steps: int = 20,
        method: str = "elbo"
    ) -> torch.Tensor:
        """
        Estimate log probability of samples.

        Methods:
        - "elbo": ELBO-style bound using denoising likelihood integral
        - "denoising": Simple average denoising score (fast but inaccurate)

        Note: For DFM, exact log probability is intractable. These are
        approximations/bounds. For accurate evaluation, use sampling-based
        metrics (energy, magnetization) instead.

        Args:
            sample: Samples of shape (B, 1, H, W) in {-1, +1}
            T: Temperature tensor (B,)
            steps: Number of time integration steps
            method: "elbo" or "denoising"

        Returns:
            Log probability estimates (B, 1)
        """
        if method == "elbo":
            return self._log_prob_elbo(sample, T, steps)
        else:
            return self._log_prob_denoising(sample, T, steps)

    def _log_prob_elbo(
        self,
        sample: torch.Tensor,
        T: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> torch.Tensor:
        """
        ELBO-style log probability estimation.

        Uses the variational bound:
          log q(x) >= E_t[ log p_θ(x | x_t) ] - D_KL(q(x_t|x) || prior)

        For Dirichlet noising, at t=t_max the distribution is close to uniform,
        so we use:
          log q(x) ≈ log(1/2^N) + ∫ weight(t) * denoising_logprob(t) dt

        where N is the number of free sites.
        """
        B, C, H, W = sample.shape

        if T is None:
            T = torch.ones(B, device=self.device) * self.temp_ref
        else:
            T = T.to(self.device)
            if T.dim() == 2:
                T = T.squeeze(-1)

        # Number of free sites
        N_free = H * W - (1 if self.fix_first is not None else 0)

        # Prior contribution: uniform at t_max
        # log p(prior) = -N * log(2)
        log_prior = -N_free * torch.log(torch.tensor(2.0, device=self.device))

        # Convert to target indices
        sample_idx = self.reverse_mapping(sample).long()

        # Mask for fixed position
        if self.fix_first is not None:
            mask = torch.ones(B, 1, H, W, device=self.device)
            mask[:, :, 0, 0] = 0
        else:
            mask = torch.ones(B, 1, H, W, device=self.device)

        # Time integration with proper weighting
        # Use Gaussian quadrature-like weighting: more weight near t_min
        t_values = torch.linspace(self.t_min, self.t_max, steps, device=self.device)
        dt = (self.t_max - self.t_min) / steps

        # Weight function: 1/(2+t) from Dirichlet velocity scaling
        # Integrand: w(t) * log p(x|x_t)
        integrated_log_prob = torch.zeros(B, device=self.device)

        for t_val in t_values:
            t_batch = torch.full((B,), t_val.item(), device=self.device)

            # Noise the sample
            x_noisy = self.apply_dirichlet_noise(sample, t_batch)

            # Denoising prediction
            logits = self.denoise(x_noisy, t_batch, T)
            log_softmax = F.log_softmax(logits, dim=1)

            # Log prob of true class
            log_prob_pixel = log_softmax.gather(1, sample_idx)
            log_prob_step = (log_prob_pixel * mask).sum(dim=[1, 2, 3])

            # Time weight: 1/(2+t) normalized
            weight = 1.0 / (2.0 + t_val.item())

            integrated_log_prob += weight * log_prob_step * dt

        # Normalize the integral
        # Total weight = ∫ 1/(2+t) dt from t_min to t_max = log(2+t_max) - log(2+t_min)
        total_weight = torch.log(torch.tensor(2.0 + self.t_max)) - torch.log(torch.tensor(2.0 + self.t_min))
        normalized_log_prob = integrated_log_prob / total_weight.item()

        # Combine: prior + denoising contribution
        # Scale denoising by N_free since each site contributes
        log_prob = log_prior + normalized_log_prob

        return log_prob.unsqueeze(-1)

    def _log_prob_denoising(
        self,
        sample: torch.Tensor,
        T: Optional[torch.Tensor] = None,
        steps: int = 20
    ) -> torch.Tensor:
        """
        Simple denoising-based log probability (original method).

        Warning: This is NOT an accurate log probability estimate!
        Use for relative comparisons only.
        """
        B, C, H, W = sample.shape

        if T is None:
            T = torch.ones(B, device=self.device) * self.temp_ref
        else:
            T = T.to(self.device)
            if T.dim() == 2:
                T = T.squeeze(-1)

        sample_idx = self.reverse_mapping(sample).long()

        if self.fix_first is not None:
            mask = torch.ones(B, 1, H, W, device=self.device)
            mask[:, :, 0, 0] = 0
        else:
            mask = torch.ones(B, 1, H, W, device=self.device)

        total_log_prob = 0.0
        t_values = torch.linspace(self.t_min, self.t_max, steps, device=self.device)

        for t_val in t_values:
            t_batch = torch.full((B,), t_val.item(), device=self.device)
            x_noisy = self.apply_dirichlet_noise(sample, t_batch)
            logits = self.denoise(x_noisy, t_batch, T)
            log_softmax = F.log_softmax(logits, dim=1)
            log_prob_pixel = log_softmax.gather(1, sample_idx)
            log_prob_step = (log_prob_pixel * mask).sum(dim=[1, 2, 3])
            total_log_prob += log_prob_step

        avg_log_prob = total_log_prob / steps

        return avg_log_prob.unsqueeze(-1)

    def training_loss(
        self,
        samples: torch.Tensor,
        T: torch.Tensor,
        energy_fn=None,
        lambda_reinforce: float = 1.0,
        training_mode: str = "energy_guided",
        energy_weight: float = 1.0,
        mh_steps: int = 10,
        use_cluster: bool = False,
        n_clusters: int = 10,
        n_sw_sweeps: int = 10,
        n_mh_after_sw: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute training loss for self-training (no external data).

        Training modes:
        - "energy_guided": Energy-guided denoising + velocity matching (RECOMMENDED)
          * Improves samples with Swendsen-Wang (if use_cluster=True) or pure MH
          * Trains denoising to predict low-energy configurations
          * Additionally matches velocity to energy-guided velocity
        - "reinforce": Pure REINFORCE with RLOO baseline
        - "hybrid": Denoising + REINFORCE (requires external target data)
        - "denoise_only": Only denoising loss (requires external target data)

        Args:
            samples: Clean samples (B, 1, H, W) in {-1, +1}
            T: Temperature values (B,)
            energy_fn: Energy function
            lambda_reinforce: Weight for REINFORCE loss (ignored in some modes)
            training_mode: "energy_guided", "reinforce", "hybrid", or "denoise_only"
            energy_weight: Weight for energy-guided velocity matching
            mh_steps: Number of Metropolis-Hastings steps (used if use_cluster=False)
            use_cluster: If True, use Swendsen-Wang instead of pure MH
            n_clusters: (deprecated) Number of SW sweeps
            n_sw_sweeps: Number of Swendsen-Wang sweeps for equilibration
            n_mh_after_sw: Number of MH steps after SW (0 to skip, Recommended: 0)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        B, C, H, W = samples.shape
        metrics = {}

        # Compute energy (always needed)
        if energy_fn is not None:
            energy = energy_fn(samples)  # (B, 1)
            metrics["energy_mean"] = energy.mean().item()
        else:
            energy = torch.zeros(B, 1, device=self.device)

        # ================================================================
        # ENERGY-GUIDED mode (recommended for self-training)
        # ================================================================
        if training_mode == "energy_guided":
            # 1. Generate improved samples via MH
            with torch.no_grad():
                # Determine initialization: Random or Model samples
                # Dynamic strategy:
                # - High T (> 2.5): Always use Random Init (Prob=0) to ensure diversity
                # - Low T (<= 2.5): Use Model Init (Prob=0.9) to help equilibration (Bootstrapping)
                
                # Create probability mask based on Temperature
                # shape of T is (B,) or (B, 1)
                T_val = T.view(B)
                init_probs = torch.where(
                    T_val > 2.5,
                    torch.zeros_like(T_val),
                    torch.full_like(T_val, self.mh_model_init_prob)
                )
                
                use_model_init = torch.rand(B, device=self.device) < init_probs
                use_model_init = use_model_init.view(B, 1, 1, 1).float()

                # Random samples (hot start)
                random_samples = torch.randint(
                    0, 2, (B, 1, H, W), device=self.device
                ).float() * 2 - 1

                # Apply fixed first spin to random samples
                if self.fix_first is not None:
                    random_samples[:, 0, 0, 0] = float(self.fix_first)

                # Mix initialization
                initial_samples = use_model_init * samples + (1 - use_model_init) * random_samples

                # Calculate initial energy for metrics
                initial_energy = energy_fn(initial_samples)

                # Run equilibration: SW-only or pure MH
                if use_cluster:
                    # Swendsen-Wang for fast equilibration
                    sw_sweeps = n_sw_sweeps if n_sw_sweeps > 0 else n_clusters
                    mh_samples = self.mcmc_update(
                        initial_samples, T, energy_fn,
                        n_sw_sweeps=sw_sweeps,
                        n_mh_steps=n_mh_after_sw
                    )
                else:
                    # Pure MH with checkerboard updates
                    mh_samples = self.improve_samples_with_energy(
                        initial_samples, T, energy_fn, n_steps=mh_steps, use_checkerboard=True
                    )
                mh_energy = energy_fn(mh_samples)

            metrics["mh_energy_mean"] = mh_energy.mean().item()
            metrics["mh_energy_delta"] = (mh_energy - initial_energy).mean().item()
            metrics["model_energy_mean"] = energy.mean().item()

            # 2. Denoising loss: train to denoise toward MH-equilibrated samples
            t = torch.rand(B, device=self.device) * (self.t_max - self.t_min) + self.t_min
            x_noisy = self.apply_dirichlet_noise(mh_samples, t)
            logits = self.denoise(x_noisy, t, T)
            target = self.reverse_mapping(mh_samples).squeeze(1).long()

            if self.fix_first is not None:
                mask = torch.ones(B, H, W, device=self.device)
                mask[:, 0, 0] = 0
                ce_loss = F.cross_entropy(logits, target, reduction='none')
                denoise_loss = (ce_loss * mask).sum() / mask.sum()
            else:
                denoise_loss = F.cross_entropy(logits, target)

            metrics["denoise_loss"] = denoise_loss.item()

            # 3. Velocity matching: guide velocity toward MH samples (NOT mean-field!)
            #    This avoids the positive feedback loop of mean-field
            #    For Dirichlet path: u_t(y|x) = (e_x - y) / (2 + t)
            v_learned = self.velocity_field(x_noisy, t, T)  # (B, 2, H, W) - already includes 1/(2+t)

            # Target velocity with correct time scaling
            mh_onehot = F.one_hot(
                self.reverse_mapping(mh_samples).squeeze(1).long(), 2
            ).permute(0, 3, 1, 2).float()  # (B, 2, H, W)

            # Apply time scaling: 1 / (2 + t)
            time_scale = 1.0 / (2.0 + t.view(B, 1, 1, 1))
            v_target = (mh_onehot - x_noisy) * time_scale

            # MSE loss on velocity
            velocity_loss = ((v_learned - v_target.detach()) ** 2).mean()
            metrics["velocity_loss"] = velocity_loss.item()

            # Total loss: denoising is primary, velocity matching is auxiliary
            total_loss = denoise_loss + energy_weight * 0.1 * velocity_loss
            metrics["total_loss"] = total_loss.item()

            return total_loss, metrics

        # ================================================================
        # REINFORCE mode (for self-training without external data)
        # ================================================================
        if training_mode == "reinforce":
            # Compute log probability via velocity field integral
            log_q = self._compute_log_prob_reinforce(samples, T)  # (B, 1)

            # Free energy: F = log q + β·E
            beta = 1.0 / T.unsqueeze(-1)  # (B, 1)
            free_energy = log_q + beta * energy  # (B, 1)

            # RLOO baseline for variance reduction
            F_sum = free_energy.sum()
            baseline = (F_sum - free_energy) / (B - 1)  # (B, 1)

            # REINFORCE gradient: (F - baseline) * ∇log_q
            advantage = (free_energy - baseline).detach()  # (B, 1)

            # Loss = E[(F - baseline) * log_q]
            reinforce_loss = (advantage * log_q).mean()

            metrics["reinforce_loss"] = reinforce_loss.item()
            metrics["log_prob_mean"] = log_q.mean().item()
            metrics["free_energy_mean"] = free_energy.mean().item()
            metrics["advantage_std"] = advantage.std().item()
            metrics["total_loss"] = reinforce_loss.item()

            return reinforce_loss, metrics

        # ================================================================
        # Hybrid or Denoise-only modes (require external target data)
        # ================================================================
        # Denoising loss
        t = torch.rand(B, device=self.device) * (self.t_max - self.t_min) + self.t_min
        x_noisy = self.apply_dirichlet_noise(samples, t)
        logits = self.denoise(x_noisy, t, T)
        target = self.reverse_mapping(samples).squeeze(1).long()

        if self.fix_first is not None:
            mask = torch.ones(B, H, W, device=self.device)
            mask[:, 0, 0] = 0
            ce_loss = F.cross_entropy(logits, target, reduction='none')
            denoise_loss = (ce_loss * mask).sum() / mask.sum()
        else:
            denoise_loss = F.cross_entropy(logits, target)

        metrics["denoise_loss"] = denoise_loss.item()

        if training_mode == "denoise_only":
            metrics["total_loss"] = denoise_loss.item()
            return denoise_loss, metrics

        # Hybrid mode: add REINFORCE
        log_q = self.log_prob(samples, T)
        beta = 1.0 / T.unsqueeze(-1)
        reinforce_weight = (log_q + beta * energy).detach()
        reinforce_loss = (reinforce_weight * log_q).mean()

        metrics["reinforce_loss"] = reinforce_loss.item()
        total_loss = denoise_loss + lambda_reinforce * reinforce_loss
        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics

    def _compute_log_prob_reinforce(
        self,
        samples: torch.Tensor,
        T: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability for REINFORCE training.

        Uses multiple time points for more stable gradient estimation.
        The key is that gradients flow through the velocity field predictions.

        Args:
            samples: Samples (B, 1, H, W) in {-1, +1}
            T: Temperature (B,)

        Returns:
            Log probability estimates (B, 1) with gradients
        """
        B, C, H, W = samples.shape
        num_sites = H * W - (1 if self.fix_first else 0)

        # Use multiple time samples for better gradient estimation
        num_time_samples = 3
        t_values = torch.linspace(
            self.t_min + 0.1 * (self.t_max - self.t_min),
            self.t_max - 0.1 * (self.t_max - self.t_min),
            num_time_samples,
            device=self.device
        )

        log_prob_sum = torch.zeros(B, device=self.device)

        for t_val in t_values:
            t = torch.full((B,), t_val.item(), device=self.device)

            # Apply noising (no gradient needed here)
            with torch.no_grad():
                x_noisy = self.apply_dirichlet_noise(samples, t)

            # Get denoising prediction (gradients flow here)
            logits = self.denoise(x_noisy, t, T)  # (B, 2, H, W)
            log_softmax = F.log_softmax(logits, dim=1)

            # Get target indices
            target_idx = self.reverse_mapping(samples).long()  # (B, 1, H, W)

            # Gather log probs for actual values
            log_prob_selected = log_softmax.gather(1, target_idx)  # (B, 1, H, W)

            # Mask fixed position
            if self.fix_first is not None:
                mask = torch.ones(1, 1, H, W, device=self.device)
                mask[:, :, 0, 0] = 0
                log_prob_selected = log_prob_selected * mask

            log_prob_sum = log_prob_sum + log_prob_selected.sum(dim=[1, 2, 3])

        # Average over time samples
        log_prob = log_prob_sum / num_time_samples

        return log_prob.unsqueeze(-1)  # (B, 1)


if __name__ == "__main__":
    """Test the DiscreteFlowMatcher model."""
    print("=" * 70)
    print("Testing DiscreteFlowMatcher")
    print("=" * 70)

    # Test configuration
    hparams = {
        "size": 8,  # Small lattice for testing
        "fix_first": 1,
        "batch_size": 4,
        "num_beta": 2,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "num_flow_steps": 20,
        "t_max": 5.0,
        "hidden_channels": 32,
        "hidden_conv_layers": 2,
        "hidden_width": 64,
        "hidden_fc_layers": 1,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create model
    model = DiscreteFlowMatcher(hparams, device=device).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test time embedding
    print("\n--- Testing SinusoidalTimeEmbedding ---")
    t = torch.tensor([0.0, 0.5, 1.0, 5.0], device=device)
    t_emb = model.time_embedding(t)
    print(f"Time embedding shape: {t_emb.shape}")

    # Test noising
    print("\n--- Testing Dirichlet Noising ---")
    x = torch.randint(0, 2, (4, 1, 8, 8), device=device).float() * 2 - 1
    t = torch.tensor([0.1, 1.0, 5.0, 9.0], device=device)
    x_noisy = model.apply_dirichlet_noise(x, t)
    print(f"Noisy sample shape: {x_noisy.shape}")
    print(f"Noisy sample sum (should be ~1): {x_noisy.sum(dim=1).mean():.4f}")

    # Test sampling
    print("\n--- Testing Sampling ---")
    T = torch.tensor([2.0, 2.5, 3.0, 4.0], device=device)
    with torch.no_grad():
        samples = model.sample(batch_size=4, T=T)
    print(f"Sample shape: {samples.shape}")
    print(f"Sample values: {samples.unique()}")
    print(f"First spin (should be 1): {samples[:, 0, 0, 0]}")

    # Test log_prob
    print("\n--- Testing Log Probability ---")
    log_probs = model.log_prob(samples, T)
    print(f"Log prob shape: {log_probs.shape}")
    print(f"Log probs: {log_probs.squeeze()}")

    # Test training loss
    print("\n--- Testing Training Loss ---")

    def dummy_energy(x):
        # Simple energy: -sum of neighboring spin products
        right = torch.roll(x, -1, dims=-1)
        down = torch.roll(x, -1, dims=-2)
        energy = -(x * right + x * down).sum(dim=[1, 2, 3])
        return energy.unsqueeze(-1)

    loss, metrics = model.training_loss(samples, T, energy_fn=dummy_energy)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Test gradient flow
    print("\n--- Testing Gradient Flow ---")
    loss.backward()
    grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
    print(f"Gradient norm: {grad_norm:.4f}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
