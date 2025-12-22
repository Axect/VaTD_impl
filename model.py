import torch
from torch import nn
import torch.nn.functional as F
import einops
import numpy as np
from typing import Tuple, Optional, List
import math


# ============================================================================
# Temperature Embedding + FiLM Conditioning
# ============================================================================


class TemperatureEmbedding(nn.Module):
    """
    Physics-motivated temperature embedding using inverse temperature (β = 1/T).

    Uses sinusoidal encoding (like positional encoding in Transformers) followed
    by MLP refinement. This allows the model to distinguish subtle temperature
    differences, especially near critical points.

    Key insight: β = 1/T is the natural variable in Boltzmann distribution:
        P(x) ∝ exp(-βE(x))

    The sinusoidal encoding captures multiple frequency scales:
    - Low frequencies: overall temperature regime (hot vs cold)
    - High frequencies: fine temperature variations (critical region)
    """

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128, num_freqs: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freqs = num_freqs

        # Frequency spectrum: from slow to fast variations
        # exp(0) = 1 to exp(-4) ≈ 0.018, covering ~2 orders of magnitude
        freqs = torch.exp(torch.linspace(0, -4, num_freqs))
        self.register_buffer('freqs', freqs)

        # MLP for learnable refinement of sinusoidal features
        sinusoidal_dim = num_freqs * 2  # sin + cos
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """
        Args:
            T: Temperature tensor of shape (B,) or (B, 1)

        Returns:
            Embedding tensor of shape (B, embed_dim)
        """
        if T.dim() == 1:
            T = T.unsqueeze(-1)  # (B, 1)

        # Convert to inverse temperature (physically natural variable)
        beta = 1.0 / T  # (B, 1)

        # Sinusoidal encoding: β * freqs
        # freqs: (num_freqs,) -> (1, num_freqs)
        x = beta * self.freqs.unsqueeze(0)  # (B, num_freqs)

        # Concatenate sin and cos for rich representation
        sinusoidal = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (B, 2*num_freqs)

        # Learnable refinement
        embedding = self.mlp(sinusoidal)  # (B, embed_dim)

        return embedding


class FiLMGenerator(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) generator.

    Generates scale (γ) and shift (β) parameters for each layer based on
    temperature embedding. This allows temperature to modulate features
    multiplicatively, which is more expressive than simple concatenation.

    FiLM: output = γ(T) * features + β(T)

    References:
    - Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)
    - Used in StyleGAN, Diffusion Models for conditional generation
    """

    def __init__(self, embed_dim: int, num_layers: int, feature_dims: List[int]):
        """
        Args:
            embed_dim: Dimension of temperature embedding
            num_layers: Number of layers to modulate
            feature_dims: List of feature dimensions for each layer
        """
        super().__init__()
        self.num_layers = num_layers
        self.feature_dims = feature_dims

        # Generate γ and β for each layer
        self.film_layers = nn.ModuleList()
        for feat_dim in feature_dims:
            self.film_layers.append(
                nn.Linear(embed_dim, feat_dim * 2)  # γ and β concatenated
            )

    def forward(self, temp_embedding: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            temp_embedding: (B, embed_dim) from TemperatureEmbedding

        Returns:
            List of (gamma, beta) tuples for each layer
            Each gamma, beta has shape (B, feat_dim)
        """
        film_params = []
        for i, layer in enumerate(self.film_layers):
            params = layer(temp_embedding)  # (B, feat_dim * 2)
            feat_dim = self.feature_dims[i]
            gamma = params[:, :feat_dim]  # (B, feat_dim)
            beta = params[:, feat_dim:]   # (B, feat_dim)
            # Initialize gamma close to 1, beta close to 0 for stable training
            gamma = 1 + gamma * 0.1
            beta = beta * 0.1
            film_params.append((gamma, beta))

        return film_params


def apply_film(features: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Apply FiLM modulation to features.

    Args:
        features: (B, C, H, W) feature tensor
        gamma: (B, C) scale parameter
        beta: (B, C) shift parameter

    Returns:
        Modulated features: (B, C, H, W)
    """
    # Expand gamma and beta to match spatial dimensions
    gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
    beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
    return gamma * features + beta


# ============================================================================
# Masked Convolution Layers (Simplified for FiLM - no augment_channels)
# ============================================================================


class MaskedConv2DSimple(nn.Conv2d):
    """
    Simplified MaskedConv2D without augment_channels.
    Temperature conditioning is handled by FiLM instead.
    """

    def __init__(self, *args, mask_type: str, data_channels: int, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {"A", "B"}, "mask_type must be either 'A' or 'B'"

        out_channels, in_channels, height, width = self.weight.size()
        y_center, x_center = height // 2, width // 2

        mask = torch.ones_like(self.weight)
        mask[:, :, y_center + 1:, :] = 0
        mask[:, :, y_center, x_center:] = 0

        if mask_type == "A":
            meta_mask = torch.tril(
                torch.ones((data_channels, data_channels)), diagonal=-1
            )
        else:
            meta_mask = torch.tril(torch.ones((data_channels, data_channels)))

        # Tile meta mask to match channel dimensions
        in_tiles = in_channels // data_channels
        out_tiles = out_channels // data_channels
        mask[:, :, y_center, x_center] = meta_mask.repeat(out_tiles, in_tiles)

        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class MaskedResConv2DFiLM(nn.Module):
    """
    Masked Residual Convolutional Network with FiLM conditioning.

    Temperature information is injected via FiLM modulation at each
    residual block, rather than as an additional input channel.
    """

    def __init__(
        self,
        channel: int,
        kernel_size: int,
        hidden_channels: int,
        hidden_conv_layers: int,
        hidden_kernel_size: int,
        hidden_width: int,
        hidden_fc_layers: int,
        category: int,
    ):
        super().__init__()

        self.channel = channel
        self.category = category
        self.hidden_channels = hidden_channels
        self.hidden_conv_layers = hidden_conv_layers
        self.hidden_fc_layers = hidden_fc_layers

        # First convolution (mask type A - can't see current pixel)
        self.first_conv = MaskedConv2DSimple(
            in_channels=channel,
            out_channels=2 * hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            mask_type="A",
            data_channels=channel,
        )

        # Hidden conv layers (residual blocks)
        hidden_convs = []
        for _ in range(hidden_conv_layers):
            hidden_convs.append(
                nn.ModuleDict({
                    'conv1': MaskedConv2DSimple(
                        in_channels=2 * hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        padding=0,
                        mask_type="B",
                        data_channels=channel,
                    ),
                    'conv2': MaskedConv2DSimple(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=hidden_kernel_size,
                        padding=(hidden_kernel_size - 1) // 2,
                        mask_type="B",
                        data_channels=channel,
                    ),
                    'conv3': MaskedConv2DSimple(
                        in_channels=hidden_channels,
                        out_channels=2 * hidden_channels,
                        kernel_size=1,
                        padding=0,
                        mask_type="B",
                        data_channels=channel,
                    ),
                })
            )
        self.hidden_convs = nn.ModuleList(hidden_convs)

        # First FC layer
        self.first_fc = MaskedConv2DSimple(
            in_channels=2 * hidden_channels,
            out_channels=hidden_width,
            kernel_size=1,
            mask_type="B",
            data_channels=channel,
        )

        # Hidden FC layers
        hidden_fcs = []
        for _ in range(hidden_fc_layers):
            hidden_fcs.append(
                MaskedConv2DSimple(
                    in_channels=hidden_width,
                    out_channels=hidden_width,
                    kernel_size=1,
                    mask_type="B",
                    data_channels=channel,
                )
            )
        self.hidden_fcs = nn.ModuleList(hidden_fcs)

        # Final output layer
        self.final_fc = MaskedConv2DSimple(
            in_channels=hidden_width,
            out_channels=category * channel,
            kernel_size=1,
            mask_type="B",
            data_channels=channel,
        )

    def get_film_feature_dims(self) -> List[int]:
        """Return feature dimensions for FiLM generator."""
        dims = []
        # After first conv
        dims.append(2 * self.hidden_channels)
        # After each residual block
        for _ in range(self.hidden_conv_layers):
            dims.append(2 * self.hidden_channels)
        # After first FC and hidden FCs
        for _ in range(1 + self.hidden_fc_layers):
            dims.append(self.hidden_fcs[0].out_channels if self.hidden_fcs else self.first_fc.out_channels)
        return dims

    def forward(
        self,
        x: torch.Tensor,
        film_params: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional FiLM conditioning.

        Args:
            x: Input tensor (B, C, H, W)
            film_params: List of (gamma, beta) tuples from FiLMGenerator

        Returns:
            Output logits (B, category, C, H, W)
        """
        size = x.shape
        film_idx = 0

        # First conv
        x = self.first_conv(x)
        x = F.gelu(x)
        if film_params is not None:
            gamma, beta = film_params[film_idx]
            x = apply_film(x, gamma, beta)
            film_idx += 1

        # Residual blocks with FiLM
        for block in self.hidden_convs:
            # Residual path
            tmp = block['conv1'](x)
            tmp = F.gelu(tmp)
            tmp = block['conv2'](tmp)
            tmp = F.gelu(tmp)
            tmp = block['conv3'](tmp)
            tmp = F.gelu(tmp)
            x = x + tmp

            # Apply FiLM after residual
            if film_params is not None:
                gamma, beta = film_params[film_idx]
                x = apply_film(x, gamma, beta)
                film_idx += 1

        # First FC
        x = self.first_fc(x)
        x = F.gelu(x)
        if film_params is not None:
            gamma, beta = film_params[film_idx]
            x = apply_film(x, gamma, beta)
            film_idx += 1

        # Hidden FCs with FiLM
        for fc in self.hidden_fcs:
            x = fc(x)
            x = F.gelu(x)
            if film_params is not None:
                gamma, beta = film_params[film_idx]
                x = apply_film(x, gamma, beta)
                film_idx += 1

        # Final output (no FiLM)
        x = self.final_fc(x)

        return x.reshape(size[0], self.category, self.channel, size[-2], size[-1])


# ============================================================================
# Original Masked Convolution (kept for backward compatibility)
# ============================================================================


class MaskedConv2D(nn.Conv2d):
    def __init__(
        self,
        *args,
        mask_type,
        data_channels,
        augment_channels=0,
        augment_output=True,
        **kwargs
    ):
        super(MaskedConv2D, self).__init__(*args, **kwargs)
        assert mask_type in {"A", "B"}, "mask_type must be either 'A' or 'B'"

        out_channels, in_channels, height, width = self.weight.size()
        if augment_output:
            assert (
                in_channels % (data_channels + augment_channels) == 0
                and out_channels % (data_channels + augment_channels) == 0
            ), "When augment_output is True, in_channels and out_channels must be multiples of (data_channels + augment_channels)"
        else:
            assert (
                in_channels % (data_channels + augment_channels) == 0
                and out_channels % data_channels == 0
            ), "When augment_output is False, in_channels must be a multiple of (data_channels + augment_channels) and out_channels must be a multiple of data_channels"
        y_center, x_center = height // 2, width // 2

        mask = torch.ones_like(self.weight)
        mask[:, :, y_center + 1 :, :] = 0
        mask[:, :, y_center, x_center:] = 0

        if mask_type == "A":
            meta_mask = torch.tril(
                torch.ones((data_channels, data_channels)), diagonal=-1
            )
        else:
            meta_mask = torch.tril(torch.ones((data_channels, data_channels)))

        # For conditional channels
        if augment_channels > 0:
            if augment_output:
                # Augment channel can't see data channel
                meta_mask = torch.cat(
                    [meta_mask, torch.zeros((augment_channels, data_channels))],
                    dim=0,
                )
                # Data channel can see augment channel & augment channel can see itself
                meta_mask = torch.cat(
                    [
                        meta_mask,
                        torch.ones(
                            (augment_channels + data_channels, augment_channels)
                        ),
                    ],
                    dim=1,
                )
            else:
                # Data channel can see augment channel & augment channel can see itself
                meta_mask = torch.cat(
                    [meta_mask, torch.ones((data_channels, augment_channels))],
                    dim=1,
                )

        # Tiling meta mask to match real channels
        in_tiles = in_channels // (data_channels + augment_channels)
        if augment_output:
            out_tiles = out_channels // (data_channels + augment_channels)
        else:
            out_tiles = out_channels // data_channels

        # Caution: Original code has error here, fixed it.
        mask[:, :, y_center, x_center] = meta_mask.repeat(out_tiles, in_tiles)

        self.register_buffer("mask", mask)

    def forward(self, x):
        # Apply mask during convolution instead of modifying weights in-place
        # This avoids redundant masking on every forward pass
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class MaskedResConv2D(nn.Module):
    def __init__(
        self,
        channel,
        kernel_size,
        hidden_channels,
        hidden_conv_layers,
        hidden_kernel_size,
        hidden_width,
        hidden_fc_layers,
        category,
        augment_channels=0,
    ):

        super().__init__()

        self.channel = channel
        self.category = category

        self.first_conv = MaskedConv2D(
            in_channels=channel + augment_channels,
            out_channels=2 * hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            mask_type="A",
            data_channels=channel,
            augment_channels=augment_channels,
            augment_output=True,
        )

        hidden_convs = []

        for i in range(hidden_conv_layers):
            hidden_convs.append(
                nn.Sequential(
                    MaskedConv2D(
                        in_channels=2 * hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        padding=0,
                        mask_type="B",
                        data_channels=channel,
                        augment_channels=augment_channels,
                        augment_output=True,
                    ),
                    nn.GELU(),
                    MaskedConv2D(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=hidden_kernel_size,
                        padding=(hidden_kernel_size - 1) // 2,
                        mask_type="B",
                        data_channels=channel,
                        augment_channels=augment_channels,
                        augment_output=True,
                    ),
                    nn.GELU(),
                    MaskedConv2D(
                        in_channels=hidden_channels,
                        out_channels=2 * hidden_channels,
                        kernel_size=1,
                        padding=0,
                        mask_type="B",
                        data_channels=channel,
                        augment_channels=augment_channels,
                        augment_output=True,
                    ),
                )
            )

        self.hidden_convs = nn.ModuleList(hidden_convs)

        self.first_fc = MaskedConv2D(
            in_channels=2 * hidden_channels,
            out_channels=hidden_width,
            kernel_size=1,
            mask_type="B",
            data_channels=channel,
            augment_channels=augment_channels,
            augment_output=True,
        )

        hidden_fcs = []
        for _ in range(hidden_fc_layers):
            hidden_fcs.append(
                MaskedConv2D(
                    in_channels=hidden_width,
                    out_channels=hidden_width,
                    kernel_size=1,
                    mask_type="B",
                    data_channels=channel,
                    augment_channels=augment_channels,
                    augment_output=True,
                )
            )
        self.hidden_fcs = nn.ModuleList(hidden_fcs)

        self.final_fc = MaskedConv2D(
            in_channels=hidden_width,
            out_channels=category * channel,
            kernel_size=1,
            mask_type="B",
            data_channels=channel,
            augment_channels=augment_channels,
            augment_output=False,
        )

    def forward(self, x):
        size = x.shape
        x = self.first_conv(x)
        x = F.gelu(x)

        for layer in self.hidden_convs:
            # Conv residual block
            tmp = layer(x)
            tmp = F.gelu(tmp)
            x = x + tmp

        x = self.first_fc(x)
        x = F.gelu(x)

        for layer in self.hidden_fcs:
            x = layer(x)
            x = F.gelu(x)

        x = self.final_fc(x)

        return x.reshape(size[0], self.category, self.channel, size[-2], size[-1])


class DiscretePixelCNN(nn.Module):
    def __init__(self, hparams, device="cpu"):
        super(DiscretePixelCNN, self).__init__()
        self.hparams = hparams
        self.device = device

        self.channel = 1  # Single channel for lattice
        self.category = 2  # Spin up/down
        self.augment_channels = 1  # Temperature

        # Lattice size
        size = hparams.get("size", 16)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

        self.fix_first = hparams.get("fix_first", 1)
        self.batch_size = hparams["batch_size"]
        self.num_beta = hparams["num_beta"]
        self.beta_min = hparams["beta_min"]
        self.beta_max = hparams["beta_max"]
        self.mapping = lambda x: 2 * x - 1  # Map {0,1} to {-1,1}
        self.reverse_mapping = lambda x: torch.div(x + 1, 2, rounding_mode="trunc")

        # Curriculum learning settings
        # High temp (low beta) → Low temp (high beta) progression
        self.curriculum_enabled = hparams.get("curriculum_enabled", False)
        self.curriculum_warmup_epochs = hparams.get("curriculum_warmup_epochs", 50)
        self.curriculum_start_beta_max = hparams.get(
            "curriculum_start_beta_max", self.beta_min * 1.5
        )

        # 3-Phase Curriculum Learning settings
        self.phase1_epochs = hparams.get("phase1_epochs", 50)
        self.phase1_beta_max = hparams.get("phase1_beta_max", 0.35)
        self.phase2_epochs = hparams.get("phase2_epochs", 100)
        self.tc_focus_ratio = hparams.get("tc_focus_ratio", 0.3)
        self.tc_beta_min = hparams.get("tc_beta_min", 0.38)
        self.tc_beta_max = hparams.get("tc_beta_max", 0.52)

        # Initialize MaskedResConv2D
        self.masked_conv = MaskedResConv2D(
            channel=self.channel,
            kernel_size=hparams.get("kernel_size", 7),
            hidden_channels=hparams.get("hidden_channels", 64),
            hidden_conv_layers=hparams.get("hidden_conv_layers", 5),
            hidden_kernel_size=hparams.get("hidden_kernel_size", 3),
            hidden_width=hparams.get("hidden_width", 128),
            hidden_fc_layers=hparams.get("hidden_fc_layers", 2),
            category=self.category,
            augment_channels=self.augment_channels,
        )

    def to(self, *args, **kwargs):
        """Override to() method to update self.device when model is moved to a different device."""
        self = super().to(*args, **kwargs)
        # Extract device from args/kwargs
        if args and isinstance(args[0], (torch.device, str)):
            self.device = args[0]
        elif "device" in kwargs:
            self.device = kwargs["device"]
        return self

    def sample(self, batch_size=None, T=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        sample = torch.zeros(batch_size, self.channel, self.size[0], self.size[1]).to(
            self.device
        )
        if T is not None:
            # (B, C) -> (B, C, H, W)
            T = T.to(self.device)
            if T.dim() == 1:
                T = T.unsqueeze(1)

            T_expanded = einops.repeat(
                T, "b c -> b c h w", h=self.size[0], w=self.size[1]
            )
            sample = torch.cat([sample, T_expanded], dim=1)

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                # Fix the first element of the samples to be a fixed value
                if self.fix_first is not None and i == 0 and j == 0:
                    if T is not None:
                        # Caution: original code has potential bug here, fixed it.
                        sample[:, : self.channel, 0, 0] = self.fix_first
                    else:
                        sample[:, :, 0, 0] = self.fix_first
                    continue

                # Compute predictions for all channels at once (B, Cat, C, H, W)
                # Optimization: move forward pass outside of channel loop to avoid redundant computation
                unnormalized = self.masked_conv.forward(sample)

                for k in range(self.channel):
                    # Use multinomial instead of argmax to allow stochastic sampling
                    sample[:, k, i, j] = (
                        torch.multinomial(
                            torch.softmax(unnormalized[:, :, k, i, j], dim=1),
                            1,  # num_samples=1
                        )
                        .squeeze()
                        .float()
                    )

        if T is not None:
            # Caution: original code has potential bug here, fixed it.
            sample = sample[:, : self.channel, :, :]

        sample = self.mapping(sample)  # Map {0,1} to {-1,1}

        return sample

    def log_prob(self, sample, T=None):
        # sample to {0,1}
        sample = self.reverse_mapping(sample)

        if self.fix_first is not None:
            assert (
                sample[:, :, 0, 0] == self.fix_first
            ).all(), "The first element of the sample does not match fix_first value."

        if T is not None:
            # (B, C) -> (B, C, H, W)
            T = T.to(self.device)
            if T.dim() == 1:
                T = T.unsqueeze(1)

            T_expanded = einops.repeat(
                T, "b c -> b c h w", h=sample.shape[2], w=sample.shape[3]
            )
            sample = torch.cat([sample, T_expanded], dim=1)

        unnormalized = self.masked_conv.forward(sample)  # (B, Cat, C, H, W)
        # Use log_softmax for numerical stability (avoids log(softmax()) underflow)
        log_prob = F.log_softmax(unnormalized, dim=1)

        if T is not None:
            # Caution: original code has potential bug here, fixed it.
            sample = sample[:, : self.channel, :, :]

        # (B, 1, C, H, W)
        log_prob_selected = log_prob.gather(
            1, sample.long().unsqueeze(1)
        )  # Find the log probabilities of the selected categories

        # (B, C, H * W)
        log_prob_selected = einops.rearrange(
            log_prob_selected, "b 1 c h w -> b c (h w)"
        )

        if self.fix_first is not None:
            log_prob_selected = log_prob_selected[..., 1:]  # Remove the first element

        # (B, 1)
        log_prob_sum = einops.reduce(log_prob_selected, "b c hw -> b 1", "sum")

        return log_prob_sum


# ============================================================================
# New FiLM-based PixelCNN (improved temperature conditioning)
# ============================================================================


class DiscretePixelCNNFiLM(nn.Module):
    """
    Discrete PixelCNN with Temperature Embedding + FiLM conditioning.

    Key improvements over DiscretePixelCNN:
    1. Temperature Embedding: Converts T to rich high-dimensional representation
       using sinusoidal encoding + MLP (like positional encoding in Transformers)

    2. FiLM Conditioning: Temperature modulates features via γ(T) * x + β(T)
       instead of simple concatenation. This is more expressive and allows
       temperature to affect the feature computation multiplicatively.

    Benefits for critical temperature learning:
    - Fine temperature differences (e.g., T=2.25 vs T=2.30) become distinguishable
    - No need to know Tc beforehand - model learns temperature sensitivity
    - Automatic differentiation fully preserved for all gradients
    """

    def __init__(self, hparams, device="cpu"):
        super().__init__()
        self.hparams = hparams
        self.device = device

        self.channel = 1  # Single channel for lattice
        self.category = 2  # Spin up/down

        # Lattice size
        size = hparams.get("size", 16)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

        self.fix_first = hparams.get("fix_first", 1)
        self.batch_size = hparams["batch_size"]
        self.num_beta = hparams["num_beta"]
        self.beta_min = hparams["beta_min"]
        self.beta_max = hparams["beta_max"]
        self.mapping = lambda x: 2 * x - 1  # Map {0,1} to {-1,1}
        self.reverse_mapping = lambda x: torch.div(x + 1, 2, rounding_mode="trunc")

        # Curriculum learning settings (kept for compatibility)
        self.curriculum_enabled = hparams.get("curriculum_enabled", False)
        self.curriculum_warmup_epochs = hparams.get("curriculum_warmup_epochs", 50)
        self.curriculum_start_beta_max = hparams.get(
            "curriculum_start_beta_max", self.beta_min * 1.5
        )
        self.phase1_epochs = hparams.get("phase1_epochs", 50)
        self.phase1_beta_max = hparams.get("phase1_beta_max", 0.35)
        self.phase2_epochs = hparams.get("phase2_epochs", 100)
        self.tc_focus_ratio = hparams.get("tc_focus_ratio", 0.3)
        self.tc_beta_min = hparams.get("tc_beta_min", 0.38)
        self.tc_beta_max = hparams.get("tc_beta_max", 0.52)

        # Temperature embedding settings
        self.temp_embed_dim = hparams.get("temp_embed_dim", 64)
        self.temp_hidden_dim = hparams.get("temp_hidden_dim", 128)
        self.temp_num_freqs = hparams.get("temp_num_freqs", 32)

        # Architecture params
        hidden_channels = hparams.get("hidden_channels", 64)
        hidden_conv_layers = hparams.get("hidden_conv_layers", 5)
        hidden_width = hparams.get("hidden_width", 128)
        hidden_fc_layers = hparams.get("hidden_fc_layers", 2)

        # Initialize MaskedResConv2D with FiLM support
        self.masked_conv = MaskedResConv2DFiLM(
            channel=self.channel,
            kernel_size=hparams.get("kernel_size", 7),
            hidden_channels=hidden_channels,
            hidden_conv_layers=hidden_conv_layers,
            hidden_kernel_size=hparams.get("hidden_kernel_size", 3),
            hidden_width=hidden_width,
            hidden_fc_layers=hidden_fc_layers,
            category=self.category,
        )

        # Temperature Embedding: T → high-dimensional representation
        self.temp_embedding = TemperatureEmbedding(
            embed_dim=self.temp_embed_dim,
            hidden_dim=self.temp_hidden_dim,
            num_freqs=self.temp_num_freqs,
        )

        # FiLM Generator: temp_embedding → (γ, β) for each layer
        film_feature_dims = self.masked_conv.get_film_feature_dims()
        self.film_generator = FiLMGenerator(
            embed_dim=self.temp_embed_dim,
            num_layers=len(film_feature_dims),
            feature_dims=film_feature_dims,
        )

    def to(self, *args, **kwargs):
        """Override to() method to update self.device when model is moved."""
        self = super().to(*args, **kwargs)
        if args and isinstance(args[0], (torch.device, str)):
            self.device = args[0]
        elif "device" in kwargs:
            self.device = kwargs["device"]
        return self

    def _get_film_params(self, T: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute FiLM parameters from temperature.

        Args:
            T: Temperature tensor (B,) or (B, 1)

        Returns:
            List of (gamma, beta) tuples for each layer
        """
        T = T.to(self.device)
        if T.dim() == 2:
            T = T.squeeze(-1)  # (B, 1) -> (B,)

        # T → embedding → FiLM params
        temp_embed = self.temp_embedding(T)  # (B, embed_dim)
        film_params = self.film_generator(temp_embed)  # List of (γ, β)

        return film_params

    def sample(self, batch_size=None, T=None):
        """
        Generate samples from the model at given temperature(s).

        Args:
            batch_size: Number of samples to generate
            T: Temperature tensor (B,) - required for conditional sampling

        Returns:
            Samples tensor (B, 1, H, W) with values in {-1, 1}
        """
        batch_size = batch_size if batch_size is not None else self.batch_size

        if T is None:
            raise ValueError("Temperature T is required for DiscretePixelCNNFiLM")

        T = T.to(self.device)
        if T.dim() == 2:
            T = T.squeeze(-1)

        # Handle scalar T for all samples
        if T.numel() == 1:
            T = T.expand(batch_size)

        # Get FiLM parameters from temperature
        film_params = self._get_film_params(T)

        # Initialize sample (no temperature channel needed - using FiLM instead)
        sample = torch.zeros(batch_size, self.channel, self.size[0], self.size[1]).to(
            self.device
        )

        # Autoregressive sampling
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                # Fix first spin if specified
                if self.fix_first is not None and i == 0 and j == 0:
                    sample[:, :, 0, 0] = self.fix_first
                    continue

                # Forward pass with FiLM conditioning
                unnormalized = self.masked_conv(sample, film_params)

                for k in range(self.channel):
                    # Stochastic sampling from categorical distribution
                    probs = torch.softmax(unnormalized[:, :, k, i, j], dim=1)
                    sample[:, k, i, j] = torch.multinomial(probs, 1).squeeze().float()

        # Map {0,1} to {-1,1}
        sample = self.mapping(sample)

        return sample

    def log_prob(self, sample, T=None):
        """
        Compute log probability of samples under the model at given temperature.

        Args:
            sample: Samples tensor (B, 1, H, W) with values in {-1, 1}
            T: Temperature tensor (B,) - required

        Returns:
            Log probabilities tensor (B, 1)

        Note: This function is fully differentiable w.r.t. model parameters.
              The gradient ∂log_prob/∂θ flows through:
              T → TemperatureEmbedding → FiLMGenerator → MaskedResConv2DFiLM → log_prob
        """
        if T is None:
            raise ValueError("Temperature T is required for DiscretePixelCNNFiLM")

        # Convert sample to {0,1}
        sample = self.reverse_mapping(sample)

        if self.fix_first is not None:
            assert (
                sample[:, :, 0, 0] == self.fix_first
            ).all(), "First element must match fix_first value"

        T = T.to(self.device)
        if T.dim() == 2:
            T = T.squeeze(-1)

        # Get FiLM parameters from temperature
        film_params = self._get_film_params(T)

        # Forward pass with FiLM conditioning
        unnormalized = self.masked_conv(sample, film_params)  # (B, Cat, C, H, W)

        # Log softmax for numerical stability
        log_prob = F.log_softmax(unnormalized, dim=1)

        # Select log probabilities of actual samples
        log_prob_selected = log_prob.gather(1, sample.long().unsqueeze(1))

        # Reshape: (B, 1, C, H, W) -> (B, C, H*W)
        log_prob_selected = einops.rearrange(
            log_prob_selected, "b 1 c h w -> b c (h w)"
        )

        # Remove first element if fixed
        if self.fix_first is not None:
            log_prob_selected = log_prob_selected[..., 1:]

        # Sum over all positions: (B, 1)
        log_prob_sum = einops.reduce(log_prob_selected, "b c hw -> b 1", "sum")

        return log_prob_sum
