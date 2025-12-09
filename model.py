import torch
from torch import nn
import torch.nn.functional as F
import einops
import numpy as np
from typing import Tuple



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
        for _ in range(hidden_conv_layers):
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


# ============================================================================
# Normalizing Flow Components for VaTD
# ============================================================================


def create_checkerboard_mask(H: int, W: int, parity: int = 0) -> torch.Tensor:
    """
    2D 격자에 대한 checkerboard 마스크 생성.

    Args:
        H: 격자 높이
        W: 격자 너비
        parity: 0이면 (i+j)가 짝수인 위치, 1이면 홀수인 위치 선택

    Returns:
        Boolean 텐서 (H*W,) - True인 위치가 transform될 위치

    Example (4x4, parity=0):
        [[T, F, T, F],
         [F, T, F, T],
         [T, F, T, F],
         [F, T, F, T]]
    """
    i = torch.arange(H).view(-1, 1).expand(H, W)
    j = torch.arange(W).view(1, -1).expand(H, W)
    mask = ((i + j) % 2 == parity).flatten()
    return mask


class Dequantizer:
    """
    Discrete {-1, +1} 스핀과 continuous 값 사이의 변환 처리.

    Training: discrete 값에 uniform noise 추가
    Inference: continuous 값을 threshold하여 discrete로 변환

    Attributes:
        noise_scale: dequantization noise 크기 (delta)
    """

    def __init__(self, noise_scale: float = 0.05):
        """
        Args:
            noise_scale: noise 범위 [-delta, delta]의 delta 값
        """
        self.noise_scale = noise_scale

    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training용: discrete 값에 uniform noise 추가.

        Args:
            x: {-1, +1} 값을 가진 discrete 텐서

        Returns:
            대략 [-1-delta, 1+delta] 범위의 continuous 텐서
        """
        noise = torch.empty_like(x).uniform_(-self.noise_scale, self.noise_scale)
        return x + noise

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference용: continuous 값을 discrete로 변환.

        Args:
            x: Continuous 텐서

        Returns:
            {-1, +1} 값을 가진 discrete 텐서
        """
        return torch.sign(x).clamp(-1, 1)  # sign(0) = 0 처리


class TemperatureConditioner(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) style temperature conditioning.

    Takes temperature T as input and outputs scale (gamma) and shift (beta)
    parameters for feature modulation in coupling networks.

    Attributes:
        net: MLP network that maps T -> (gamma, beta)
    """

    def __init__(self, hidden_dim: int, output_dim: int):
        """
        Args:
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (= hidden_channels of coupling network)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * output_dim)  # Output gamma and beta together
        )

    def forward(self, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            T: Temperature tensor of shape (B, 1)

        Returns:
            gamma: Scale parameters (B, output_dim)
            beta: Shift parameters (B, output_dim)
        """
        params = self.net(T)
        gamma, beta = params.chunk(2, dim=-1)
        # Initialize gamma around 1, beta around 0 for stable training start
        gamma = 1.0 + 0.1 * gamma
        beta = 0.1 * beta
        return gamma, beta


class ConditionalCouplingNet(nn.Module):
    """
    CNN-based coupling network with FiLM temperature conditioning.

    Takes masked input features and temperature, outputs affine transform
    parameters (log_scale, shift) for the unmasked features.

    Attributes:
        temp_cond: Temperature conditioning module
        input_proj: Input projection layer
        hidden_layers: ResNet-style hidden layers
        output_proj: Output projection layer
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_hidden_layers: int = 2,
        kernel_size: int = 3,
        H: int = 16,
        W: int = 16,
    ):
        """
        Args:
            hidden_channels: Number of channels in hidden layers
            num_hidden_layers: Number of hidden layers
            kernel_size: Convolution kernel size
            H, W: Lattice dimensions
        """
        super().__init__()
        self.H = H
        self.W = W

        # Temperature conditioning
        self.temp_cond = TemperatureConditioner(
            hidden_dim=hidden_channels,
            output_dim=hidden_channels
        )

        # Initial projection: 1 channel -> hidden_channels
        self.input_proj = nn.Conv2d(1, hidden_channels, 1)

        # Hidden layers (ResNet-style)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size,
                         padding=kernel_size // 2),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size,
                         padding=kernel_size // 2),
            ))

        # Output projection: hidden_channels -> 2 (log_scale, shift)
        self.output_proj = nn.Conv2d(hidden_channels, 2, 1)

        # Zero initialization for stable training (identity transform at start)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x_masked: torch.Tensor,
        T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_masked: Masked input (B, 1, H, W) - unmasked positions are zeroed
            T: Temperature (B, 1)

        Returns:
            log_scale: (B, 1, H, W) log-scale parameters
            shift: (B, 1, H, W) shift parameters
        """
        # Compute FiLM parameters
        gamma, beta = self.temp_cond(T)  # (B, hidden_channels)
        gamma = gamma.view(-1, gamma.shape[-1], 1, 1)  # (B, C, 1, 1)
        beta = beta.view(-1, beta.shape[-1], 1, 1)

        # Process input
        h = self.input_proj(x_masked)  # (B, hidden_channels, H, W)

        # Apply FiLM conditioning
        h = gamma * h + beta

        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            h = h + layer(h)
            h = F.silu(h)

        # Output
        params = self.output_proj(h)  # (B, 2, H, W)
        log_scale, shift = params.chunk(2, dim=1)  # Each (B, 1, H, W)

        # Constrain log_scale for numerical stability
        log_scale = torch.tanh(log_scale) * 2.0  # Range [-2, 2]

        return log_scale, shift


class CheckerboardAffineCoupling(nn.Module):
    """
    Affine coupling layer with checkerboard masking for 2D lattice.

    Transform: x_transform = x_transform * exp(s) + t
    where (s, t) = net(x_frozen, T)

    Attributes:
        H, W: Lattice dimensions
        mask: Checkerboard mask (True positions are transformed)
        coupling_net: Network computing transform parameters
    """

    def __init__(
        self,
        H: int,
        W: int,
        parity: int,
        hidden_channels: int = 64,
        num_hidden_layers: int = 2,
    ):
        """
        Args:
            H, W: Lattice dimensions
            parity: Mask parity (0 or 1)
            hidden_channels: Hidden channels in coupling network
            num_hidden_layers: Number of hidden layers in coupling network
        """
        super().__init__()
        self.H = H
        self.W = W

        # Create and register checkerboard mask
        self.register_buffer('mask', create_checkerboard_mask(H, W, parity))

        # Coupling network
        self.coupling_net = ConditionalCouplingNet(
            hidden_channels=hidden_channels,
            num_hidden_layers=num_hidden_layers,
            H=H,
            W=W,
        )

    def forward(
        self,
        x: torch.Tensor,
        T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transform: z = f(x)

        Args:
            x: Input tensor (B, 1, H, W)
            T: Temperature (B, 1)

        Returns:
            z: Transformed tensor (B, 1, H, W)
            log_det: Log determinant of Jacobian (B,)
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)  # (B, H*W)

        # Split by mask
        mask_float = self.mask.float()
        inv_mask_float = (~self.mask).float()

        x_frozen = x_flat * inv_mask_float  # Frozen features
        x_transform = x_flat * mask_float   # Features to transform

        # Reshape for CNN
        x_frozen_2d = x_frozen.view(B, 1, self.H, self.W)

        # Compute transform parameters
        log_scale, shift = self.coupling_net(x_frozen_2d, T)
        log_scale_flat = log_scale.view(B, -1) * mask_float
        shift_flat = shift.view(B, -1) * mask_float

        # Apply affine transform
        z_transform = x_transform * torch.exp(log_scale_flat) + shift_flat
        z_flat = z_transform + x_frozen
        z = z_flat.view(B, 1, self.H, self.W)

        # Log determinant (sum of log_scales for transformed features)
        log_det = log_scale_flat.sum(dim=-1)

        return z, log_det

    def inverse(self, z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform: x = f^{-1}(z)

        Args:
            z: Transformed tensor (B, 1, H, W)
            T: Temperature (B, 1)

        Returns:
            x: Original tensor (B, 1, H, W)
        """
        B = z.shape[0]
        z_flat = z.view(B, -1)

        mask_float = self.mask.float()
        inv_mask_float = (~self.mask).float()

        z_frozen = z_flat * inv_mask_float
        z_transform = z_flat * mask_float

        z_frozen_2d = z_frozen.view(B, 1, self.H, self.W)
        log_scale, shift = self.coupling_net(z_frozen_2d, T)
        log_scale_flat = log_scale.view(B, -1) * mask_float
        shift_flat = shift.view(B, -1) * mask_float

        # Inverse affine transform
        x_transform = (z_transform - shift_flat) * torch.exp(-log_scale_flat)
        x_flat = x_transform + z_frozen

        return x_flat.view(B, 1, self.H, self.W)


class CheckerboardFlowModel(nn.Module):
    """
    Complete Normalizing Flow model for 2D Ising model.

    Features:
    - Checkerboard-masked affine coupling layers
    - FiLM-style temperature conditioning
    - Dequantization for discrete spin handling

    Interface matches DiscretePixelCNN:
    - sample(batch_size, T) -> (B, 1, H, W) in {-1, +1}
    - log_prob(samples, T) -> (B, 1)

    Attributes:
        size: Lattice size (H, W)
        batch_size: Batch size per temperature
        num_beta: Number of temperature samples per batch
        beta_min, beta_max: Inverse temperature range
        coupling_layers: List of coupling layers
        dequantizer: Dequantization handler
    """

    def __init__(self, hparams: dict, device: str = "cpu"):
        """
        Args:
            hparams: Hyperparameter dictionary
                - size: int or (H, W) tuple, lattice size
                - batch_size: int, samples per temperature
                - num_beta: int, temperature samples per batch
                - beta_min, beta_max: float, inverse temperature range
                - num_flow_layers: int, number of coupling layers (default: 8)
                - hidden_channels: int, hidden channels (default: 64)
                - num_hidden_layers: int, hidden layers in coupling net (default: 2)
                - dequant_noise: float, dequantization noise scale (default: 0.05)
            device: Device string
        """
        super().__init__()
        self.hparams = hparams
        self.device = device

        # Lattice configuration
        size = hparams.get("size", 16)
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.H, self.W = self.size

        # Training parameters (same interface as DiscretePixelCNN)
        self.batch_size = hparams["batch_size"]
        self.num_beta = hparams["num_beta"]
        self.beta_min = hparams["beta_min"]
        self.beta_max = hparams["beta_max"]

        # Flow architecture parameters
        num_flow_layers = hparams.get("num_flow_layers", 8)
        hidden_channels = hparams.get("hidden_channels", 64)
        num_hidden_layers = hparams.get("num_hidden_layers", 2)

        # Dequantization
        self.dequant_noise = hparams.get("dequant_noise", 0.05)
        self.dequantizer = Dequantizer(noise_scale=self.dequant_noise)

        # Build coupling layers (alternating parity)
        self.coupling_layers = nn.ModuleList()
        for i in range(num_flow_layers):
            parity = i % 2  # Alternate 0, 1, 0, 1, ...
            layer = CheckerboardAffineCoupling(
                H=self.H,
                W=self.W,
                parity=parity,
                hidden_channels=hidden_channels,
                num_hidden_layers=num_hidden_layers,
            )
            self.coupling_layers.append(layer)

        # Base distribution range (accounting for dequantization)
        self.base_half_width = 1.0 + self.dequant_noise

    def _get_base_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under base distribution (Uniform).

        Args:
            z: Latent tensor (B, 1, H, W)

        Returns:
            Log probability (B,)
        """
        B = z.shape[0]
        z_flat = z.view(B, -1)

        # Uniform(-half_width, half_width)
        half_width = self.base_half_width

        # Uniform distribution log prob: log(1 / (2 * half_width)) * num_dims
        num_dims = z_flat.shape[-1]
        log_prob_per_dim = -torch.log(torch.tensor(2 * half_width, device=z.device))
        log_prob = log_prob_per_dim * num_dims

        # Apply penalty for out-of-bounds values (soft boundary)
        # This is more stable than hard -inf boundaries
        out_of_bounds_penalty = torch.relu(z_flat.abs() - half_width).pow(2).sum(dim=-1)
        log_prob = log_prob - out_of_bounds_penalty

        return log_prob

    def _sample_base(self, batch_size: int) -> torch.Tensor:
        """
        Sample from base distribution.

        Args:
            batch_size: Number of samples

        Returns:
            Sample tensor (B, 1, H, W)
        """
        shape = (batch_size, 1, self.H, self.W)
        half_width = self.base_half_width
        return torch.empty(shape, device=self.device).uniform_(-half_width, half_width)

    def _prepare_temperature(
        self,
        T: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Prepare temperature tensor for conditioning.

        Args:
            T: Temperature tensor (B,) or (B, 1) or None
            batch_size: Batch size

        Returns:
            Prepared temperature tensor (B, 1)
        """
        if T is None:
            # Default: middle of range
            mid_T = 0.5 * (1 / self.beta_min + 1 / self.beta_max)
            T = torch.full((batch_size, 1), mid_T, device=self.device)
        else:
            T = T.to(self.device)
            if T.dim() == 1:
                T = T.unsqueeze(1)
        return T

    def forward_flow(
        self,
        x: torch.Tensor,
        T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: x -> z (data space to latent space)

        Args:
            x: Input tensor (B, 1, H, W)
            T: Temperature (B, 1)

        Returns:
            z: Latent tensor (B, 1, H, W)
            log_det_sum: Total log determinant sum (B,)
        """
        z = x
        log_det_sum = torch.zeros(x.shape[0], device=x.device)

        for layer in self.coupling_layers:
            z, log_det = layer(z, T)
            log_det_sum = log_det_sum + log_det

        return z, log_det_sum

    def inverse_flow(self, z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: z -> x (latent space to data space)

        Args:
            z: Latent tensor (B, 1, H, W)
            T: Temperature (B, 1)

        Returns:
            x: Reconstructed data tensor (B, 1, H, W)
        """
        x = z
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x, T)
        return x

    def sample(
        self,
        batch_size: int = None,
        T: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Sample discrete spin configurations.

        Args:
            batch_size: Number of samples (None uses self.batch_size)
            T: Temperature tensor (B,) or (B, 1)

        Returns:
            Tensor of shape (B, 1, H, W) with values in {-1, +1}
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        T = self._prepare_temperature(T, batch_size)

        # Sample from base distribution
        z = self._sample_base(batch_size)

        # Inverse flow: z -> x (continuous)
        x_continuous = self.inverse_flow(z, T)

        # Convert to discrete
        x_discrete = self.dequantizer.quantize(x_continuous)

        return x_discrete

    def sample_continuous(
        self,
        batch_size: int = None,
        T: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Sample continuous values (for gradient flow during training).

        Args:
            batch_size: Number of samples
            T: Temperature tensor

        Returns:
            Continuous tensor (B, 1, H, W)
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        T = self._prepare_temperature(T, batch_size)

        z = self._sample_base(batch_size)
        x_continuous = self.inverse_flow(z, T)

        return x_continuous

    def log_prob(
        self,
        samples: torch.Tensor,
        T: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute log probability of samples.

        Args:
            samples: Sample tensor (B, 1, H, W) - discrete {-1, +1} or continuous
            T: Temperature tensor (B,) or (B, 1)

        Returns:
            Log probability (B, 1)
        """
        batch_size = samples.shape[0]
        T = self._prepare_temperature(T, batch_size)

        # Apply dequantization during training
        if self.training:
            x_dequant = self.dequantizer.dequantize(samples)
        else:
            # At inference, use continuous values directly
            x_dequant = samples.float()

        # Forward flow: x -> z
        z, log_det = self.forward_flow(x_dequant, T)

        # Log probability: log p(x) = log p(z) + log |det df/dx|
        log_prob_base = self._get_base_log_prob(z)
        log_prob = log_prob_base + log_det

        return log_prob.unsqueeze(-1)  # (B, 1)

    def to(self, *args, **kwargs):
        """Override to update self.device."""
        result = super().to(*args, **kwargs)
        if args and isinstance(args[0], (torch.device, str)):
            self.device = str(args[0])
        elif 'device' in kwargs:
            self.device = str(kwargs['device'])
        return result


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
        elif 'device' in kwargs:
            self.device = kwargs['device']
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
        prob = torch.softmax(unnormalized, dim=1)

        if T is not None:
            # Caution: original code has potential bug here, fixed it.
            sample = sample[:, : self.channel, :, :]

        # (B, 1, C, H, W)
        log_prob_selected = torch.log(
            prob.gather(
                1, sample.long().unsqueeze(1)
            )  # Find the probabilities of the selected categories
        )

        # (B, C, H * W)
        log_prob_selected = einops.rearrange(
            log_prob_selected, "b 1 c h w -> b c (h w)"
        )

        if self.fix_first is not None:
            log_prob_selected = log_prob_selected[..., 1:]  # Remove the first element

        # (B, 1)
        log_prob_sum = einops.reduce(log_prob_selected, "b c hw -> b 1", "sum")

        return log_prob_sum

    def sample_gumbel_softmax(self, batch_size=None, T=None, temperature=1.0, hard=True):
        """
        Sample using Gumbel-Softmax (continuous relaxation of discrete sampling).

        This enables gradient flow through sampling via reparameterization trick.

        Args:
            batch_size (int): Number of samples to generate
            T (torch.Tensor): Temperature values for conditional generation
            temperature (float): Gumbel-Softmax temperature (tau)
                - Lower temperature → closer to discrete (more deterministic)
                - Higher temperature → more uniform (more stochastic)
                - Typically annealed from 1.0 to 0.1 during training
            hard (bool): If True, use straight-through estimator
                - Forward pass: discrete (one-hot)
                - Backward pass: continuous (soft probabilities)

        Returns:
            torch.Tensor: Samples in {-1, +1} space with shape (B, C, H, W)
                If hard=False, returns soft samples (weighted average)
                If hard=True, returns hard samples but with soft gradients
        """
        batch_size = batch_size if batch_size is not None else self.batch_size

        # Initialize with zeros (will be filled sequentially)
        sample = torch.zeros(batch_size, self.channel, self.size[0], self.size[1]).to(
            self.device
        )

        # Temperature conditioning
        if T is not None:
            T = T.to(self.device)
            if T.dim() == 1:
                T = T.unsqueeze(1)

            T_expanded = einops.repeat(
                T, "b c -> b c h w", h=self.size[0], w=self.size[1]
            )
            sample = torch.cat([sample, T_expanded], dim=1)

        # Sequential sampling with Gumbel-Softmax
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                # Fix first element if needed
                if self.fix_first is not None and i == 0 and j == 0:
                    if T is not None:
                        sample[:, : self.channel, 0, 0] = self.fix_first
                    else:
                        sample[:, :, 0, 0] = self.fix_first
                    continue

                # Get logits: (B, Cat, C, H, W)
                logits = self.masked_conv.forward(sample)

                for k in range(self.channel):
                    # Extract logits for current position: (B, Cat)
                    logits_ijk = logits[:, :, k, i, j]

                    # Sample Gumbel noise
                    gumbel_noise = -torch.log(-torch.log(
                        torch.rand_like(logits_ijk) + 1e-20
                    ) + 1e-20)

                    # Gumbel-Softmax: (B, Cat)
                    y_soft = F.softmax((logits_ijk + gumbel_noise) / temperature, dim=1)

                    if hard:
                        # Straight-through estimator
                        # Forward: one-hot (discrete)
                        y_hard_indices = y_soft.argmax(dim=1)
                        y_hard = F.one_hot(y_hard_indices, num_classes=self.category).float()

                        # Backward: soft probabilities
                        # This gradient trick allows gradients to flow through discrete sampling
                        y = y_hard - y_soft.detach() + y_soft
                    else:
                        # Pure soft (continuous relaxation)
                        y = y_soft

                    # Convert from one-hot/soft to scalar value
                    # For hard mode: effectively argmax
                    # For soft mode: weighted average (0 * p(0) + 1 * p(1))
                    if hard:
                        sample[:, k, i, j] = y_hard_indices.float()
                    else:
                        # Weighted sum: E[value] = sum_c (c * p(c))
                        values = torch.arange(self.category, device=self.device).float()
                        sample[:, k, i, j] = (y * values.unsqueeze(0)).sum(dim=1)

        # Remove temperature channel if present
        if T is not None:
            sample = sample[:, : self.channel, :, :]

        # Map {0,1} to {-1,1}
        sample = self.mapping(sample)

        return sample

    def get_logits(self, sample, T=None):
        """
        Get logits for given samples (helper method for Gumbel-Softmax training).

        Args:
            sample (torch.Tensor): Samples in {-1, +1} space
            T (torch.Tensor): Temperature values

        Returns:
            torch.Tensor: Logits with shape (B, Cat, C, H, W)
        """
        # Map {-1,1} to {0,1}
        sample = self.reverse_mapping(sample)

        if T is not None:
            T = T.to(self.device)
            if T.dim() == 1:
                T = T.unsqueeze(1)

            T_expanded = einops.repeat(
                T, "b c -> b c h w", h=sample.shape[2], w=sample.shape[3]
            )
            sample = torch.cat([sample, T_expanded], dim=1)

        logits = self.masked_conv.forward(sample)  # (B, Cat, C, H, W)

        return logits
