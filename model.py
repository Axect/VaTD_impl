import torch
from torch import nn
import torch.nn.functional as F
import einops
import numpy as np
from typing import Tuple, List, Optional
import math

# Optional MHC import - fallback to standard fusion if not available
try:
    from mhc import MHCLayer
    MHC_AVAILABLE = True
except ImportError:
    MHC_AVAILABLE = False


class HCLayer(nn.Module):
    """
    HC (Hyper-Connections) Layer - Unconstrained version.

    Unlike MHCLayer which constrains H_res to be doubly stochastic via Sinkhorn-Knopp,
    HCLayer uses an unconstrained H_res matrix. This can lead to signal
    explosion/vanishing in deep networks but provides more expressive power.

    Interface matches MHCLayer for easy swapping.

    Args:
        hidden_dim: The hidden dimension (C).
        expansion_rate: The expansion rate (n).
        alpha_init: Initialization scale for H_res perturbation from identity.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        alpha_init: float = 0.01,
        **kwargs,  # Accept but ignore MHCLayer-specific args (sinkhorn_iters, use_dynamic_h, etc.)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate

        n = expansion_rate

        # RMSNorm weight
        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.rmsnorm_eps = 1e-5

        # H_res: unconstrained, initialized near identity for stability
        self.H_res = nn.Parameter(torch.eye(n) + alpha_init * torch.randn(n, n))

        # H_pre and H_post
        self.H_pre = nn.Parameter(torch.zeros(n))
        self.H_post = nn.Parameter(torch.zeros(n))

    def _rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm: x / sqrt(mean(x^2) + eps) * weight"""
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.rmsnorm_eps)
        return x / rms * self.rmsnorm_weight

    def forward(self, x_expanded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_expanded: Input tensor of shape [B, n, C]

        Returns:
            Output tensor of shape [B, n, C]
        """
        B, n, C = x_expanded.shape
        assert n == self.expansion_rate
        assert C == self.hidden_dim

        # H_pre/H_post activation (matching MHCLayer behavior)
        H_pre = torch.sigmoid(self.H_pre)  # [n]
        H_post = 2.0 * torch.sigmoid(self.H_post)  # [n]

        # Aggregate input: x_agg = sum(H_pre[k] * x[:, k, :])
        x_agg = torch.einsum('k,bkc->bc', H_pre, x_expanded)  # [B, C]

        # Apply RMSNorm
        y_norm = self._rmsnorm(x_agg)  # [B, C]

        # Mix streams with unconstrained H_res
        h_mixed = torch.einsum('ij,bjc->bic', self.H_res, x_expanded)  # [B, n, C]

        # Add post-processed output
        output = h_mixed + torch.einsum('k,bc->bkc', H_post, y_norm)  # [B, n, C]

        return output


class HCFusion2D(nn.Module):
    """
    2D Spatial wrapper for HC/mHC Layer fusion.

    Supports both MHCLayer (manifold-constrained) and HCLayer (unconstrained)
    through the manifold_constraint parameter.

    Args:
        hidden_channels: Number of channels in conv features
        num_layers: Number of skip connection layers to fuse
        use_dynamic_h: If True, uses input-dependent H values (MHCLayer only)
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations (MHCLayer only)
        aggregation: How to aggregate n outputs - 'sum', 'mean', or 'last'
        manifold_constraint: If True, use MHCLayer (doubly stochastic H_res).
                            If False, use HCLayer (unconstrained H_res).
    """

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        use_dynamic_h: bool = False,
        sinkhorn_iters: int = 10,
        aggregation: str = 'sum',
        manifold_constraint: bool = True,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.manifold_constraint = manifold_constraint

        if manifold_constraint:
            if not MHC_AVAILABLE:
                raise ImportError(
                    "MHCLayer not available. Install mHC.cu: pip install -e ./mHC.cu/"
                )
            self.layer = MHCLayer(
                hidden_dim=hidden_channels,
                expansion_rate=num_layers,
                sinkhorn_iters=sinkhorn_iters,
                use_dynamic_h=use_dynamic_h,
            )
        else:
            self.layer = HCLayer(
                hidden_dim=hidden_channels,
                expansion_rate=num_layers,
            )

    def forward(self, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple skip connection features using HC/mHC Layer.

        Args:
            skip_features: List of n tensors, each [B, C, H, W]

        Returns:
            Fused features [B, C, H, W]
        """
        assert len(skip_features) == self.num_layers, \
            f"Expected {self.num_layers} features, got {len(skip_features)}"

        B, C, H, W = skip_features[0].shape

        # Stack: [B, n, C, H, W]
        stacked = torch.stack(skip_features, dim=1)

        # Reshape for Layer: [B, n, C, H, W] -> [B*H*W, n, C]
        x = einops.rearrange(stacked, 'b n c h w -> (b h w) n c')

        # Apply HC/mHC Layer: [B*H*W, n, C] -> [B*H*W, n, C]
        y = self.layer(x)

        # Aggregate across n dimension based on strategy
        if self.aggregation == 'sum':
            y = y.sum(dim=1)  # [B*H*W, C]
        elif self.aggregation == 'mean':
            y = y.mean(dim=1)  # [B*H*W, C]
        elif self.aggregation == 'last':
            y = y[:, -1, :]  # [B*H*W, C]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Reshape back: [B*H*W, C] -> [B, C, H, W]
        out = einops.rearrange(y, '(b h w) c -> b c h w', b=B, h=H, w=W)

        return out


class MHCFusion2D(nn.Module):
    """
    2D Spatial wrapper for MHCLayer (Manifold-Constrained Hyper-Connections).

    Transforms 2D conv features to work with MHCLayer which expects [B, n, C] input.
    Each spatial position (h, w) is treated as an independent sequence position.

    Args:
        hidden_channels: Number of channels in conv features (C in conv = hidden_dim in MHC)
        num_layers: Number of skip connection layers to fuse (n = expansion_rate in MHC)
        use_dynamic_h: If True, uses input-dependent H values (more expressive but slower)
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations for doubly stochastic constraint
        aggregation: How to aggregate n outputs - 'sum', 'mean', or 'last'
    """

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        use_dynamic_h: bool = False,
        sinkhorn_iters: int = 10,
        aggregation: str = 'sum',
    ):
        super().__init__()

        if not MHC_AVAILABLE:
            raise ImportError(
                "MHCLayer not available. Install mHC.cu: pip install -e ./mHC.cu/"
            )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.aggregation = aggregation

        # MHCLayer expects hidden_dim = C, expansion_rate = n
        self.mhc = MHCLayer(
            hidden_dim=hidden_channels,
            expansion_rate=num_layers,
            sinkhorn_iters=sinkhorn_iters,
            use_dynamic_h=use_dynamic_h,
        )

    def forward(self, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple skip connection features using MHCLayer.

        Args:
            skip_features: List of n tensors, each [B, C, H, W]

        Returns:
            Fused features [B, C, H, W]
        """
        assert len(skip_features) == self.num_layers, \
            f"Expected {self.num_layers} features, got {len(skip_features)}"

        B, C, H, W = skip_features[0].shape
        device = skip_features[0].device

        # Stack: [B, n, C, H, W]
        stacked = torch.stack(skip_features, dim=1)

        # Reshape for MHCLayer: [B, n, C, H, W] -> [B*H*W, n, C]
        x = einops.rearrange(stacked, 'b n c h w -> (b h w) n c')

        # Apply MHCLayer: [B*H*W, n, C] -> [B*H*W, n, C]
        y = self.mhc(x)

        # Aggregate across n dimension based on strategy
        if self.aggregation == 'sum':
            y = y.sum(dim=1)  # [B*H*W, C]
        elif self.aggregation == 'mean':
            y = y.mean(dim=1)  # [B*H*W, C]
        elif self.aggregation == 'last':
            y = y[:, -1, :]  # [B*H*W, C]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Reshape back: [B*H*W, C] -> [B, C, H, W]
        out = einops.rearrange(y, '(b h w) c -> b c h w', b=B, h=H, w=W)

        return out


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


class DiagonalMaskedConv2D(nn.Conv2d):
    """
    Masked convolution for diagonal scan ordering.

    Causality: Position (i, j) can see position (i', j') if:
      - i' + j' < i + j (strictly previous diagonal), OR
      - i' + j' == i + j AND mask_type == "B" (same diagonal, Type B only)

    This enables 8x faster sampling (31 steps vs 256) by allowing parallel
    sampling within each anti-diagonal.
    """

    def __init__(
        self,
        *args,
        mask_type,
        data_channels,
        augment_channels=0,
        augment_output=True,
        **kwargs
    ):
        super(DiagonalMaskedConv2D, self).__init__(*args, **kwargs)
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

        # Diagonal masking: can see if diag_diff < 0 (or == 0 for Type B)
        for ky in range(height):
            for kx in range(width):
                rel_y = ky - y_center
                rel_x = kx - x_center
                diag_diff = rel_y + rel_x

                if diag_diff > 0:
                    mask[:, :, ky, kx] = 0
                elif diag_diff == 0 and mask_type == "A":
                    mask[:, :, ky, kx] = 0

        # Meta mask for channel ordering (same as raster)
        if mask_type == "A":
            meta_mask = torch.tril(
                torch.ones((data_channels, data_channels)), diagonal=-1
            )
        else:
            meta_mask = torch.tril(torch.ones((data_channels, data_channels)))

        # For conditional channels
        if augment_channels > 0:
            if augment_output:
                meta_mask = torch.cat(
                    [meta_mask, torch.zeros((augment_channels, data_channels))],
                    dim=0,
                )
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

        # Apply meta mask at center position
        mask[:, :, y_center, x_center] = meta_mask.repeat(out_tiles, in_tiles)

        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def hilbert_curve_order(n: int) -> List[Tuple[int, int]]:
    """
    Generate Hilbert curve indices for n x n grid.

    Args:
        n: Grid size (must be power of 2)

    Returns:
        List of (row, col) tuples in Hilbert curve order
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"

    def hilbert_d2xy(n: int, d: int) -> Tuple[int, int]:
        """Convert Hilbert curve index d to (x, y) coordinates."""
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (d // 2)
            ry = 1 & (d ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            d //= 4
            s *= 2
        return x, y

    order = []
    for d in range(n * n):
        x, y = hilbert_d2xy(n, d)
        order.append((y, x))  # Return as (row, col)
    return order


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
        dilation_pattern=None,
        skip_connection_indices=None,
        use_mhc_fusion=False,
        mhc_config=None,
        conv_class=None,
    ):
        """
        Masked Residual Convolutional Network for autoregressive modeling.

        Args:
            dilation_pattern: List of dilation values (e.g., [1,2,4,8]).
                              If None, uses default 2**(i % 4) pattern.
            skip_connection_indices: List of layer indices to collect skip features from.
                                     Features are fused and added to final conv output.
            use_mhc_fusion: If True, use MHCLayer for skip fusion instead of Conv2d.
                           Requires mHC.cu to be installed.
            mhc_config: Optional dict with MHC configuration:
                        - use_dynamic_h: bool (default: False)
                        - sinkhorn_iters: int (default: 10)
                        - aggregation: str 'sum'|'mean'|'last' (default: 'sum')
            conv_class: Masked convolution class to use (default: MaskedConv2D).
                       Can be DiagonalMaskedConv2D for diagonal scan ordering.
        """
        super().__init__()

        self.use_mhc_fusion = use_mhc_fusion

        self.channel = channel
        self.category = category
        self.hidden_channels = hidden_channels

        # Select convolution class
        ConvClass = conv_class if conv_class is not None else MaskedConv2D

        self.first_conv = ConvClass(
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
            # Dilation: use pattern if provided, else default exponential cycle
            if dilation_pattern is not None:
                dilation = dilation_pattern[i % len(dilation_pattern)]
            else:
                dilation = 2**(i % 4)  # Cycle dilations 1, 2, 4, 8

            hidden_convs.append(
                nn.Sequential(
                    ConvClass(
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
                    ConvClass(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=hidden_kernel_size,
                        padding=(hidden_kernel_size - 1) * dilation // 2, # Adjust padding for dilation
                        dilation=dilation,
                        mask_type="B",
                        data_channels=channel,
                        augment_channels=augment_channels,
                        augment_output=True,
                    ),
                    nn.GELU(),
                    ConvClass(
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

        self.first_fc = ConvClass(
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
                ConvClass(
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

        self.final_fc = ConvClass(
            in_channels=hidden_width,
            out_channels=category * channel,
            kernel_size=1,
            mask_type="B",
            data_channels=channel,
            augment_channels=augment_channels,
            augment_output=False,
        )

        # Skip connection fusion layer
        # Collects features from specified intermediate layers and fuses them
        if skip_connection_indices:
            self.skip_indices = set(skip_connection_indices)
            num_skips = len(skip_connection_indices)

            if use_mhc_fusion:
                # Use HC/mHC Layer for learned multi-scale fusion
                mhc_cfg = mhc_config or {}
                self.skip_fusion = HCFusion2D(
                    hidden_channels=2 * hidden_channels,
                    num_layers=num_skips,
                    use_dynamic_h=mhc_cfg.get('use_dynamic_h', False),
                    sinkhorn_iters=mhc_cfg.get('sinkhorn_iters', 10),
                    aggregation=mhc_cfg.get('aggregation', 'sum'),
                    manifold_constraint=mhc_cfg.get('manifold_constraint', True),
                )
            else:
                # Standard Conv2d fusion (concat + 1x1 conv)
                self.skip_fusion = nn.Sequential(
                    nn.Conv2d(2 * hidden_channels * num_skips, 2 * hidden_channels, 1),
                    nn.GELU(),
                )
        else:
            self.skip_indices = None
            self.skip_fusion = None

        # Initialize ALL biases to 0 for symmetric initialization
        # This ensures initial prediction is p=0.5 (unbiased) and prevents
        # systematic bias accumulation through the network
        self._init_zero_bias()

    def _init_zero_bias(self):
        """Initialize all biases to 0 for spin symmetry."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        size = x.shape
        x = self.first_conv(x)
        x = F.gelu(x)

        # Collect skip features from specified layers for multi-scale fusion
        skip_features = []

        for idx, layer in enumerate(self.hidden_convs):
            # Conv residual block
            tmp = layer(x)
            tmp = F.gelu(tmp)
            x = x + tmp

            # Collect skip connection if this layer is in skip_indices
            if self.skip_indices is not None and idx in self.skip_indices:
                skip_features.append(x)

        # Fuse multi-scale skip features and add to output
        if skip_features and self.skip_fusion is not None:
            if self.use_mhc_fusion:
                # HCFusion2D expects list of tensors
                fused = self.skip_fusion(skip_features)
            else:
                # Standard fusion expects concatenated tensor
                fused = self.skip_fusion(torch.cat(skip_features, dim=1))
            x = x + fused

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

        # Path type: "raster" (default), "diagonal" (8x faster), or "hilbert" (best locality)
        self.path_type = hparams.get("path_type", "raster")
        assert self.path_type in {"raster", "diagonal", "hilbert"}, \
            f"path_type must be 'raster', 'diagonal', or 'hilbert', got {self.path_type}"

        # Precompute path orderings
        H, W = self.size
        if self.path_type == "diagonal":
            # Diagonal groups: list of position lists per anti-diagonal
            # Total 2*N-1 diagonals for NxN grid
            self.diagonal_groups = []
            for d in range(H + W - 1):
                positions = [(i, d - i) for i in range(max(0, d - W + 1), min(H, d + 1))]
                self.diagonal_groups.append(positions)
        elif self.path_type == "hilbert":
            # Hilbert curve order: list of (row, col) in Hilbert order
            assert H == W and (H & (H - 1)) == 0, \
                f"Hilbert curve requires square grid with power-of-2 size, got {self.size}"
            self.hilbert_order = hilbert_curve_order(H)

        # Select convolution class based on path_type
        if self.path_type == "diagonal":
            conv_class = DiagonalMaskedConv2D
        else:
            # Raster and Hilbert use standard MaskedConv2D
            # (Hilbert only changes sampling order, not masking)
            conv_class = MaskedConv2D

        # Curriculum learning settings
        # High temp (low beta) → Low temp (high beta) progression
        self.curriculum_enabled = hparams.get("curriculum_enabled", False)
        self.curriculum_warmup_epochs = hparams.get("curriculum_warmup_epochs", 50)
        self.curriculum_start_beta_max = hparams.get(
            "curriculum_start_beta_max", self.beta_min * 1.5
        )

        # Curriculum Learning settings
        self.phase1_epochs = hparams.get("phase1_epochs", 50)
        self.phase1_beta_max = hparams.get("phase1_beta_max", 0.35)
        self.phase2_epochs = hparams.get("phase2_epochs", 100)

        # Temperature-dependent Output Scaling (Parameter-free: uses β = 1/T)
        # High temp: scale small → logits smaller → softmax closer to 0.5
        # Low temp: scale large → logits larger → softmax sharper
        self.logit_temp_scale = hparams.get("logit_temp_scale", False)
        self.temp_scale_min = hparams.get("temp_scale_min", 0.1)
        self.temp_scale_max = hparams.get("temp_scale_max", 2.0)

        # Initialize MaskedResConv2D with selected convolution class
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
            dilation_pattern=hparams.get("dilation_pattern", None),
            skip_connection_indices=hparams.get("skip_connection_indices", None),
            use_mhc_fusion=hparams.get("use_mhc_fusion", False),
            mhc_config=hparams.get("mhc_config", None),
            conv_class=conv_class,
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

    def _compute_temp_scale(self, T):
        """
        Compute temperature-dependent scaling factor for logits.

        Parameter-free scaling using inverse temperature (β = 1/T):
        - High temp: β small → logits smaller → softmax closer to 0.5 (disorder)
        - Low temp: β large → logits larger → softmax sharper (order)

        This is physically motivated: Boltzmann distribution P(x) ∝ exp(-βE)
        naturally uses β = 1/T as the scaling factor.

        Args:
            T: Temperature tensor of shape (B,) or (B, 1)

        Returns:
            scale: Scaling factor of shape (B, 1, 1, 1, 1) for broadcasting with logits (B, Cat, C, H, W)
        """
        if not self.logit_temp_scale:
            return 1.0

        # Ensure T is the right shape: (B,) → (B, 1)
        if T.dim() == 1:
            T = T.unsqueeze(1)

        # Parameter-free: use inverse temperature β = 1/T
        scale = 1.0 / T

        # Clamp to prevent extreme values
        scale = scale.clamp(min=self.temp_scale_min, max=self.temp_scale_max)

        # Reshape for broadcasting with (B, Cat, C, H, W): (B, 1) → (B, 1, 1, 1, 1)
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return scale

    def sample(self, batch_size=None, T=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        sample = torch.zeros(batch_size, self.channel, self.size[0], self.size[1]).to(
            self.device
        )

        # Compute temperature-dependent scaling
        temp_scale = 1.0
        if T is not None:
            T = T.to(self.device)
            temp_scale = self._compute_temp_scale(T)

            # (B, C) -> (B, C, H, W)
            if T.dim() == 1:
                T = T.unsqueeze(1)

            T_expanded = einops.repeat(
                T, "b c -> b c h w", h=self.size[0], w=self.size[1]
            )
            sample = torch.cat([sample, T_expanded], dim=1)

        # Dispatch to appropriate sampling method based on path_type
        if self.path_type == "diagonal":
            sample = self._sample_diagonal(sample, temp_scale, T)
        elif self.path_type == "hilbert":
            sample = self._sample_hilbert(sample, temp_scale, T)
        else:  # raster (default)
            sample = self._sample_raster(sample, temp_scale, T)

        if T is not None:
            sample = sample[:, : self.channel, :, :]

        sample = self.mapping(sample)  # Map {0,1} to {-1,1}

        return sample

    def _sample_raster(self, sample, temp_scale, T):
        """Sample using raster scan (standard left-to-right, top-to-bottom)."""
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                # Fix the first element of the samples to be a fixed value
                if self.fix_first is not None and i == 0 and j == 0:
                    if T is not None:
                        sample[:, : self.channel, 0, 0] = self.fix_first
                    else:
                        sample[:, :, 0, 0] = self.fix_first
                    continue

                # Compute predictions for all channels at once (B, Cat, C, H, W)
                unnormalized = self.masked_conv.forward(sample)

                # Apply temperature-dependent scaling
                if self.logit_temp_scale and T is not None:
                    unnormalized = unnormalized * temp_scale

                for k in range(self.channel):
                    sample[:, k, i, j] = (
                        torch.multinomial(
                            torch.softmax(unnormalized[:, :, k, i, j], dim=1),
                            1,
                        )
                        .squeeze()
                        .float()
                    )
        return sample

    def _sample_diagonal(self, sample, temp_scale, T):
        """
        Sample using diagonal scan (anti-diagonal ordering).

        This enables ~8x speedup (31 steps vs 256 for 16x16 grid) by sampling
        all positions in the same anti-diagonal in parallel within a single forward pass.
        """
        for d, positions in enumerate(self.diagonal_groups):
            # Handle fix_first at (0, 0) which is in diagonal 0
            if d == 0 and self.fix_first is not None:
                if T is not None:
                    sample[:, : self.channel, 0, 0] = self.fix_first
                else:
                    sample[:, :, 0, 0] = self.fix_first
                continue

            # Single forward pass for this diagonal
            unnormalized = self.masked_conv.forward(sample)

            if self.logit_temp_scale and T is not None:
                unnormalized = unnormalized * temp_scale

            # Sample all positions in this diagonal (can be parallelized on GPU)
            for (i, j) in positions:
                # Skip (0,0) if in later diagonal (shouldn't happen, but safety check)
                if i == 0 and j == 0 and self.fix_first is not None:
                    continue

                for k in range(self.channel):
                    sample[:, k, i, j] = (
                        torch.multinomial(
                            torch.softmax(unnormalized[:, :, k, i, j], dim=1),
                            1,
                        )
                        .squeeze()
                        .float()
                    )
        return sample

    def _sample_hilbert(self, sample, temp_scale, T):
        """
        Sample using Hilbert curve ordering.

        Preserves 2D locality - adjacent positions in the sequence are neighbors
        in 2D space. Still 256 steps but may improve learning of local correlations.

        Note: Uses raster masking, so the network has raster-biased receptive field.
        Only the sampling ORDER follows Hilbert curve.
        """
        first_pos = self.hilbert_order[0]

        for idx, (i, j) in enumerate(self.hilbert_order):
            # Fix the first element (which is hilbert_order[0])
            if idx == 0 and self.fix_first is not None:
                if T is not None:
                    sample[:, : self.channel, i, j] = self.fix_first
                else:
                    sample[:, :, i, j] = self.fix_first
                continue

            # Compute predictions for all channels at once (B, Cat, C, H, W)
            unnormalized = self.masked_conv.forward(sample)

            if self.logit_temp_scale and T is not None:
                unnormalized = unnormalized * temp_scale

            for k in range(self.channel):
                sample[:, k, i, j] = (
                    torch.multinomial(
                        torch.softmax(unnormalized[:, :, k, i, j], dim=1),
                        1,
                    )
                    .squeeze()
                    .float()
                )
        return sample

    def log_prob(self, sample, T=None):
        # sample to {0,1}
        sample = self.reverse_mapping(sample)

        if self.fix_first is not None:
            assert (
                sample[:, :, 0, 0] == self.fix_first
            ).all(), "The first element of the sample does not match fix_first value."

        # Compute temperature-dependent scaling
        temp_scale = 1.0
        if T is not None:
            T = T.to(self.device)
            temp_scale = self._compute_temp_scale(T)

            # (B, C) -> (B, C, H, W)
            if T.dim() == 1:
                T = T.unsqueeze(1)

            T_expanded = einops.repeat(
                T, "b c -> b c h w", h=sample.shape[2], w=sample.shape[3]
            )
            sample = torch.cat([sample, T_expanded], dim=1)

        unnormalized = self.masked_conv.forward(sample)  # (B, Cat, C, H, W)

        # Apply temperature-dependent scaling
        # High temp: scale < 1 → logits smaller → softmax closer to 0.5
        if self.logit_temp_scale and T is not None:
            unnormalized = unnormalized * temp_scale

        # Use log_softmax for numerical stability (avoids log(softmax()) underflow)
        log_prob = F.log_softmax(unnormalized, dim=1)

        if T is not None:
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
