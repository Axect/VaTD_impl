import torch
from torch import nn
import torch.nn.functional as F
import einops
import numpy as np
from typing import Tuple
import math


class CausalSelfAttention2D(nn.Module):
    """
    Causal Self-Attention for 2D grids (raster scan order).

    For autoregressive models like PixelCNN, position (i,j) can only attend to
    positions that come before it in raster scan order:
    - All positions in rows < i
    - Positions (i, j') where j' < j
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Cache for causal mask (will be created on first forward)
        self._causal_mask = None
        self._mask_size = None

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for raster scan order.
        Position i can attend to positions j where j <= i (type-B behavior).
        This allows each position to see itself and all previous positions.
        """
        if self._causal_mask is not None and self._mask_size == seq_len:
            return self._causal_mask.to(device)

        # Lower triangular mask including diagonal (type-B behavior)
        # Each position can attend to itself and all previous positions
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0)
        # Convert to attention mask format (0 = attend, -inf = mask)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)

        self._causal_mask = mask
        self._mask_size = seq_len
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) tensor
        Returns:
            (B, C, H, W) tensor
        """
        B, C, H, W = x.shape
        seq_len = H * W

        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        x_seq = einops.rearrange(x, 'b c h w -> b (h w) c')

        # QKV projection
        qkv = self.qkv(x_seq)  # (B, seq_len, 3 * embed_dim)
        qkv = einops.rearrange(qkv, 'b n (three h d) -> three b h n d',
                               three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, seq_len, head_dim)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, seq_len, seq_len)

        # Apply causal mask
        causal_mask = self._get_causal_mask(seq_len, x.device)
        attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, num_heads, seq_len, head_dim)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')

        # Output projection
        out = self.proj(out)  # (B, seq_len, embed_dim)

        # Reshape back to 2D: (B, H*W, C) -> (B, C, H, W)
        out = einops.rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)

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
        # Attention parameters
        use_attention=False,
        attention_heads=4,
        attention_every_n_layers=2,
        attention_dropout=0.0,
    ):

        super().__init__()

        self.channel = channel
        self.category = category
        self.use_attention = use_attention
        self.attention_every_n_layers = attention_every_n_layers

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
        attention_layers = []
        attention_norms = []

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

            # Add attention after every N conv layers
            if use_attention and (i + 1) % attention_every_n_layers == 0:
                attention_layers.append(
                    CausalSelfAttention2D(
                        embed_dim=2 * hidden_channels,
                        num_heads=attention_heads,
                        dropout=attention_dropout,
                    )
                )
                # LayerNorm for attention (applied channel-wise)
                attention_norms.append(nn.LayerNorm(2 * hidden_channels))
            else:
                attention_layers.append(None)
                attention_norms.append(None)

        self.hidden_convs = nn.ModuleList(hidden_convs)
        self.attention_layers = nn.ModuleList(
            [layer for layer in attention_layers if layer is not None]
        )
        self.attention_norms = nn.ModuleList(
            [norm for norm in attention_norms if norm is not None]
        )
        # Track which conv layers have attention
        self.attention_indices = [
            i for i, layer in enumerate(attention_layers) if layer is not None
        ]

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

        attn_idx = 0
        for i, layer in enumerate(self.hidden_convs):
            # Conv residual block
            tmp = layer(x)
            tmp = F.gelu(tmp)
            x = x + tmp

            # Apply attention if this layer has one
            if self.use_attention and i in self.attention_indices:
                # Pre-norm attention with residual
                B, C, H, W = x.shape
                x_norm = einops.rearrange(x, 'b c h w -> b (h w) c')
                x_norm = self.attention_norms[attn_idx](x_norm)
                x_norm = einops.rearrange(x_norm, 'b (h w) c -> b c h w', h=H, w=W)
                x = x + self.attention_layers[attn_idx](x_norm)
                attn_idx += 1

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
            # Attention parameters
            use_attention=hparams.get("use_attention", False),
            attention_heads=hparams.get("attention_heads", 4),
            attention_every_n_layers=hparams.get("attention_every_n_layers", 2),
            attention_dropout=hparams.get("attention_dropout", 0.0),
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
