import torch
from torch import nn
import torch.nn.functional as F
import einops


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


# ============================================================================
# RealNVP (Normalizing Flow) Implementation
# ============================================================================


class ResidualBlock(nn.Module):
    """
    Simple Residual Block used inside the Scale and Translation networks.
    """

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.net(x)


class STNetwork(nn.Module):
    """
    Scale (s) and Translation (t) Network.
    It predicts the affine transformation parameters based on the input.
    """

    def __init__(self, in_channels, hidden_channels, num_layers, augment_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.augment_channels = augment_channels

        # Input layer: takes split data + conditional channels (if any)
        self.in_conv = nn.Conv2d(
            in_channels + augment_channels, hidden_channels, kernel_size=3, padding=1
        )

        # Hidden layers (Residual blocks)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels) for _ in range(num_layers)]
        )

        # Output layer: predicts both scale (s) and translation (t)
        # We output 2 * in_channels (first half for s, second half for t)
        self.out_conv = nn.Conv2d(
            hidden_channels, 2 * in_channels, kernel_size=3, padding=1
        )

        # Initialize output layer with zeros for identity mapping at the start
        self.out_conv.weight.data.zero_()
        self.out_conv.bias.data.zero_()

    def forward(self, x, T_cond=None):
        """
        x: Input feature map
        T_cond: Conditional input (e.g., Temperature), shape (B, C, H, W)
        """
        if self.augment_channels > 0 and T_cond is not None:
            # Concatenate conditional information along channel dimension
            x = torch.cat([x, T_cond], dim=1)

        x = F.gelu(self.in_conv(x))
        for block in self.res_blocks:
            x = block(x)

        st = self.out_conv(x)
        s, t = st.chunk(2, dim=1)

        # Apply tanh to scale factor to prevent numerical instability
        s = torch.tanh(s)
        return s, t


class AffineCouplingLayer(nn.Module):
    """
    Affine Coupling Layer.
    Splits the input into two parts (A and B).
    A is kept constant, B is transformed based on A.
    """

    def __init__(self, in_channels, hidden_channels, num_layers, augment_channels=0):
        super().__init__()
        # We process half of the channels
        self.split_len = in_channels // 2

        # The network that predicts s and t from x_a
        self.st_net = STNetwork(
            in_channels=self.split_len,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            augment_channels=augment_channels,
        )

    def forward(self, x, log_det, T_cond=None, reverse=False):
        # Split input into x_a (frozen) and x_b (transformed)
        x_a = x[:, : self.split_len, :, :]
        x_b = x[:, self.split_len :, :, :]

        # Get scale and translation parameters from x_a
        s, t = self.st_net(x_a, T_cond)

        if not reverse:
            # Forward: z = f(x) -> Used for likelihood calculation (training)
            # y_b = x_b * exp(s) + t
            y_b = x_b * torch.exp(s) + t
            y_a = x_a

            # Update log-determinant of Jacobian
            # log_det += sum(s)
            log_det = log_det + torch.sum(s, dim=[1, 2, 3])
        else:
            # Inverse: x = f^-1(z) -> Used for sampling (generation)
            # x_b = (y_b - t) * exp(-s)
            y_b = (x_b - t) * torch.exp(-s)
            y_a = x_a

            # We calculate the log determinant of the FORWARD transformation (dz/dx)
            # even during the inverse pass, because we need it for log_prob calculation
            # in sample_and_log_prob.
            log_det = log_det + torch.sum(s, dim=[1, 2, 3])

        return torch.cat([y_a, y_b], dim=1), log_det


class RealNVP(nn.Module):
    def __init__(self, hparams, device="cpu"):
        super().__init__()
        self.hparams = hparams
        self.device = device
        self.differentiable_sampling = (
            True  # allow trainer to keep gradients during sampling
        )

        # --- Compatibility with PixelCNN setup ---
        self.channel = 1
        self.size = hparams.get("size", 16)
        if isinstance(self.size, int):
            self.size = (self.size, self.size)

        self.batch_size = hparams.get("batch_size", 64)

        # FIXED: Add required attributes for Trainer compatibility
        self.beta_min = hparams.get("beta_min", 0.2)
        self.beta_max = hparams.get("beta_max", 1.0)
        self.num_beta = hparams.get("num_beta", 8)

        # Flow Hyperparameters
        hidden_channels = hparams.get("hidden_channels", 64)
        num_blocks = hparams.get(
            "hidden_conv_layers", 4
        )  # Reuse this param for ResBlocks
        num_flow_layers = hparams.get("num_flow_layers", 6)  # Number of coupling layers
        self.augment_channels = 1  # For Temperature (T)

        # Squeeze factor: 1x16x16 -> 4x8x8
        self.squeeze_factor = 2
        self.in_channels = self.channel * (self.squeeze_factor**2)  # 1 * 4 = 4 channels

        # FIXED: Create permutation indices for channel mixing
        # For 4 channels: [0,1,2,3] -> [1,0,3,2]
        self.register_buffer(
            "perm_indices",
            torch.tensor(
                [1, 0, 3, 2] if self.in_channels == 4 else list(range(self.in_channels))
            ),
        )
        self.register_buffer(
            "inv_perm_indices",
            torch.argsort(self.perm_indices),
        )

        # Construct Flow Layers
        self.layers = nn.ModuleList()
        for i in range(num_flow_layers):
            self.layers.append(
                AffineCouplingLayer(
                    in_channels=self.in_channels,
                    hidden_channels=hidden_channels,
                    num_layers=num_blocks,
                    augment_channels=self.augment_channels,
                )
            )

        # Base distribution: Gaussian N(0, 1)
        self.prior = torch.distributions.Normal(0, 1)

    def to(self, *args, **kwargs):
        """Override to() method to update self.device when model is moved to a different device."""
        self = super().to(*args, **kwargs)
        # Extract device from args/kwargs
        if args and isinstance(args[0], (torch.device, str)):
            self.device = args[0]
        elif 'device' in kwargs:
            self.device = kwargs['device']
        return self

    def squeeze(self, x):
        """Reshape (B, C, H, W) -> (B, 4*C, H/2, W/2)"""
        b, c, h, w = x.size()
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 4, h // 2, w // 2)
        return x

    def undo_squeeze(self, x):
        """Reshape (B, 4*C, H/2, W/2) -> (B, C, H, W)"""
        b, c, h, w = x.size()
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
        return x

    def _prepare_conditional(self, T, h, w):
        """Resizes Temperature T to match the current feature map spatial size."""
        if T is None:
            return None

        T = T.to(self.device)
        if T.dim() == 1:
            T = T.unsqueeze(1)  # (B, 1)

        # Repeat T to match spatial dimensions (B, 1, H, W)
        T_map = T.view(T.shape[0], T.shape[1], 1, 1).repeat(1, 1, h, w)
        return T_map

    def forward(self, x, T=None):
        """
        Goes from Data (x) -> Latent (z). Used for training (Likelihood).

        NOTE: RealNVP generates continuous distributions directly, so no dequantization needed.
        The input x should be continuous values from the model's own sample() method.
        """
        # Inverse of the final tanh in sample()
        # x is in (-1, 1). We need to apply atanh to map it back to (-inf, inf)
        # and account for the Jacobian determinant change.

        # Stability: clip x to (-1 + eps, 1 - eps)
        eps = 1e-6
        x = torch.clamp(x, -1 + eps, 1 - eps)

        # Log-determinant adjustment for atanh
        # log |det dy/dx| = - sum(log(1 - x^2))
        delta_log_det = -torch.sum(torch.log(1 - x**2), dim=[1, 2, 3])

        x = torch.atanh(x)

        # Squeeze spatial dims to channels
        x = self.squeeze(x)

        log_det_sum = delta_log_det
        b, c, h, w = x.shape
        T_cond = self._prepare_conditional(T, h, w)

        for i, layer in enumerate(self.layers):
            x, log_det_sum = layer(x, log_det_sum, T_cond=T_cond, reverse=False)
            # FIXED: Use learnable permutation instead of flip
            x = x[:, self.perm_indices, :, :]

        z = x
        return z, log_det_sum

    def log_prob(self, x, T=None):
        """
        Calculates log p(x).
        Compatible with the existing training loop.

        FIXED: Returns (B, 1) instead of scalar for compatibility with util.py
        """
        z, log_det = self.forward(x, T)

        # log p(z) = -0.5 * z^2 - 0.5 * log(2pi)
        log_prob_z = torch.sum(
            -0.5 * (z**2) - 0.5 * torch.tensor(2 * torch.pi).log(),
            dim=[1, 2, 3],
        )

        # log p(x) = log p(z) + log |det(J)|
        log_prob_x = log_prob_z + log_det

        # FIXED: Return (B, 1) tensor instead of scalar
        return log_prob_x.unsqueeze(-1)

    def sample_and_log_prob(self, batch_size=None, T=None):
        """
        Efficiently computes both sample and log_prob.
        Avoids the numerical instability of atanh(tanh(x)) at boundaries.
        """
        bs = batch_size if batch_size is not None else self.batch_size

        # Latent shape must match the squeezed shape
        h_s, w_s = self.size[0] // 2, self.size[1] // 2
        c_s = self.channel * 4

        # Sample z from Gaussian Prior
        z_0 = self.prior.rsample((bs, c_s, h_s, w_s)).to(self.device)

        # FIXED: Compute log p(z_0) immediately before z is overwritten
        log_prob_z = torch.sum(
            -0.5 * (z_0**2) - 0.5 * torch.tensor(2 * torch.pi).log(),
            dim=[1, 2, 3],
        )

        T_cond = self._prepare_conditional(T, h_s, w_s)

        # Inverse flow: z -> x_pre (unbounded)
        # We need to track log_det during inverse flow as well
        # Note: The existing layers return log_det for forward pass.
        # For inverse pass, log_det_inverse = -log_det_forward.
        # But we need to compute it.

        # Our AffineCouplingLayer calculates log_det += sum(s) in forward.
        # In inverse: y_b = (x_b - t) * exp(-s).
        # Jacobian is exp(-s). Log det is -sum(s).
        # So we can just accumulate sum(s) and subtract it.

        log_det_flow = 0

        # Inverse flow: iterate backwards
        z = z_0
        for layer in reversed(self.layers):
            z = z[:, self.inv_perm_indices, :, :]
            # We need to modify layer to return log_det in reverse mode?
            # Or just access s?
            # The current implementation of AffineCouplingLayer.forward:
            # if reverse: return ... (doesn't return log_det)

            # We need to fix AffineCouplingLayer to return log_det in reverse too
            # Or manually compute it here.
            # But 's' is internal to layer.

            # Let's assume I update AffineCouplingLayer to return log_det in reverse mode.
            z, log_det_layer = layer(z, 0, T_cond=T_cond, reverse=True)
            log_det_flow += log_det_layer

        # Undo squeeze to get original image size
        x_pre = self.undo_squeeze(z)

        # log_prob calculation
        # log q(x) = log p(z) - log |det J_flow| - log |det J_tanh|
        # J_flow: dz/dx_pre.
        # We computed z -> x_pre. The accumulated log_det_flow is log |det dx_pre/dz|.
        # So log |det dz/dx_pre| = -log_det_flow.

        # Wait, let's verify sign.
        # Forward (x->z): log_det += sum(s). log q(x) = log p(z) + sum(s).
        # Inverse (z->x): x = z * e^{-s}. dx/dz = e^{-s}. log det = -s.
        # log q(x) = log p(z) - log |dx/dz| = log p(z) - (-s) = log p(z) + s.
        # So we need the sum(s).
        # If 'log_det_flow' accumulates sum(s) (magnitude of scaling),
        # then log_prob = log p(z) + log_det_flow.

        # log |det J_tanh|
        # x = tanh(x_pre). dx/dx_pre = 1 - x^2 = sech^2(x_pre).
        # log |dx/dx_pre| = log(sech^2 x_pre) = -2 log cosh x_pre.
        # We need to subtract this from log density of x_pre.
        # log q(x) = log q(x_pre) - log |dx/dx_pre|
        #          = (log p(z) + log_det_flow) - (-2 log cosh x_pre)
        #          = log p(z) + log_det_flow + 2 log cosh x_pre.

        # FIXED: Stable implementation of 2 log cosh(x) = 2 * (|x| + softplus(-2|x|) - log(2))
        log_det_tanh = 2 * (
            torch.abs(x_pre) - torch.log(torch.tensor(2.0)) + F.softplus(-2 * torch.abs(x_pre))
        )
        log_det_tanh = log_det_tanh.sum(dim=[1, 2, 3])

        # Note: Previous reasoning:
        # log q(x) = log p(x_pre) - log |det tanh'|
        # log |det tanh'| = log(sech^2) = -2 log cosh.
        # - log |det tanh'| = + 2 log cosh.
        # So we ADD 2 log cosh.

        # Correction on sign:
        # log_det_tanh variable above computes 2 log cosh.
        # We want to compute: log p(x) = log p(x_pre) + log |dx_pre/dx|
        # dx_pre/dx = 1/sech^2 = cosh^2
        # log |dx_pre/dx| = 2 log cosh = log_det_tanh
        # So we ADD log_det_tanh.

        # FIXED: Changed from - to + (correcting previous error)
        log_prob_x = log_prob_z + log_det_flow + log_det_tanh

        x = torch.tanh(x_pre)

        return log_prob_x.unsqueeze(-1), x

    def sample(self, batch_size=None, T=None):
        """
        Goes from Latent (z) -> Data (x).
        Used for Generation. O(1) complexity.
        Used for Generation. O(1) complexity.

        Returns continuous values near {-1, 1} using tanh activation.
        For VaTD's temperature-differentiable objective, gradients flow through sampling.

        For energy calculation with discrete Ising spins, use straight-through:
            samples_continuous = model.sample(T=T)
            samples_discrete = torch.sign(samples_continuous).detach()
            energy = energy_fn(samples_discrete + samples_continuous - samples_continuous.detach())
        """
        bs = batch_size if batch_size is not None else self.batch_size

        # Latent shape must match the squeezed shape
        h_s, w_s = self.size[0] // 2, self.size[1] // 2
        c_s = self.channel * 4

        # Sample z from Gaussian Prior
        # rsample keeps the reparameterization path for lower-variance gradients
        z = self.prior.rsample((bs, c_s, h_s, w_s)).to(self.device)

        T_cond = self._prepare_conditional(T, h_s, w_s)

        # Inverse flow: iterate backwards
        for layer in reversed(self.layers):
            # FIXED: Undo permutation first (because we did it after layer in forward)
            z = z[:, self.inv_perm_indices, :, :]
            z, _ = layer(z, 0, T_cond=T_cond, reverse=True)

        # Undo squeeze to get original image size
        x = self.undo_squeeze(z)

        # Return continuous soft output with gradients
        # tanh keeps values in (-1, 1) range suitable for Ising spins
        return torch.tanh(x)
