"""
Checkerboard PixelCNN for 2D Ising Model.

This module implements a checkerboard-factorized autoregressive model that
exploits the nearest-neighbor structure of the Ising model for parallel sampling.

Key insight: In a checkerboard pattern, black cells only neighbor white cells
and vice versa. This means:
- All black cells are conditionally independent given white cells
- All white cells are conditionally independent given black cells

This reduces sampling from O(N) sequential steps to just 2K parallel steps!
(where K = num_iterations, typically 1-4)

Iterative Refinement (Gibbs-style):
    For k = 1 to K:
        x_black^{(k)} ~ p(x_black | x_white^{(k-1)})
        x_white^{(k)} ~ p(x_white | x_black^{(k)})

    This allows black cells to benefit from white cell information after the first iteration,
    significantly improving sample quality and convergence speed.
"""

import torch
from torch import nn
import torch.nn.functional as F
import einops
import numpy as np
from typing import Tuple, Optional


class ResBlock(nn.Module):
    """
    Residual block without causal masking.

    For checkerboard models, we don't need autoregressive masking because
    we only condition on the opposite color cells, which are always available.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels, hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CheckerboardNet(nn.Module):
    """
    Neural network backbone for checkerboard generation.

    Takes as input:
    - Current lattice state (with mask indicating which cells to predict)
    - Temperature conditioning
    - Phase indicator (which color we're predicting)

    Outputs logits for the masked (to-be-predicted) cells.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_blocks: int = 8,
        kernel_size: int = 3,
        hidden_width: int = 128,
        num_fc_layers: int = 2,
        dilation_pattern: Optional[list] = None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels

        # Input: spin channel + temperature channel + phase channel
        # Spin: 1 channel (values in {-1, 0, 1} where 0 = masked/unknown)
        # Temperature: 1 channel
        # Phase: 1 channel (0 = predicting black, 1 = predicting white)
        in_channels = 3

        self.first_conv = nn.Conv2d(
            in_channels, hidden_channels,
            kernel_size=7, padding=3,
        )

        # Residual blocks with dilations for large receptive field
        if dilation_pattern is None:
            dilation_pattern = [1, 2, 4, 8]

        blocks = []
        for i in range(num_blocks):
            dilation = dilation_pattern[i % len(dilation_pattern)]
            blocks.append(ResBlock(
                hidden_channels, hidden_channels // 2,
                kernel_size=kernel_size, dilation=dilation,
            ))
        self.blocks = nn.ModuleList(blocks)

        # FC layers (1x1 convs)
        self.first_fc = nn.Conv2d(hidden_channels, hidden_width, kernel_size=1)

        fc_layers = []
        for _ in range(num_fc_layers):
            fc_layers.append(nn.Conv2d(hidden_width, hidden_width, kernel_size=1))
        self.fc_layers = nn.ModuleList(fc_layers)

        # Output: 2 logits per position (spin up/down)
        self.final_fc = nn.Conv2d(hidden_width, 2, kernel_size=1)

        # Initialize biases to zero for symmetry
        self._init_zero_bias()

    def _init_zero_bias(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        T: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Spin configuration (B, 1, H, W), values in {-1, 0, 1}
               where 0 indicates masked/unknown cells
            T: Temperature (B, 1) or (B, 1, H, W)
            phase: Phase indicator (B, 1, H, W), 0=predicting black, 1=predicting white

        Returns:
            Logits (B, 2, H, W) for spin down/up at each position
        """
        B, _, H, W = x.shape

        # Expand temperature if needed
        if T.dim() == 2:
            T = T.unsqueeze(-1).unsqueeze(-1).expand(B, 1, H, W)

        # Concatenate inputs
        inp = torch.cat([x, T, phase], dim=1)  # (B, 3, H, W)

        # Forward through network
        h = self.first_conv(inp)
        h = F.gelu(h)

        for block in self.blocks:
            h = block(h)

        h = self.first_fc(h)
        h = F.gelu(h)

        for fc in self.fc_layers:
            h = fc(h)
            h = F.gelu(h)

        logits = self.final_fc(h)  # (B, 2, H, W)

        return logits


class CheckerboardPixelCNN(nn.Module):
    """
    Checkerboard-factorized PixelCNN for Ising model.

    Exploits the bipartite structure of the square lattice:
    - Black cells: (i + j) % 2 == 0
    - White cells: (i + j) % 2 == 1

    Sampling in 2 steps instead of H*W steps!

    Factorization:
        log p(x) = log p(x_black | x_white_init) + log p(x_white | x_black)
    """

    def __init__(self, hparams, device="cpu"):
        super().__init__()

        self.hparams = hparams
        self.device = device

        # Lattice configuration
        size = hparams.get("size", 16)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

        self.batch_size = hparams["batch_size"]
        self.num_beta = hparams["num_beta"]
        self.beta_min = hparams["beta_min"]
        self.beta_max = hparams["beta_max"]

        # Fix first spin: For checkerboard, we use post-hoc Z2 correction
        # If fix_first is set, after sampling we flip all spins if x[0,0] != fix_first
        # This exploits global Z2 symmetry: p(x) = p(-x)
        self.fix_first = hparams.get("fix_first", None)

        # Iterative refinement: Gibbs-style iterations for better sample quality
        # num_iterations=1 is the original 2-step checkerboard
        # num_iterations>1 allows black cells to see white cell information
        self.num_iterations = hparams.get("num_iterations", 1)

        # Spin mapping
        self.mapping = lambda x: 2 * x - 1  # {0,1} -> {-1,1}
        self.reverse_mapping = lambda x: torch.div(x + 1, 2, rounding_mode="trunc")

        # Network
        self.net = CheckerboardNet(
            hidden_channels=hparams.get("hidden_channels", 64),
            num_blocks=hparams.get("hidden_conv_layers", 8),
            kernel_size=hparams.get("hidden_kernel_size", 3),
            hidden_width=hparams.get("hidden_width", 128),
            num_fc_layers=hparams.get("hidden_fc_layers", 2),
            dilation_pattern=hparams.get("dilation_pattern", [1, 2, 4, 8]),
        )

        # Create checkerboard masks (precomputed)
        self._create_masks()

        # Curriculum learning settings (inherited from PixelCNN)
        self.curriculum_enabled = hparams.get("curriculum_enabled", False)
        self.phase1_epochs = hparams.get("phase1_epochs", 50)
        self.phase1_beta_max = hparams.get("phase1_beta_max", 0.35)
        self.phase2_epochs = hparams.get("phase2_epochs", 100)

    def _create_masks(self):
        """Create checkerboard masks for black and white cells."""
        H, W = self.size

        # Create coordinate grids
        i_coords = torch.arange(H).unsqueeze(1).expand(H, W)
        j_coords = torch.arange(W).unsqueeze(0).expand(H, W)

        # Black: (i+j) % 2 == 0, White: (i+j) % 2 == 1
        black_mask = ((i_coords + j_coords) % 2 == 0).float()
        white_mask = ((i_coords + j_coords) % 2 == 1).float()

        # Register as buffers (not parameters, but move with model)
        self.register_buffer("black_mask", black_mask.unsqueeze(0).unsqueeze(0))  # (1, 1, H, W)
        self.register_buffer("white_mask", white_mask.unsqueeze(0).unsqueeze(0))  # (1, 1, H, W)

    def to(self, *args, **kwargs):
        """Override to() to update self.device."""
        self = super().to(*args, **kwargs)
        if args and isinstance(args[0], (torch.device, str)):
            self.device = args[0]
        elif "device" in kwargs:
            self.device = kwargs["device"]
        return self

    def sample(self, batch_size=None, T=None) -> torch.Tensor:
        """
        Generate samples using iterative checkerboard refinement (Gibbs-style).

        For k = 1 to num_iterations:
            x_black^{(k)} ~ p(x_black | x_white^{(k-1)})
            x_white^{(k)} ~ p(x_white | x_black^{(k)})

        Args:
            batch_size: Number of samples to generate
            T: Temperature tensor (B,) or (B, 1)

        Returns:
            samples: (B, 1, H, W) tensor with values in {-1, +1}
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        H, W = self.size

        # Initialize lattice with zeros (masked state)
        x = torch.zeros(batch_size, 1, H, W, device=self.device)

        # Temperature handling
        if T is not None:
            T = T.to(self.device)
            if T.dim() == 1:
                T = T.unsqueeze(1)  # (B, 1)
        else:
            T = torch.ones(batch_size, 1, device=self.device) * 2.27

        # Expand masks for batch
        black_mask = self.black_mask.expand(batch_size, 1, H, W)
        white_mask = self.white_mask.expand(batch_size, 1, H, W)

        # Phase tensors (reused across iterations)
        phase_black = torch.zeros(batch_size, 1, H, W, device=self.device)
        phase_white = torch.ones(batch_size, 1, H, W, device=self.device)

        # ===== Iterative Refinement =====
        for iteration in range(self.num_iterations):
            # ----- Resample BLACK cells given WHITE cells -----
            # On first iteration, white cells are 0 (unknown)
            # On subsequent iterations, white cells contain actual values
            x_context_black = x * white_mask  # Keep white, zero out black

            logits_black = self.net(x_context_black, T, phase_black)
            probs_black = F.softmax(logits_black, dim=1)
            samples_black = torch.bernoulli(probs_black[:, 1:2])
            samples_black = self.mapping(samples_black)

            # Update black cells
            x = samples_black * black_mask + x * white_mask

            # ----- Resample WHITE cells given BLACK cells -----
            x_context_white = x * black_mask  # Keep black, zero out white

            logits_white = self.net(x_context_white, T, phase_white)
            probs_white = F.softmax(logits_white, dim=1)
            samples_white = torch.bernoulli(probs_white[:, 1:2])
            samples_white = self.mapping(samples_white)

            # Update white cells
            x = x * black_mask + samples_white * white_mask

        # ===== Fix first spin using Z2 symmetry =====
        if self.fix_first is not None:
            needs_flip = (x[:, 0, 0, 0] != self.fix_first)
            flip_mask = needs_flip.float().view(-1, 1, 1, 1)
            x = x * (1 - 2 * flip_mask)

        return x

    def log_prob(self, sample: torch.Tensor, T=None) -> torch.Tensor:
        """
        Compute log pseudo-probability using checkerboard factorization.

        For iterative refinement (num_iterations > 1), we use pseudo-likelihood:
            log q(x) = log p(x_black | x_white) + log p(x_white | x_black)

        This conditions each color on the actual values of the other color,
        which is consistent with what the Gibbs sampler produces at equilibrium.

        For num_iterations=1 (original 2-step), we use:
            log q(x) = log p(x_black | 0) + log p(x_white | x_black)

        Args:
            sample: (B, 1, H, W) tensor with values in {-1, +1}
            T: Temperature tensor (B,) or (B, 1)

        Returns:
            log_prob: (B, 1) tensor of log probabilities
        """
        B, _, H, W = sample.shape

        # Temperature handling
        if T is not None:
            T = T.to(self.device)
            if T.dim() == 1:
                T = T.unsqueeze(1)
        else:
            T = torch.ones(B, 1, device=self.device) * 2.27

        # Convert sample to {0, 1} for indexing
        sample_01 = self.reverse_mapping(sample)  # {-1,+1} -> {0,1}

        # Expand masks
        black_mask = self.black_mask.expand(B, 1, H, W)
        white_mask = self.white_mask.expand(B, 1, H, W)

        # Phase tensors
        phase_black = torch.zeros(B, 1, H, W, device=self.device)
        phase_white = torch.ones(B, 1, H, W, device=self.device)

        # ===== log p(x_black | x_white) =====
        # For num_iterations > 1: condition on actual white values (pseudo-likelihood)
        # For num_iterations = 1: condition on zeros (original 2-step)
        if self.num_iterations > 1:
            x_context_black = sample * white_mask  # Actual white values, black zeroed
        else:
            x_context_black = torch.zeros_like(sample)  # Original: all zeros

        logits_black = self.net(x_context_black, T, phase_black)
        log_probs_black = F.log_softmax(logits_black, dim=1)
        log_prob_black_selected = log_probs_black.gather(1, sample_01.long())
        log_prob_black_sum = (log_prob_black_selected * black_mask).sum(dim=(1, 2, 3))

        # ===== log p(x_white | x_black) =====
        # Always condition on actual black values
        x_context_white = sample * black_mask  # Actual black values, white zeroed

        logits_white = self.net(x_context_white, T, phase_white)
        log_probs_white = F.log_softmax(logits_white, dim=1)
        log_prob_white_selected = log_probs_white.gather(1, sample_01.long())
        log_prob_white_sum = (log_prob_white_selected * white_mask).sum(dim=(1, 2, 3))

        # Total log probability
        log_prob_total = log_prob_black_sum + log_prob_white_sum

        # Handle fix_first: verify constraint is satisfied
        if self.fix_first is not None:
            assert (sample[:, 0, 0, 0] == self.fix_first).all(), \
                f"fix_first={self.fix_first} but sample[0,0] has different values"

        return log_prob_total.unsqueeze(1)  # (B, 1)


if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Test different num_iterations
    for num_iter in [1, 2, 4]:
        print(f"===== num_iterations = {num_iter} =====")
        hparams = {
            "size": 16,
            "batch_size": 32,
            "num_beta": 8,
            "beta_min": 0.1,
            "beta_max": 0.6,
            "hidden_channels": 32,
            "hidden_conv_layers": 4,
            "fix_first": 1,
            "num_iterations": num_iter,
        }

        model = CheckerboardPixelCNN(hparams, device=device).to(device)
        model.eval()

        T = torch.ones(32, device=device) * 2.27

        # Warmup
        with torch.no_grad():
            _ = model.sample(batch_size=32, T=T)

        # Benchmark
        n_runs = 10
        with torch.no_grad():
            start = time.time()
            for _ in range(n_runs):
                samples = model.sample(batch_size=32, T=T)
                if device == "cuda":
                    torch.cuda.synchronize()
            elapsed = (time.time() - start) / n_runs

        print(f"  Sample time: {elapsed*1000:.1f}ms ({2*num_iter} forward passes)")
        print(f"  fix_first check: all x[0,0]={samples[:, 0, 0, 0][0].item():.0f} ✓")

        # Test log_prob
        log_prob = model.log_prob(samples, T=T)
        print(f"  Log prob mean: {log_prob.mean().item():.2f}")
        print()

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
