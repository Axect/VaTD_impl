"""
Muon optimizer wrapper for VaTD implementation.

Muon (MomentUm Orthogonalized by Newton-Schulz) is designed for 2D weight matrices
in hidden layers. For 1D parameters (biases, gains) and embeddings, AdamW is used.

References:
- Keller Jordan: https://kellerjordan.github.io/posts/muon/
- GitHub: https://github.com/KellerJordan/Muon
- PyTorch 2.9+: torch.optim.Muon (native implementation)
"""

import torch
from torch import Tensor
from torch.optim import Optimizer, AdamW
from typing import List, Optional, Tuple, Union


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """
    Compute the zeroth power / orthogonalization of G using Newton-Schulz iteration.

    Uses quintic coefficients (a, b, c) that minimize convergence time for matrices
    close to the identity. These coefficients were derived by solving for the
    polynomial that maximizes the spectral gap.

    Args:
        G: Gradient tensor of shape (..., m, n)
        steps: Number of Newton-Schulz iterations (default: 5)

    Returns:
        Orthogonalized tensor of same shape as G
    """
    assert G.ndim >= 2, "Input must be at least 2D"

    # Quintic coefficients optimized for fast convergence
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Use bfloat16 for efficiency
    X = G.bfloat16()

    # Transpose if tall matrix (more rows than columns)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    # Normalize to unit spectral norm
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    # Transpose back if needed
    if transposed:
        X = X.mT

    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer for 2D weight matrices in hidden layers.

    Applies momentum followed by Newton-Schulz orthogonalization.
    For non-2D parameters, falls back to standard momentum SGD behavior.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        weight_decay: Weight decay coefficient (default: 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay (decoupled, AdamW-style)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Get or initialize momentum buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                buf.lerp_(grad, 1 - momentum)

                # Compute update with optional Nesterov momentum
                if nesterov:
                    update = grad.lerp(buf, momentum)
                else:
                    update = buf.clone()

                # Apply Newton-Schulz orthogonalization for 2D+ parameters
                if update.ndim >= 2:
                    # For 4D conv weights, reshape to 2D
                    original_shape = update.shape
                    if update.ndim == 4:
                        update = update.view(update.size(0), -1)

                    # Orthogonalize
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                    # Scale by aspect ratio
                    update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5

                    # Reshape back if needed
                    if len(original_shape) == 4:
                        update = update.view(original_shape)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss


class MuonWithAdamW(Optimizer):
    """
    Hybrid optimizer: Muon for 2D weights, AdamW for 1D params (biases, gains).

    This wrapper automatically separates parameters based on their dimensionality:
    - 2D+ parameters (conv weights, linear weights): Muon optimizer
    - 1D parameters (biases, LayerNorm weights): AdamW optimizer

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate for Muon (default: 0.02)
        adam_lr: Learning rate for AdamW. If None, uses lr/10 (default: None)
        momentum: Muon momentum (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iterations (default: 5)
        weight_decay: Weight decay for Muon (default: 0.01)
        adam_weight_decay: Weight decay for AdamW. If None, uses weight_decay (default: None)
        adam_betas: AdamW beta coefficients (default: (0.9, 0.95))
        adam_eps: AdamW epsilon (default: 1e-8)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        adam_lr: Optional[float] = None,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        adam_weight_decay: Optional[float] = None,
        adam_betas: Tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
    ):
        # Collect all parameters
        params_list = list(params)

        # Separate 2D+ params (Muon) from 1D params (AdamW)
        muon_params = []
        adam_params = []

        for p in params_list:
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adam_params.append(p)

        # Store for state dict compatibility
        self._all_params = params_list
        self._muon_params = muon_params
        self._adam_params = adam_params

        # Initialize Muon optimizer for 2D+ params
        if muon_params:
            self.muon = Muon(
                muon_params,
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                weight_decay=weight_decay,
            )
        else:
            self.muon = None

        # Initialize AdamW optimizer for 1D params
        adam_lr = adam_lr if adam_lr is not None else lr / 10
        adam_weight_decay = adam_weight_decay if adam_weight_decay is not None else weight_decay

        if adam_params:
            self.adam = AdamW(
                adam_params,
                lr=adam_lr,
                betas=adam_betas,
                eps=adam_eps,
                weight_decay=adam_weight_decay,
            )
        else:
            self.adam = None

        # Initialize base optimizer (required for LR schedulers)
        defaults = dict(
            lr=lr,
            adam_lr=adam_lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            adam_weight_decay=adam_weight_decay,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
        )
        super().__init__(params_list, defaults)

        # Report parameter distribution
        n_muon = sum(p.numel() for p in muon_params)
        n_adam = sum(p.numel() for p in adam_params)
        total = n_muon + n_adam
        print(f"MuonWithAdamW: {len(muon_params)} Muon params ({n_muon:,} elements, {100*n_muon/total:.1f}%), "
              f"{len(adam_params)} AdamW params ({n_adam:,} elements, {100*n_adam/total:.1f}%)")

    @property
    def param_groups(self):
        """Return combined param groups for scheduler compatibility."""
        groups = []
        if self.muon is not None:
            groups.extend(self.muon.param_groups)
        if self.adam is not None:
            groups.extend(self.adam.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value):
        """Required for optimizer base class compatibility."""
        pass

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients of all optimized parameters."""
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adam is not None:
            self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.muon is not None:
            self.muon.step()
        if self.adam is not None:
            self.adam.step()

        return loss

    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        return {
            'muon': self.muon.state_dict() if self.muon else None,
            'adam': self.adam.state_dict() if self.adam else None,
        }

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        if self.muon is not None and state_dict.get('muon') is not None:
            self.muon.load_state_dict(state_dict['muon'])
        if self.adam is not None and state_dict.get('adam') is not None:
            self.adam.load_state_dict(state_dict['adam'])
