"""
q-state Potts model energy functions and MCMC samplers.

The q-state Potts model generalizes the Ising model (q=2) to q discrete spin states.
Energy: E = -J * Σ_{<i,j>} δ(s_i, s_j), where s_i ∈ {0, 1, ..., q-1}.

Critical temperature: Tc = 1 / ln(1 + √q)
- q=2: Tc ≈ 2.269 (Ising)
- q=3: Tc ≈ 0.995
- q=4: Tc ≈ 0.910

References:
- Wu, F.Y. (1982). "The Potts model." Rev. Mod. Phys. 54, 235.
- Baxter, R.J. (1973). "Potts model at the critical temperature." J. Phys. C 6, L445.
"""

import torch
import numpy as np


def potts_critical_temperature(q: int) -> float:
    """
    Exact critical temperature of the q-state Potts model on 2D square lattice.

    Tc = 1 / ln(1 + √q)

    Valid for all q ≥ 2. The transition is second-order for q ≤ 4
    and first-order for q ≥ 5.
    """
    return 1.0 / np.log(1.0 + np.sqrt(q))


def create_potts_energy_fn(L: int, q: int = 3, d: int = 2, device: str = "cpu"):
    """
    Create energy function for q-state Potts model on a d-dimensional lattice.

    Uses O(N) sparse computation with periodic boundary conditions.
    Energy: E = -J * Σ_{<i,j>} δ(s_i, s_j), J = 1.

    Args:
        L: Lattice size
        q: Number of Potts states (default 3)
        d: Dimension (default 2)
        device: Torch device

    Returns:
        Energy function: samples (B, 1, H, W) with values in {0,...,q-1} → (B, 1)
    """

    def energy_fn(samples):
        """
        Calculate Potts energy for batch of samples.

        Args:
            samples: (B, 1, H, W) tensor with values in {0, ..., q-1}

        Returns:
            (B, 1) energy values

        Energy: E = -J * Σ_{<i,j>} δ(s_i, s_j)
        With periodic BCs, each bond counted once (right + down).
        """
        right_neighbor = torch.roll(samples, shifts=-1, dims=-1)
        down_neighbor = torch.roll(samples, shifts=-1, dims=-2)

        # Kronecker delta: 1 if same state, 0 otherwise
        delta_right = (samples == right_neighbor).float()
        delta_down = (samples == down_neighbor).float()

        energy = -(delta_right + delta_down)
        energy = energy.sum(dim=[-1, -2, -3]).unsqueeze(-1)

        return energy

    # Attach metadata
    energy_fn.model_type = "potts"
    energy_fn.q = q
    energy_fn.lattice_size = L

    return energy_fn


def swendsen_wang_potts_update(
    samples: torch.Tensor,
    T: torch.Tensor,
    q: int = 3,
    n_sweeps: int = 1,
    fix_first: bool = True,
) -> torch.Tensor:
    """
    Swendsen-Wang cluster algorithm for q-state Potts model.

    Generalizes the Ising SW algorithm: bonds activated between same-state
    neighbors with probability p = 1 - exp(-β·J). Each cluster is assigned
    a uniformly random new state from {0, ..., q-1}.

    Args:
        samples: Input samples (B, 1, H, W) with values in {0, ..., q-1}
        T: Temperature values (B,)
        q: Number of Potts states
        n_sweeps: Number of full SW sweeps
        fix_first: Whether to fix the first spin (0,0)

    Returns:
        Updated samples (B, 1, H, W) with values in {0, ..., q-1}
    """
    B, C, H, W = samples.shape
    device = samples.device
    improved = samples.clone()

    # Bond probability: p = 1 - exp(-β)
    beta = 1.0 / T.view(B, 1, 1)
    p_bond = 1.0 - torch.exp(-beta)

    for _ in range(n_sweeps):
        spins = improved[:, 0]  # (B, H, W)

        # Activate bonds between same-state neighbors
        same_right = (spins == torch.roll(spins, -1, dims=2))
        same_down = (spins == torch.roll(spins, -1, dims=1))

        bond_right = same_right & (torch.rand(B, H, W, device=device) < p_bond)
        bond_down = same_down & (torch.rand(B, H, W, device=device) < p_bond)

        # Find connected components via iterative label propagation
        labels = torch.arange(H * W, device=device).view(1, H, W).expand(B, -1, -1).float()
        labels = labels + torch.arange(B, device=device).view(B, 1, 1) * H * W

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

        # Assign random new state to each cluster
        # Hash label to get deterministic random state per cluster
        cluster_state = torch.fmod(labels * 2654435761, 2**32) / 2**32  # [0, 1)
        new_state = (cluster_state * q).long().clamp(0, q - 1)

        # Keep fixed position's cluster at its original state
        if fix_first:
            fixed_label = labels[:, 0, 0].view(B, 1, 1)
            fixed_state = spins[:, 0, 0].view(B, 1, 1).long()
            new_state = torch.where(labels == fixed_label, fixed_state, new_state)

        improved[:, 0] = new_state

    return improved


def metropolis_hastings_potts_update(
    samples: torch.Tensor,
    T: torch.Tensor,
    q: int = 3,
    n_steps: int = 10,
    fix_first: bool = True,
) -> torch.Tensor:
    """
    Metropolis-Hastings update for q-state Potts model with checkerboard parallelization.

    Args:
        samples: Input samples (B, 1, H, W) with values in {0, ..., q-1}
        T: Temperature values (B,)
        q: Number of Potts states
        n_steps: Number of MH sweeps
        fix_first: Whether to fix the first spin

    Returns:
        Updated samples (B, 1, H, W) with values in {0, ..., q-1}
    """
    B, C, H, W = samples.shape
    device = samples.device
    improved = samples.clone()
    beta = 1.0 / T.view(B, 1, 1)  # (B, 1, 1)

    # Checkerboard masks
    row_idx = torch.arange(H, device=device).view(1, H, 1)
    col_idx = torch.arange(W, device=device).view(1, 1, W)
    even_mask = ((row_idx + col_idx) % 2 == 0).expand(B, H, W)
    odd_mask = ~even_mask

    if fix_first:
        fixed_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
        fixed_mask[:, 0, 0] = True
    else:
        fixed_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

    for _ in range(n_steps):
        for mask in [even_mask, odd_mask]:
            spins = improved[:, 0]  # (B, H, W)

            # Propose random new state
            proposal = torch.randint(0, q, (B, H, W), device=device)

            # Count same-state neighbors for current and proposed
            neighbors = [
                torch.roll(spins, 1, dims=1),
                torch.roll(spins, -1, dims=1),
                torch.roll(spins, 1, dims=2),
                torch.roll(spins, -1, dims=2),
            ]

            n_same_current = sum((spins == nb).float() for nb in neighbors)
            n_same_proposal = sum((proposal == nb).float() for nb in neighbors)

            # ΔE = -(n_same_proposal - n_same_current)
            delta_E = -(n_same_proposal - n_same_current)

            # Metropolis acceptance
            accept_prob = torch.where(
                delta_E <= 0,
                torch.ones_like(delta_E),
                torch.exp(-beta * delta_E),
            )

            accept = torch.rand_like(accept_prob) < accept_prob
            accept = accept & mask & ~fixed_mask

            improved[:, 0] = torch.where(accept, proposal, spins)

    return improved


def mcmc_potts_update(
    samples: torch.Tensor,
    T: torch.Tensor,
    q: int = 3,
    n_sw_sweeps: int = 5,
    n_mh_steps: int = 0,
    fix_first: bool = True,
) -> torch.Tensor:
    """
    MCMC update for q-state Potts model.

    Args:
        samples: Input samples (B, 1, H, W) with values in {0, ..., q-1}
        T: Temperature values (B,)
        q: Number of Potts states
        n_sw_sweeps: Number of Swendsen-Wang sweeps
        n_mh_steps: Number of MH sweeps (optional refinement)
        fix_first: Whether to fix the first spin

    Returns:
        Equilibrated samples (B, 1, H, W)
    """
    improved = swendsen_wang_potts_update(
        samples, T, q=q, n_sweeps=n_sw_sweeps, fix_first=fix_first
    )

    if n_mh_steps > 0:
        improved = metropolis_hastings_potts_update(
            improved, T, q=q, n_steps=n_mh_steps, fix_first=fix_first
        )

    return improved
