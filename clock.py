"""
q-state clock model energy functions and MCMC samplers.

The q-state clock model discretizes the XY model into q equally spaced angles.
Energy: E = -J * Σ_{<i,j>} cos(2π(s_i - s_j)/q), where s_i ∈ {0, 1, ..., q-1}.

For q → ∞, recovers the continuous XY model (BKT transition at T_BKT ≈ 0.89).
For q ≤ 4, the model is in the Potts universality class.
For q ≥ 5, the clock model has TWO transitions:
  - T_1: ordered → BKT-like (lower)
  - T_2: BKT-like → disordered (upper)
For large q, both transitions converge to T_BKT ≈ 0.89.

References:
- José, J.V. et al. (1977). "Renormalization, vortices, and symmetry-breaking
  perturbations in the two-dimensional planar model." Phys. Rev. B 16, 1217.
- Lapilli, C.M. et al. (2006). "Universality away from critical points in
  two-dimensional phase transitions." Phys. Rev. Lett. 96, 140603.
"""

import torch
import math


# Approximate BKT transition temperatures for various q
# For q >= 5, two transitions exist; we store the upper one (T_2 ~ T_BKT)
# Values from MC literature (Lapilli et al. 2006, Borisenko et al. 2011)
CLOCK_TC = {
    6: 0.70,
    8: 0.80,
    12: 0.88,
    16: 0.89,
    24: 0.89,
    36: 0.89,
    48: 0.89,
    64: 0.89,
}

# Central charge: c=1 free boson CFT (same as XY model for q >= 5)
CLOCK_CENTRAL_CHARGE = 1.0


def create_clock_energy_fn(L: int, q: int, d: int = 2, device: str = "cpu", J: float = 1.0):
    """
    Create energy function for q-state clock model on a d-dimensional lattice.

    Uses O(N) computation with periodic boundary conditions.
    Energy: E = -J * Σ_{<i,j>} cos(2π(s_i - s_j)/q)

    Args:
        L: Lattice size
        q: Number of clock states
        d: Dimension (default 2)
        device: Torch device
        J: Coupling constant (default 1.0)

    Returns:
        Energy function: samples (B, 1, H, W) with values in {0,...,q-1} → (B, 1)
    """

    def energy_fn(samples):
        """
        Calculate clock energy for batch of samples.

        Args:
            samples: (B, 1, H, W) tensor with values in {0, ..., q-1}

        Returns:
            (B, 1) energy values

        Energy: E = -J * Σ_{<i,j>} cos(2π(s_i - s_j)/q)
        With periodic BCs, each bond counted once (right + down).
        """
        angles = 2.0 * math.pi / q
        diff_right = samples - torch.roll(samples, shifts=-1, dims=-1)
        diff_down = samples - torch.roll(samples, shifts=-1, dims=-2)
        energy = -J * (torch.cos(angles * diff_right) + torch.cos(angles * diff_down))
        energy = energy.sum(dim=[-1, -2, -3]).unsqueeze(-1)

        return energy

    # Attach metadata
    energy_fn.model_type = "clock"
    energy_fn.q = q
    energy_fn.lattice_size = L
    energy_fn.J = J

    return energy_fn


def wolff_clock_update(
    samples: torch.Tensor,
    T: torch.Tensor,
    q: int,
    n_clusters: int = 10,
    fix_first: bool = True,
    J: float = 1.0,
) -> torch.Tensor:
    """
    Wolff cluster algorithm adapted for the clock model using the embedding trick.

    Chooses a random reflection axis r ∈ [0, 2π), projects spins onto it via
    cos(2πs/q - r), then grows a cluster based on same-sign projections.
    Bond probability: p = 1 - exp(-2βJ * |proj_i| * |proj_j|) when both
    projections have the same sign as the seed. Reflects cluster spins about
    the axis r: new_angle = 2r - old_angle, then snapped to nearest clock state.

    Args:
        samples: Input samples (B, 1, H, W) with values in {0, ..., q-1}
        T: Temperature values (B,)
        q: Number of clock states
        n_clusters: Number of cluster updates per sample
        fix_first: Whether to fix the first spin (0,0)
        J: Coupling constant

    Returns:
        Updated samples (B, 1, H, W) with values in {0, ..., q-1}
    """
    B, _, H, W = samples.shape
    device = samples.device
    improved = samples.clone()

    angle_step = 2.0 * math.pi / q
    beta = 1.0 / T  # (B,)

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    for _ in range(n_clusters):
        # Random reflection axis per batch
        r = torch.rand(B, device=device) * 2 * math.pi

        for b in range(B):
            spins = improved[b, 0]  # (H, W)
            r_b = r[b].item()
            beta_b = beta[b].item()

            # Project all spins onto axis
            angles = spins.float() * angle_step
            projections = torch.cos(angles - r_b)

            # Random seed position
            seed_i = torch.randint(0, H, (1,), device=device).item()
            seed_j = torch.randint(0, W, (1,), device=device).item()

            if fix_first and seed_i == 0 and seed_j == 0:
                continue

            seed_proj = projections[seed_i, seed_j].item()
            seed_sign = 1 if seed_proj >= 0 else -1

            # BFS cluster growth
            cluster = set()
            cluster.add((seed_i, seed_j))
            queue = [(seed_i, seed_j)]

            while queue:
                ci, cj = queue.pop(0)
                proj_c = projections[ci, cj].item()

                for di, dj in directions:
                    ni, nj = (ci + di) % H, (cj + dj) % W

                    if (ni, nj) in cluster:
                        continue
                    if fix_first and ni == 0 and nj == 0:
                        continue

                    proj_n = projections[ni, nj].item()

                    # Both must have same sign as seed
                    if (proj_n >= 0) != (seed_sign >= 0):
                        continue

                    # Bond probability based on projection magnitudes
                    p_bond = 1.0 - math.exp(-2.0 * beta_b * J * abs(proj_c) * abs(proj_n))

                    if torch.rand(1, device=device).item() < p_bond:
                        cluster.add((ni, nj))
                        queue.append((ni, nj))

            # Reflect cluster spins about axis r
            for (fi, fj) in cluster:
                old_angle = improved[b, 0, fi, fj].item() * angle_step
                # Reflect: new_angle = 2*r - old_angle
                new_angle = 2.0 * r_b - old_angle
                # Snap to nearest clock state
                new_state = round(new_angle / angle_step) % q
                improved[b, 0, fi, fj] = new_state

    return improved


def overrelaxation_clock_update(
    samples: torch.Tensor,
    T: torch.Tensor,
    q: int,
    n_sweeps: int = 1,
    fix_first: bool = True,
    J: float = 1.0,
) -> torch.Tensor:
    """
    Microcanonical overrelaxation update for the clock model.

    Computes the local field direction from neighbors, reflects the spin angle
    about it, then snaps to the nearest clock state. This is a deterministic
    (microcanonical) update that preserves the energy to leading order.
    Uses checkerboard parallelization for efficient batched execution.

    Args:
        samples: Input samples (B, 1, H, W) with values in {0, ..., q-1}
        T: Temperature values (B,) — not used directly (microcanonical), kept for API consistency
        q: Number of clock states
        n_sweeps: Number of checkerboard sweep pairs
        fix_first: Whether to fix the first spin (0,0)
        J: Coupling constant

    Returns:
        Updated samples (B, 1, H, W) with values in {0, ..., q-1}
    """
    B, _, H, W = samples.shape
    device = samples.device
    improved = samples.clone()
    angle_step = 2.0 * math.pi / q

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

    for _ in range(n_sweeps):
        for mask in [even_mask, odd_mask]:
            spins = improved[:, 0]  # (B, H, W)
            current_angles = spins.float() * angle_step

            # Compute local field from neighbors
            neighbors = [
                torch.roll(spins, 1, dims=1),
                torch.roll(spins, -1, dims=1),
                torch.roll(spins, 1, dims=2),
                torch.roll(spins, -1, dims=2),
            ]

            # Local field direction: sum of cos and sin components from neighbors
            hx = sum(torch.cos(nb.float() * angle_step) for nb in neighbors) * J
            hy = sum(torch.sin(nb.float() * angle_step) for nb in neighbors) * J
            h_angle = torch.atan2(hy, hx)  # (B, H, W)

            # Reflect current angle about local field direction
            new_angle = 2.0 * h_angle - current_angles

            # Snap to nearest clock state
            new_state = torch.round(new_angle / angle_step) % q
            new_state = new_state.long()

            update_mask = mask & ~fixed_mask
            improved[:, 0] = torch.where(update_mask, new_state, spins)

    return improved


def metropolis_clock_update(
    samples: torch.Tensor,
    T: torch.Tensor,
    q: int,
    n_steps: int = 10,
    fix_first: bool = True,
    J: float = 1.0,
) -> torch.Tensor:
    """
    Metropolis-Hastings update for the clock model with checkerboard parallelization.

    Proposes random new clock states and accepts/rejects based on the cosine
    energy difference. Uses checkerboard (even/odd sublattice) decomposition
    to update all sites in a sublattice simultaneously without conflicts.

    Args:
        samples: Input samples (B, 1, H, W) with values in {0, ..., q-1}
        T: Temperature values (B,)
        q: Number of clock states
        n_steps: Number of MH sweeps (each sweep = even + odd sublattice pass)
        fix_first: Whether to fix the first spin (0,0)
        J: Coupling constant

    Returns:
        Updated samples (B, 1, H, W) with values in {0, ..., q-1}
    """
    B, _, H, W = samples.shape
    device = samples.device
    improved = samples.clone()
    beta = 1.0 / T.view(B, 1, 1)
    angle_step = 2.0 * math.pi / q

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
            spins = improved[:, 0]
            proposal = torch.randint(0, q, (B, H, W), device=device)

            # Neighbor spins
            neighbors = [
                torch.roll(spins, 1, dims=1),
                torch.roll(spins, -1, dims=1),
                torch.roll(spins, 1, dims=2),
                torch.roll(spins, -1, dims=2),
            ]

            # Current energy contribution (sum of cos interactions)
            E_current = sum(
                torch.cos((spins.float() - nb.float()) * angle_step) for nb in neighbors
            )

            # Proposed energy contribution
            E_proposal = sum(
                torch.cos((proposal.float() - nb.float()) * angle_step) for nb in neighbors
            )

            # ΔE = -J * (E_proposal - E_current) [note: cosine already has the sign]
            delta_E = -J * (E_proposal - E_current)

            accept_prob = torch.where(
                delta_E <= 0,
                torch.ones_like(delta_E),
                torch.exp(-beta * delta_E),
            )

            accept = torch.rand_like(accept_prob) < accept_prob
            accept = accept & mask & ~fixed_mask

            improved[:, 0] = torch.where(accept, proposal, spins)

    return improved


def mcmc_clock_update(
    samples: torch.Tensor,
    T: torch.Tensor,
    q: int,
    n_wolff_clusters: int = 5,
    n_or_sweeps: int = 2,
    n_mh_steps: int = 0,
    fix_first: bool = True,
    J: float = 1.0,
) -> torch.Tensor:
    """
    Combined MCMC update for the clock model: Wolff + overrelaxation + optional MH.

    The recommended update sequence combines the non-local cluster moves of
    Wolff (which eliminate critical slowing down) with microcanonical
    overrelaxation (which efficiently explores constant-energy surfaces).
    Optional MH steps provide exact detailed balance.

    Args:
        samples: Input samples (B, 1, H, W) with values in {0, ..., q-1}
        T: Temperature values (B,)
        q: Number of clock states
        n_wolff_clusters: Number of Wolff cluster updates per sample
        n_or_sweeps: Number of overrelaxation sweep pairs
        n_mh_steps: Number of Metropolis-Hastings sweeps (0 = skip)
        fix_first: Whether to fix the first spin (0,0)
        J: Coupling constant

    Returns:
        Equilibrated samples (B, 1, H, W) with values in {0, ..., q-1}
    """
    improved = wolff_clock_update(
        samples, T, q, n_clusters=n_wolff_clusters, fix_first=fix_first, J=J
    )

    if n_or_sweeps > 0:
        improved = overrelaxation_clock_update(
            improved, T, q, n_sweeps=n_or_sweeps, fix_first=fix_first, J=J
        )

    if n_mh_steps > 0:
        improved = metropolis_clock_update(
            improved, T, q, n_steps=n_mh_steps, fix_first=fix_first, J=J
        )

    return improved


def compute_helicity_modulus(
    samples: torch.Tensor,
    T: torch.Tensor,
    q: int,
    J: float = 1.0,
) -> float:
    """
    Helicity modulus (spin stiffness) Υ.

    Υ = (1/N)[<e_x> - β<j_x²>]
    where e_x = J*cos(Δθ), j_x = J*sin(Δθ) for horizontal bonds.

    The helicity modulus is the order parameter for the BKT transition:
    it jumps from (2/π)T_BKT to 0 at the BKT transition.

    Args:
        samples: (B, 1, H, W) tensor with values in {0, ..., q-1}
        T: Temperature values (B,)
        q: Number of clock states
        J: Coupling constant

    Returns:
        Scalar helicity modulus averaged over the batch
    """
    H, W = samples.shape[2], samples.shape[3]
    N = H * W
    angle_step = 2.0 * math.pi / q

    angles = samples[:, 0].float() * angle_step  # (B, H, W)
    diff_x = angles - torch.roll(angles, shifts=-1, dims=-1)  # horizontal bonds

    e_x = J * torch.cos(diff_x)  # (B, H, W)
    j_x = J * torch.sin(diff_x)  # (B, H, W)

    # Average over sites and batch
    e_x_mean = e_x.sum(dim=[-1, -2]).mean() / N  # scalar
    j_x_sq_mean = (j_x ** 2).sum(dim=[-1, -2]).mean() / N  # scalar

    beta = 1.0 / T.mean()
    upsilon = e_x_mean - beta * j_x_sq_mean * N  # extensive j_x² needs N factor

    return upsilon.item()


def compute_vortex_density(
    samples: torch.Tensor,
    q: int,
) -> float:
    """
    Vortex density: fraction of plaquettes with non-zero winding number.

    For each plaquette (elementary square), compute the sum of angle differences
    around the plaquette. Non-zero winding (|sum| > π) indicates a vortex.

    Args:
        samples: (B, 1, H, W) tensor with values in {0, ..., q-1}
        q: Number of clock states

    Returns:
        Scalar vortex density (fraction of plaquettes containing a vortex)
    """
    angle_step = 2.0 * math.pi / q
    angles = samples[:, 0].float() * angle_step  # (B, H, W)

    def wrap(x):
        """Wrap angle difference to [-π, π]."""
        return torch.remainder(x + math.pi, 2 * math.pi) - math.pi

    # Angle differences around each plaquette (counterclockwise)
    d1 = wrap(torch.roll(angles, -1, dims=-1) - angles)  # right
    d2 = wrap(torch.roll(torch.roll(angles, -1, dims=-1), -1, dims=-2)
              - torch.roll(angles, -1, dims=-1))  # down from right
    d3 = wrap(torch.roll(angles, -1, dims=-2)
              - torch.roll(torch.roll(angles, -1, dims=-1), -1, dims=-2))  # left from bottom-right
    d4 = wrap(angles - torch.roll(angles, -1, dims=-2))  # up from bottom

    winding = (d1 + d2 + d3 + d4) / (2 * math.pi)
    # Vortices have |winding| ≈ ±1
    n_vortices = (winding.abs() > 0.5).float().mean()

    return n_vortices.item()


def compute_clock_magnetization(
    samples: torch.Tensor,
    q: int,
) -> float:
    """
    Vector magnetization: M = (1/N)|Σ exp(i·2π·s/q)|

    Returns scalar magnetization averaged over batch.

    Args:
        samples: (B, 1, H, W) tensor with values in {0, ..., q-1}
        q: Number of clock states

    Returns:
        Scalar magnetization averaged over the batch
    """
    angle_step = 2.0 * math.pi / q
    angles = samples[:, 0].float() * angle_step  # (B, H, W)

    cos_sum = torch.cos(angles).sum(dim=[-1, -2])  # (B,)
    sin_sum = torch.sin(angles).sum(dim=[-1, -2])  # (B,)
    N = samples.shape[2] * samples.shape[3]

    mag = torch.sqrt(cos_sum ** 2 + sin_sum ** 2) / N  # (B,)
    return mag.mean().item()
