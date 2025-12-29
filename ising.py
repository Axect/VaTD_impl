import numpy as np
import torch


def energy(x):
    """
    x: 2D numpy array of spins

    Returns the energy of the Ising model configuration.
    E(x) = -J * sum(s_i * s_j) - h * sum(s_i)
    where J is the interaction strength, h is the external magnetic field,
    and the sums are over nearest neighbors.
    Here, we assume J = 1 and h = 0 for simplicity.
    """
    J = 1  # Interaction strength
    h = 0  # External magnetic field
    n, m = x.shape
    E = 0

    # Sum over nearest neighbors
    for i in range(n):
        for j in range(m):
            E -= J * x[i, j] * (x[(i + 1) % n, j] + x[i, (j + 1) % m])
            E -= h * x[i, j]

    return E


def swendsen_wang_update(
    samples: torch.Tensor,
    T: torch.Tensor,
    n_sweeps: int = 1,
    fix_first: bool = True
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
        fix_first: Whether to fix the first spin (0,0) to +1 or -1 (prevents global flip)

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
        # Worst case diameter is H*W, but usually converges much faster
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

        # Don't flip fixed position's cluster if fix_first is requested
        if fix_first:
            fixed_label = labels[:, 0, 0].view(B, 1, 1)
            # If a cluster has the same label as the fixed position, do not flip it
            flip_mask = flip_mask & (labels != fixed_label)

        # Flip selected clusters
        improved[:, 0] = torch.where(flip_mask, -spins, spins)

    return improved


def wolff_cluster_update(
    samples: torch.Tensor,
    T: torch.Tensor,
    n_clusters: int = 10,
    fix_first: bool = True
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
        fix_first: Whether to fix the first spin

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
            if fix_first and seed_i == 0 and seed_j == 0:
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
                    if fix_first and ni == 0 and nj == 0:
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


def metropolis_hastings_update(
    samples: torch.Tensor,
    T: torch.Tensor,
    n_steps: int = 10,
    fix_first: bool = True
) -> torch.Tensor:
    """
    Metropolis-Hastings update with checkerboard parallelization.

    Unlike cluster algorithms, MH explicitly uses energy changes to accept/reject
    moves, ensuring convergence to the Boltzmann distribution regardless of
    the initial distribution.

    Args:
        samples: Input samples (B, 1, H, W) in {-1, +1}
        T: Temperature values (B,)
        n_steps: Number of MH sweeps (each sweep updates all sites)
        fix_first: Whether to fix the first spin

    Returns:
        Improved samples (B, 1, H, W) in {-1, +1}
    """
    B, C, H, W = samples.shape
    device = samples.device
    improved = samples.clone()
    beta = 1.0 / T.view(B, 1, 1)  # (B, 1, 1)

    # Create checkerboard masks
    row_idx = torch.arange(H, device=device).view(1, H, 1)
    col_idx = torch.arange(W, device=device).view(1, 1, W)
    even_mask = ((row_idx + col_idx) % 2 == 0).expand(B, H, W)
    odd_mask = ~even_mask

    # Fixed first position mask
    if fix_first:
        fixed_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
        fixed_mask[:, 0, 0] = True
    else:
        fixed_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

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

    return improved


def hybrid_cluster_mh_update(
    samples: torch.Tensor,
    T: torch.Tensor,
    n_sw_sweeps: int = 5,
    n_mh_steps: int = 20,
    fix_first: bool = True
) -> torch.Tensor:
    """
    Hybrid cluster + MH update for optimal equilibration.

    Combines the strengths of both algorithms:
    1. Swendsen-Wang: Fast decorrelation, eliminates critical slowing down
    2. Metropolis-Hastings: Energy-guided convergence to Boltzmann distribution

    This is particularly effective at Tc where:
    - SW moves quickly through phase space (no critical slowing down)
    - MH then "fine-tunes" toward low-energy configurations

    Args:
        samples: Input samples (B, 1, H, W) in {-1, +1}
        T: Temperature values (B,)
        n_sw_sweeps: Number of Swendsen-Wang sweeps for decorrelation
        n_mh_steps: Number of MH sweeps for energy-guided refinement
        fix_first: Whether to fix the first spin

    Returns:
        Improved samples (B, 1, H, W) in {-1, +1}
    """
    # Step 1: Swendsen-Wang for fast decorrelation
    # This efficiently explores configuration space without critical slowing down
    decorrelated = swendsen_wang_update(samples, T, n_sweeps=n_sw_sweeps, fix_first=fix_first)

    # Step 2: Metropolis-Hastings for energy-guided convergence
    # This ensures we actually move toward the Boltzmann distribution
    improved = metropolis_hastings_update(decorrelated, T, n_steps=n_mh_steps, fix_first=fix_first)

    return improved
