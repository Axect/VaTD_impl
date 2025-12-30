#!/usr/bin/env python3
"""
Generate MCMC reference data for XY model validation.

This script runs Metropolis-Hastings MCMC to generate ground truth values
for energy and magnetization at various temperatures. These values are used
to validate the ContinuousFlowMatcher model since no exact analytical solution
exists for the 2D XY model (unlike the Ising model's Onsager solution).

Usage:
    python generate_xy_reference.py --L 16 --output refs/xy_L16.pt
    python generate_xy_reference.py --L 16 --n_samples 10000 --n_thermalize 5000

The output file contains a dictionary mapping temperature to observables:
    {
        T: {
            'energy_mean': float,
            'energy_std': float,
            'mag_mean': float,
            'mag_std': float,
            'n_samples': int,
        },
        ...
    }
"""

import torch
import numpy as np
import math
import argparse
import os
from tqdm import tqdm


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def compute_xy_energy(angles: torch.Tensor, J: float = 1.0) -> torch.Tensor:
    """
    Compute XY energy for a batch of configurations.

    E = -J * sum_{<ij>} cos(theta_i - theta_j)

    Args:
        angles: (B, 1, H, W) angles in [-pi, pi]
        J: Coupling constant

    Returns:
        (B, 1) energy values
    """
    diff_right = angles - torch.roll(angles, shifts=-1, dims=-1)
    diff_down = angles - torch.roll(angles, shifts=-1, dims=-2)

    energy = -J * (torch.cos(diff_right) + torch.cos(diff_down))
    energy = energy.sum(dim=[-1, -2, -3]).unsqueeze(-1)

    return energy


def compute_xy_magnetization(angles: torch.Tensor) -> torch.Tensor:
    """
    Compute XY magnetization.

    M = (1/N) * |sum_i exp(i*theta_i)|
      = (1/N) * sqrt((sum cos(theta))^2 + (sum sin(theta))^2)

    Args:
        angles: (B, 1, H, W) angles in [-pi, pi]

    Returns:
        (B,) magnetization in [0, 1]
    """
    cos_sum = torch.cos(angles).sum(dim=[1, 2, 3])
    sin_sum = torch.sin(angles).sum(dim=[1, 2, 3])
    N = angles[0].numel()
    mag = torch.sqrt(cos_sum ** 2 + sin_sum ** 2) / N
    return mag


def metropolis_xy_sweep(
    angles: torch.Tensor,
    T: float,
    J: float = 1.0,
    proposal_std: float = 0.5,
    fix_first: bool = True
) -> torch.Tensor:
    """
    Perform one Metropolis-Hastings sweep over all sites.

    Args:
        angles: (B, 1, H, W) current angles
        T: Temperature
        J: Coupling constant
        proposal_std: Standard deviation of angle proposal
        fix_first: If True, keep first spin at 0

    Returns:
        Updated angles
    """
    B, _, H, W = angles.shape
    beta = 1.0 / T

    for i in range(H):
        for j in range(W):
            # Skip fixed first spin
            if fix_first and i == 0 and j == 0:
                continue

            # Current angle
            theta_old = angles[:, 0, i, j]

            # Propose new angle
            delta = torch.randn_like(theta_old) * proposal_std
            theta_new = wrap_angle(theta_old + delta)

            # Get neighbor angles
            theta_right = angles[:, 0, i, (j + 1) % W]
            theta_left = angles[:, 0, i, (j - 1) % W]
            theta_down = angles[:, 0, (i + 1) % H, j]
            theta_up = angles[:, 0, (i - 1) % H, j]

            # Old energy contribution at this site
            E_old = -J * (
                torch.cos(theta_old - theta_right) +
                torch.cos(theta_old - theta_left) +
                torch.cos(theta_old - theta_down) +
                torch.cos(theta_old - theta_up)
            )

            # New energy contribution
            E_new = -J * (
                torch.cos(theta_new - theta_right) +
                torch.cos(theta_new - theta_left) +
                torch.cos(theta_new - theta_down) +
                torch.cos(theta_new - theta_up)
            )

            # Acceptance probability
            delta_E = E_new - E_old
            accept_prob = torch.exp(-beta * delta_E).clamp(max=1.0)
            accept = torch.rand_like(accept_prob) < accept_prob

            # Update
            angles[:, 0, i, j] = torch.where(accept, theta_new, theta_old)

    return angles


def generate_reference_data(
    L: int,
    temperatures: list,
    n_samples: int = 10000,
    n_thermalize: int = 5000,
    n_decorrelate: int = 10,
    batch_size: int = 64,
    J: float = 1.0,
    fix_first: bool = True,
    device: str = "cpu",
    verbose: bool = True
) -> dict:
    """
    Generate MCMC reference data for each temperature.

    Args:
        L: Lattice size
        temperatures: List of temperatures to simulate
        n_samples: Number of samples to collect per temperature
        n_thermalize: Number of sweeps for thermalization
        n_decorrelate: Sweeps between samples for decorrelation
        batch_size: Number of parallel chains
        J: Coupling constant
        fix_first: Fix first spin at 0
        device: Torch device
        verbose: Print progress

    Returns:
        Dictionary mapping temperature to observables
    """
    reference_data = {}

    for T in tqdm(temperatures, desc="Temperatures", disable=not verbose):
        if verbose:
            print(f"\nSimulating T = {T:.3f} (beta = {1/T:.3f})")

        # Initialize random angles
        angles = torch.rand(batch_size, 1, L, L, device=device) * 2 * math.pi - math.pi

        # Fix first spin
        if fix_first:
            angles[:, :, 0, 0] = 0.0

        # Thermalization
        if verbose:
            print(f"  Thermalizing ({n_thermalize} sweeps)...")
        for _ in tqdm(range(n_thermalize), desc="  Thermalize", leave=False, disable=not verbose):
            angles = metropolis_xy_sweep(angles, T, J=J, fix_first=fix_first)

        # Collection
        energies = []
        magnetizations = []
        samples_collected = 0

        if verbose:
            print(f"  Collecting {n_samples} samples...")

        pbar = tqdm(total=n_samples, desc="  Collect", leave=False, disable=not verbose)
        while samples_collected < n_samples:
            # Decorrelate
            for _ in range(n_decorrelate):
                angles = metropolis_xy_sweep(angles, T, J=J, fix_first=fix_first)

            # Compute observables
            energy = compute_xy_energy(angles, J=J)
            mag = compute_xy_magnetization(angles)

            energies.append(energy.squeeze(-1).cpu())
            magnetizations.append(mag.cpu())

            samples_collected += batch_size
            pbar.update(batch_size)

        pbar.close()

        # Stack all samples
        all_energies = torch.cat(energies, dim=0)[:n_samples]
        all_mags = torch.cat(magnetizations, dim=0)[:n_samples]

        # Compute statistics
        reference_data[T] = {
            'energy_mean': all_energies.mean().item(),
            'energy_std': all_energies.std().item(),
            'mag_mean': all_mags.mean().item(),
            'mag_std': all_mags.std().item(),
            'n_samples': len(all_energies),
        }

        if verbose:
            print(f"  E = {reference_data[T]['energy_mean']:.4f} +/- {reference_data[T]['energy_std']:.4f}")
            print(f"  M = {reference_data[T]['mag_mean']:.4f} +/- {reference_data[T]['mag_std']:.4f}")

    return reference_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate MCMC reference data for XY model validation"
    )
    parser.add_argument("--L", type=int, default=16, help="Lattice size")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--n_samples", type=int, default=10000, help="Samples per temperature")
    parser.add_argument("--n_thermalize", type=int, default=5000, help="Thermalization sweeps")
    parser.add_argument("--n_decorrelate", type=int, default=10, help="Decorrelation sweeps")
    parser.add_argument("--batch_size", type=int, default=64, help="Parallel chains")
    parser.add_argument("--J", type=float, default=1.0, help="Coupling constant")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--T_min", type=float, default=0.5, help="Minimum temperature")
    parser.add_argument("--T_max", type=float, default=2.0, help="Maximum temperature")
    parser.add_argument("--n_temps", type=int, default=16, help="Number of temperatures")
    parser.add_argument("--no_fix_first", action="store_true", help="Don't fix first spin")

    args = parser.parse_args()

    # Generate temperature range
    # Include temperatures around BKT transition (~0.89)
    temperatures = np.linspace(args.T_min, args.T_max, args.n_temps).tolist()

    print("=" * 60)
    print("XY Model MCMC Reference Data Generator")
    print("=" * 60)
    print(f"Lattice size: {args.L}x{args.L}")
    print(f"Temperatures: {args.n_temps} values in [{args.T_min:.2f}, {args.T_max:.2f}]")
    print(f"BKT transition: T_BKT ~ 0.89")
    print(f"Samples per T: {args.n_samples}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Generate data
    reference_data = generate_reference_data(
        L=args.L,
        temperatures=temperatures,
        n_samples=args.n_samples,
        n_thermalize=args.n_thermalize,
        n_decorrelate=args.n_decorrelate,
        batch_size=args.batch_size,
        J=args.J,
        fix_first=not args.no_fix_first,
        device=args.device,
        verbose=True
    )

    # Add metadata
    reference_data['_metadata'] = {
        'L': args.L,
        'J': args.J,
        'n_samples': args.n_samples,
        'n_thermalize': args.n_thermalize,
        'n_decorrelate': args.n_decorrelate,
        'fix_first': not args.no_fix_first,
        'temperatures': temperatures,
    }

    # Save
    if args.output is None:
        args.output = f"refs/xy_L{args.L}.pt"

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    torch.save(reference_data, args.output)

    print("\n" + "=" * 60)
    print(f"Reference data saved to: {args.output}")
    print("=" * 60)

    # Summary table
    print("\nSummary:")
    print("-" * 50)
    print(f"{'T':>8} {'E':>12} {'M':>12}")
    print("-" * 50)
    for T in temperatures:
        data = reference_data[T]
        print(f"{T:>8.3f} {data['energy_mean']:>12.4f} {data['mag_mean']:>12.4f}")
    print("-" * 50)


if __name__ == "__main__":
    main()
