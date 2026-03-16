#!/usr/bin/env python3
"""
Generate MCMC reference data for q-state clock model validation.

This script runs combined Wolff + overrelaxation MCMC to generate ground truth
values for energy, magnetization, helicity modulus, and vortex density at various
temperatures. These values are used to validate the autoregressive model since no
exact analytical solution exists for the clock model.

Usage:
    python generate_clock_reference.py --L 16 --q 36 --device cuda:0
    python generate_clock_reference.py --L 16 --q 12 --n_samples 1000 --device cpu

The output file contains a dictionary mapping temperature to observables:
    {
        T: {
            'energy_mean': float,
            'energy_std': float,
            'mag_mean': float,
            'mag_std': float,
            'helicity_mean': float,
            'helicity_std': float,
            'vortex_mean': float,
            'vortex_std': float,
            'n_samples': int,
        },
        ...
    }
"""

import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

from clock import (
    create_clock_energy_fn,
    mcmc_clock_update,
    compute_clock_magnetization,
    compute_helicity_modulus,
    compute_vortex_density,
    CLOCK_TC,
)


def generate_reference_data(
    L: int,
    q: int,
    temperatures: list,
    n_samples: int = 50000,
    n_thermalize: int = 200,
    n_decorrelate: int = 5,
    batch_size: int = 256,
    J: float = 1.0,
    fix_first: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Generate MCMC reference data for each temperature.

    Args:
        L: Lattice size
        q: Number of clock states
        temperatures: List of temperatures to simulate
        n_samples: Number of samples to collect per temperature
        n_thermalize: Number of MCMC sweeps for thermalization
        n_decorrelate: MCMC sweeps between samples for decorrelation
        batch_size: Number of parallel chains
        J: Coupling constant
        fix_first: Fix first spin at state 0
        device: Torch device
        verbose: Print progress

    Returns:
        Dictionary mapping temperature to observables
    """
    energy_fn = create_clock_energy_fn(L=L, q=q, device=device, J=J)
    reference_data = {}

    for T_val in tqdm(temperatures, desc="Temperatures", disable=not verbose):
        if verbose:
            print(f"\nSimulating T = {T_val:.4f} (beta = {1/T_val:.4f})")

        # Initialize random clock states
        samples = torch.randint(0, q, (batch_size, 1, L, L), device=device).float()

        # Fix first spin
        if fix_first:
            samples[:, :, 0, 0] = 0

        # Temperature tensor
        T_tensor = torch.full((batch_size,), T_val, device=device)

        # Thermalization
        if verbose:
            print(f"  Thermalizing ({n_thermalize} sweeps)...")
        for _ in tqdm(range(n_thermalize), desc="  Thermalize", leave=False, disable=not verbose):
            samples = mcmc_clock_update(
                samples, T_tensor, q=q,
                n_wolff_clusters=5, n_or_sweeps=2, n_mh_steps=0,
                fix_first=fix_first, J=J,
            )

        # Collection
        energies = []
        magnetizations = []
        helicities = []
        vortex_densities = []
        samples_collected = 0

        if verbose:
            print(f"  Collecting {n_samples} samples...")

        pbar = tqdm(total=n_samples, desc="  Collect", leave=False, disable=not verbose)
        while samples_collected < n_samples:
            # Decorrelate
            for _ in range(n_decorrelate):
                samples = mcmc_clock_update(
                    samples, T_tensor, q=q,
                    n_wolff_clusters=3, n_or_sweeps=1, n_mh_steps=0,
                    fix_first=fix_first, J=J,
                )

            # Compute observables
            energy = energy_fn(samples)  # (B, 1)
            mag = compute_clock_magnetization(samples, q)
            helicity = compute_helicity_modulus(samples, T_tensor, q, J=J)
            vortex = compute_vortex_density(samples, q)

            energies.append(energy.squeeze(-1).cpu())
            magnetizations.append(mag)
            helicities.append(helicity)
            vortex_densities.append(vortex)

            samples_collected += batch_size
            pbar.update(batch_size)

        pbar.close()

        # Stack energy samples for statistics
        all_energies = torch.cat(energies, dim=0)[:n_samples]
        N = L * L

        # Compute statistics
        reference_data[T_val] = {
            'energy_mean': all_energies.mean().item() / N,  # energy per site
            'energy_std': all_energies.std().item() / N,
            'mag_mean': np.mean(magnetizations),
            'mag_std': np.std(magnetizations),
            'helicity_mean': np.mean(helicities),
            'helicity_std': np.std(helicities),
            'vortex_mean': np.mean(vortex_densities),
            'vortex_std': np.std(vortex_densities),
            'n_samples': min(len(all_energies), n_samples),
        }

        if verbose:
            d = reference_data[T_val]
            print(f"  E/N = {d['energy_mean']:.4f} ± {d['energy_std']:.4f}")
            print(f"  M   = {d['mag_mean']:.4f} ± {d['mag_std']:.4f}")
            print(f"  Υ   = {d['helicity_mean']:.4f} ± {d['helicity_std']:.4f}")
            print(f"  ρ_v = {d['vortex_mean']:.4f} ± {d['vortex_std']:.4f}")

    return reference_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate MCMC reference data for q-state clock model validation"
    )
    parser.add_argument("--L", type=int, default=16, help="Lattice size")
    parser.add_argument("--q", type=int, default=36, help="Number of clock states")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--n_samples", type=int, default=50000, help="Samples per temperature")
    parser.add_argument("--n_thermalize", type=int, default=200, help="Thermalization sweeps")
    parser.add_argument("--n_decorrelate", type=int, default=5, help="Decorrelation sweeps")
    parser.add_argument("--batch_size", type=int, default=256, help="Parallel chains")
    parser.add_argument("--J", type=float, default=1.0, help="Coupling constant")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_fix_first", action="store_true", help="Don't fix first spin")

    args = parser.parse_args()

    # Temperature grid: dense around BKT transition
    # 6 points in [0.3, 0.6], 35 in [0.6, 1.2], 9 in [1.2, 2.5] = 50 total
    T_BKT = CLOCK_TC.get(args.q, 0.89)
    temps_low = np.linspace(0.3, 0.6, 6, endpoint=False).tolist()
    temps_mid = np.linspace(0.6, 1.2, 35, endpoint=False).tolist()
    temps_high = np.linspace(1.2, 2.5, 9).tolist()
    temperatures = temps_low + temps_mid + temps_high

    print("=" * 60)
    print(f"Clock Model (q={args.q}) MCMC Reference Data Generator")
    print("=" * 60)
    print(f"Lattice size: {args.L}×{args.L}")
    print(f"Clock states: q = {args.q}")
    print(f"Temperatures: {len(temperatures)} values in [{temperatures[0]:.2f}, {temperatures[-1]:.2f}]")
    print(f"BKT transition: T_BKT ≈ {T_BKT:.3f}")
    print(f"Samples per T: {args.n_samples}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Generate data
    reference_data = generate_reference_data(
        L=args.L,
        q=args.q,
        temperatures=temperatures,
        n_samples=args.n_samples,
        n_thermalize=args.n_thermalize,
        n_decorrelate=args.n_decorrelate,
        batch_size=args.batch_size,
        J=args.J,
        fix_first=not args.no_fix_first,
        device=args.device,
        verbose=True,
    )

    # Add metadata
    reference_data['_metadata'] = {
        'L': args.L,
        'q': args.q,
        'J': args.J,
        'n_samples': args.n_samples,
        'n_thermalize': args.n_thermalize,
        'n_decorrelate': args.n_decorrelate,
        'fix_first': not args.no_fix_first,
        'temperatures': temperatures,
        'T_BKT': T_BKT,
    }

    # Save
    if args.output is None:
        args.output = f"refs/clock{args.q}_L{args.L}.pt"

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    torch.save(reference_data, args.output)

    print("\n" + "=" * 60)
    print(f"Reference data saved to: {args.output}")
    print("=" * 60)

    # Summary table
    print(f"\n{'T':>8} {'E/N':>10} {'M':>10} {'Υ':>10} {'ρ_v':>10}")
    print("-" * 52)
    for T_val in temperatures:
        d = reference_data[T_val]
        print(f"{T_val:>8.4f} {d['energy_mean']:>10.4f} {d['mag_mean']:>10.4f} "
              f"{d['helicity_mean']:>10.4f} {d['vortex_mean']:>10.4f}")
    print("-" * 52)


if __name__ == "__main__":
    main()
