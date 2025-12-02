#!/usr/bin/env python3
"""
Compare trained model with reference implementation at reference temperatures.

Usage:
    python compare_with_reference.py --project MyProject --group my_group --seed 42
    python compare_with_reference.py --project MyProject --group my_group --seed 42 --device cuda:0
"""

import torch
import numpy as np
import pandas as pd
import argparse

from util import load_model
from main import create_ising_energy_fn
from vatd_exact_partition import logZ as exact_logZ, CRITICAL_TEMPERATURE

# Reference temperature configuration (from ~/zbin/vatd/isingPixelCNN.py)
T0_REFERENCE = 2.269  # Critical temperature used in reference
FACTOR_LIST = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.8]


def evaluate_at_reference_temps(model, energy_fn, L, device="cpu", batch_size=500):
    """Evaluate model at reference implementation temperatures"""

    T_list = [T0_REFERENCE * factor for factor in FACTOR_LIST]
    results = []

    model.eval()
    with torch.no_grad():
        for factor, T in zip(FACTOR_LIST, T_list):
            beta = 1.0 / T
            T_tensor = torch.tensor([T], device=device)

            # Sample and compute loss
            samples = model.sample(batch_size=batch_size, T=T_tensor)
            log_prob = model.log_prob(samples, T=T_tensor)
            energy = energy_fn(samples)

            beta_tensor = torch.tensor([beta], device=device).unsqueeze(-1)
            loss_raw = log_prob + beta_tensor * energy
            loss = loss_raw.mean().item()

            # Compute exact
            exact_logz = exact_logZ(n=L, j=1.0, beta=torch.tensor(beta)).item()
            exact_loss = -exact_logz

            # Normalized loss (like reference: loss / (L*L*0.45))
            normalized_loss = loss / (L * L * 0.45)

            # Error
            error = loss - exact_loss

            results.append(
                {
                    "factor": factor,
                    "T": T,
                    "beta": beta,
                    "loss": loss,
                    "exact_logz": exact_logz,
                    "exact_loss": exact_loss,
                    "error": error,
                    "abs_error": abs(error),
                    "normalized_loss": normalized_loss,
                }
            )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--seed", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=500)
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.project}/{args.group}/{args.seed}")
    model, config = load_model(args.project, args.group, args.seed)
    model = model.to(args.device)

    L = model.size[0]
    print(f"Lattice size: {L}×{L}")

    # Create energy function
    energy_fn = create_ising_energy_fn(L=L, d=2, device=args.device)

    # Evaluate
    print("\nEvaluating at reference temperatures...")
    df = evaluate_at_reference_temps(model, energy_fn, L, args.device, args.batch_size)

    # Print comparison table (matching reference format)
    print("\n" + "=" * 80)
    print("COMPARISON WITH REFERENCE IMPLEMENTATION")
    print("=" * 80)
    print(f"Reference T0 = {T0_REFERENCE}")
    print(f"Factor list: {FACTOR_LIST}")
    print("=" * 80)

    print("\nResults:")
    print(df[["factor", "T", "normalized_loss", "error"]].to_string(index=False))

    print("\n" + "=" * 80)
    print(f"Mean absolute error: {df['abs_error'].mean():.6f}")
    print(f"Max absolute error: {df['abs_error'].max():.6f}")
    print(f"Error at Tc (factor=1.0): {df[df['factor']==1.0]['error'].values[0]:.6f}")
    print("=" * 80)

    # Save
    output_file = f"runs/{args.project}/{args.group}/reference_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
