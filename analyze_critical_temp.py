#!/usr/bin/env python3
"""
Analyze model performance around critical temperature Tc = 2.269

Usage:
    python analyze_critical_temp.py --project MyProject --group my_group --seed 42
    python analyze_critical_temp.py --project MyProject --group my_group --seed 42 --device cuda:0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from util import load_model
from main import create_ising_energy_fn
from vatd_exact_partition import logZ as exact_logZ, CRITICAL_TEMPERATURE


def analyze_around_critical(
    model, energy_fn, L, device="cpu", num_temps=20, batch_size=500
):
    """
    Evaluate model at temperatures around critical point.

    Returns:
        DataFrame with columns: T, beta, loss, exact_logz, error, normalized_error
    """
    # Temperature range: 0.7*Tc to 1.3*Tc
    T_min = 0.7 * CRITICAL_TEMPERATURE
    T_max = 1.3 * CRITICAL_TEMPERATURE
    T_values = np.linspace(T_min, T_max, num_temps)

    results = []

    model.eval()
    with torch.no_grad():
        for T in T_values:
            beta = 1.0 / T
            T_tensor = torch.tensor([T], device=device)

            # Sample from model
            samples = model.sample(batch_size=batch_size, T=T_tensor)
            log_prob = model.log_prob(samples, T=T_tensor)
            energy = energy_fn(samples)

            # Compute loss
            beta_tensor = torch.tensor([beta], device=device).unsqueeze(-1)
            num_pixels = L * L
            loss_raw = (log_prob + beta_tensor * energy) / num_pixels
            loss = loss_raw.mean().item()

            # Compute exact
            exact_logz = (
                exact_logZ(n=L, j=1.0, beta=torch.tensor(beta)).item() / num_pixels
            )
            exact_loss = -exact_logz

            # Error
            error = loss - exact_loss
            normalized_error = error  # Already normalized by L*L

            results.append(
                {
                    "T": T,
                    "beta": beta,
                    "T/Tc": T / CRITICAL_TEMPERATURE,
                    "loss": loss,
                    "exact_logz": exact_logz,
                    "exact_loss": exact_loss,
                    "error": error,
                    "normalized_error": normalized_error,
                    "abs_error": abs(error),
                }
            )

    import pandas as pd

    return pd.DataFrame(results)


def plot_critical_analysis(df, output_dir="plots"):
    """Generate plots for critical temperature analysis"""
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss comparison
    ax = axes[0, 0]
    ax.plot(df["T"], df["loss"], "o-", label="Model Loss", markersize=4)
    ax.plot(df["T"], df["exact_loss"], "s-", label="Exact Loss", markersize=4)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Loss: Model vs Exact")

    # Plot 2: Error vs Temperature
    ax = axes[0, 1]
    ax.plot(df["T"], df["error"], "o-", markersize=4)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Error (Model - Exact)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Error vs Temperature")

    # Plot 3: Absolute error
    ax = axes[1, 0]
    ax.semilogy(df["T"], df["abs_error"], "o-", markersize=4)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("|Error| (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Absolute Error (log scale)")

    # Plot 4: Error vs T/Tc
    ax = axes[1, 1]
    ax.plot(df["T/Tc"], df["error"], "o-", markersize=4)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(1.0, color="r", linestyle="--", alpha=0.5, label="T/Tc = 1")
    ax.set_xlabel("T / Tc")
    ax.set_ylabel("Error (Model - Exact)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Error vs Reduced Temperature")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/critical_temp_analysis.png", dpi=150)
    print(f"Saved plot to {output_dir}/critical_temp_analysis.png")

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--seed", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_temps", type=int, default=20)
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.project}/{args.group}/{args.seed}")
    model, config = load_model(args.project, args.group, args.seed)
    model = model.to(args.device)
    model.eval()

    # Get lattice size
    L = model.size[0]
    print(f"Lattice size: {L}×{L}")
    print(f"Critical temperature: {CRITICAL_TEMPERATURE:.4f}")

    # Create energy function
    energy_fn = create_ising_energy_fn(L=L, d=2, device=args.device)

    # Analyze
    print(f"\nAnalyzing {args.num_temps} temperatures around Tc...")
    df = analyze_around_critical(
        model,
        energy_fn,
        L,
        device=args.device,
        num_temps=args.num_temps,
        batch_size=args.batch_size,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("CRITICAL TEMPERATURE ANALYSIS")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    # Find best and worst temperatures
    idx_min_error = df["abs_error"].idxmin()
    idx_max_error = df["abs_error"].idxmax()

    print(f"\nBest accuracy:")
    print(
        f"  T = {df.loc[idx_min_error, 'T']:.4f} (T/Tc = {df.loc[idx_min_error, 'T/Tc']:.3f})"
    )
    print(f"  |Error| = {df.loc[idx_min_error, 'abs_error']:.6f}")

    print(f"\nWorst accuracy:")
    print(
        f"  T = {df.loc[idx_max_error, 'T']:.4f} (T/Tc = {df.loc[idx_max_error, 'T/Tc']:.3f})"
    )
    print(f"  |Error| = {df.loc[idx_max_error, 'abs_error']:.6f}")

    # Error at Tc
    idx_critical = (df["T"] - CRITICAL_TEMPERATURE).abs().idxmin()
    print(f"\nAt critical temperature Tc = {CRITICAL_TEMPERATURE:.4f}:")
    print(f"  T = {df.loc[idx_critical, 'T']:.4f}")
    print(f"  Error = {df.loc[idx_critical, 'error']:.6f}")
    print(f"  |Error| = {df.loc[idx_critical, 'abs_error']:.6f}")

    # Generate plots
    print("\nGenerating plots...")
    plot_critical_analysis(df)

    # Save results
    output_file = f"runs/{args.project}/{args.group}/critical_temp_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    main()
