"""
Rank-Exponent Duality Analysis (Result 3).

Fits a power law to the effective rank near the critical temperature:

    eRank(T) - eRank_min ~ A * |T - Tc|^{-φ} / Tc^{-φ}

in the reduced temperature variable t = |T - Tc| / Tc.

The exponent φ may represent a new universal exponent characterizing
how the neural network's internal representation complexity diverges
at the critical point.

Usage:
    python analyze_rank_exponent.py --csv <rank_analysis_critical.csv> [--q 2]
    python analyze_rank_exponent.py --csv_ising <path> --csv_potts3 <path>
"""

import os
os.environ['VATD_NO_MHC'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from scipy.optimize import curve_fit
from rich.console import Console
import argparse

from potts_exact_partition import POTTS_TC


# ──────────────────────────────────────────────────────────────
# Power Law Fitting
# ──────────────────────────────────────────────────────────────


def power_law(t_reduced, phi, A):
    """Power law: A * |t|^{-φ}"""
    return A * np.power(t_reduced, -phi)


def fit_rank_exponent(df, Tc, side="both", t_min=0.05, t_max=0.8):
    """
    Fit eRank vs reduced temperature to a power law.

    Args:
        df: DataFrame with columns T, channel_erank (layer-averaged)
        Tc: critical temperature
        side: "above" (T > Tc), "below" (T < Tc), or "both"
        t_min: minimum |T-Tc|/Tc to include (avoid Tc divergence)
        t_max: maximum |T-Tc|/Tc to include (stay near critical region)

    Returns:
        dict with: phi, phi_err, A, R2, t_data, erank_data, side
    """
    avg_erank = df.groupby("T")["channel_erank"].mean().sort_index()
    temperatures = avg_erank.index.values
    erank_values = avg_erank.values

    # Find eRank minimum
    erank_min = erank_values.min()

    # Compute reduced temperature
    t_reduced = np.abs(temperatures - Tc) / Tc

    # Filter by side and range
    if side == "above":
        mask = (temperatures > Tc) & (t_reduced >= t_min) & (t_reduced <= t_max)
    elif side == "below":
        mask = (temperatures < Tc) & (t_reduced >= t_min) & (t_reduced <= t_max)
    else:
        mask = (t_reduced >= t_min) & (t_reduced <= t_max)

    t_data = t_reduced[mask]
    erank_data = erank_values[mask] - erank_min  # Shift to zero at minimum

    if len(t_data) < 4:
        return None

    # Remove non-positive values (can't fit power law to zero/negative)
    pos_mask = erank_data > 0
    t_data = t_data[pos_mask]
    erank_data = erank_data[pos_mask]

    if len(t_data) < 4:
        return None

    try:
        popt, pcov = curve_fit(
            power_law, t_data, erank_data,
            p0=[0.5, erank_data.max()],
            bounds=([0.01, 0.01], [5.0, 1000.0]),
            maxfev=10000,
        )
        phi, A = popt
        phi_err = np.sqrt(np.diag(pcov))[0]

        # R²
        y_pred = power_law(t_data, phi, A)
        ss_res = np.sum((erank_data - y_pred) ** 2)
        ss_tot = np.sum((erank_data - erank_data.mean()) ** 2)
        R2 = 1 - ss_res / (ss_tot + 1e-15)

        return {
            "phi": phi,
            "phi_err": phi_err,
            "A": A,
            "R2": R2,
            "t_data": t_data,
            "erank_data": erank_data,
            "erank_min": erank_min,
            "side": side,
        }
    except (RuntimeError, ValueError) as e:
        print(f"  Fit failed ({side}): {e}")
        return None


def bootstrap_exponent(df, Tc, side="both", n_bootstrap=1000, t_min=0.05, t_max=0.8):
    """
    Bootstrap confidence interval for the power-law exponent φ.

    Args:
        df: DataFrame
        Tc: critical temperature
        side: fitting side
        n_bootstrap: number of bootstrap samples
        t_min, t_max: reduced temperature range

    Returns:
        dict with: phi_median, phi_ci_low, phi_ci_high, phi_std, all_phis
    """
    avg_erank = df.groupby("T")["channel_erank"].mean().sort_index()
    temperatures = avg_erank.index.values
    erank_values = avg_erank.values
    erank_min = erank_values.min()

    t_reduced = np.abs(temperatures - Tc) / Tc

    if side == "above":
        mask = (temperatures > Tc) & (t_reduced >= t_min) & (t_reduced <= t_max)
    elif side == "below":
        mask = (temperatures < Tc) & (t_reduced >= t_min) & (t_reduced <= t_max)
    else:
        mask = (t_reduced >= t_min) & (t_reduced <= t_max)

    t_data = t_reduced[mask]
    erank_data = erank_values[mask] - erank_min

    pos_mask = erank_data > 0
    t_data = t_data[pos_mask]
    erank_data = erank_data[pos_mask]

    if len(t_data) < 4:
        return None

    all_phis = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(t_data), size=len(t_data), replace=True)
        t_boot = t_data[indices]
        e_boot = erank_data[indices]

        try:
            popt, _ = curve_fit(
                power_law, t_boot, e_boot,
                p0=[0.5, erank_data.max()],
                bounds=([0.01, 0.01], [5.0, 1000.0]),
                maxfev=5000,
            )
            all_phis.append(popt[0])
        except (RuntimeError, ValueError):
            continue

    if len(all_phis) < 100:
        return None

    all_phis = np.array(all_phis)
    return {
        "phi_median": np.median(all_phis),
        "phi_ci_low": np.percentile(all_phis, 2.5),
        "phi_ci_high": np.percentile(all_phis, 97.5),
        "phi_std": np.std(all_phis),
        "all_phis": all_phis,
        "n_valid": len(all_phis),
    }


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────


MODEL_COLORS = {2: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c"}
MODEL_LABELS = {2: "Ising ($q=2$)", 3: "3-Potts ($q=3$)", 4: "4-Potts ($q=4$)"}


def plot_rank_exponent(results_dict, figs_dir):
    """
    Log-log plot of eRank - eRank_min vs |T-Tc|/Tc with power-law fits.

    Args:
        results_dict: {q: {side: fit_result, ...}}
        figs_dir: output directory
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for q, side_results in sorted(results_dict.items()):
        color = MODEL_COLORS[q]
        label = MODEL_LABELS[q]

        for side, result in side_results.items():
            if result is None:
                continue

            marker = "o" if side in ("both", "above") else "s"
            alpha = 0.8 if side in ("both", "above") else 0.5
            side_label = f"{label}" if side == "both" else f"{label} ({side})"

            # Data points
            ax.loglog(
                result["t_data"], result["erank_data"],
                marker, color=color, markersize=4, alpha=alpha,
                label=side_label,
            )

            # Fit line
            t_fit = np.logspace(
                np.log10(result["t_data"].min()),
                np.log10(result["t_data"].max()),
                100,
            )
            y_fit = power_law(t_fit, result["phi"], result["A"])
            ax.loglog(
                t_fit, y_fit, "-", color=color, alpha=0.6, linewidth=1.5,
                label=f"$\\phi = {result['phi']:.2f} \\pm {result['phi_err']:.2f}$"
                      f" ($R^2 = {result['R2']:.3f}$)",
            )

    ax.set_xlabel("$|T - T_c| / T_c$", fontsize=13)
    ax.set_ylabel("$\\mathrm{eRank}(T) - \\mathrm{eRank}_{\\mathrm{min}}$", fontsize=13)
    ax.set_title("Rank-Exponent Duality: Power-Law Scaling near $T_c$",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2, which="both")

    plt.tight_layout()
    path = Path(figs_dir) / "rank_exponent_duality.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bootstrap_distribution(bootstrap_results, figs_dir):
    """Plot bootstrap distribution of φ for each model."""
    fig, axes = plt.subplots(1, len(bootstrap_results), figsize=(4 * len(bootstrap_results), 4))
    if len(bootstrap_results) == 1:
        axes = [axes]

    for ax, (q, boot) in zip(axes, sorted(bootstrap_results.items())):
        if boot is None:
            ax.text(0.5, 0.5, "Fit failed", transform=ax.transAxes, ha="center")
            continue
        ax.hist(
            boot["all_phis"], bins=40, density=True,
            color=MODEL_COLORS[q], alpha=0.7, edgecolor="black", linewidth=0.5,
        )
        ax.axvline(boot["phi_median"], color="red", ls="--", lw=1.5,
                    label=f"$\\phi = {boot['phi_median']:.3f}$")
        ax.axvline(boot["phi_ci_low"], color="red", ls=":", alpha=0.5)
        ax.axvline(boot["phi_ci_high"], color="red", ls=":", alpha=0.5)
        ax.set_xlabel("$\\phi$", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{MODEL_LABELS[q]}: $\\phi = {boot['phi_median']:.3f}$\n"
                     f"95% CI: [{boot['phi_ci_low']:.3f}, {boot['phi_ci_high']:.3f}]",
                     fontsize=10)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = Path(figs_dir) / "rank_exponent_bootstrap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Rank-Exponent Duality Analysis"
    )
    parser.add_argument("--csv", type=str, help="Single model CSV path")
    parser.add_argument("--q", type=int, default=2, help="Number of states (for --csv)")
    parser.add_argument("--csv_ising", type=str, help="Ising rank analysis CSV (critical mode)")
    parser.add_argument("--csv_potts3", type=str, help="3-Potts rank analysis CSV (critical mode)")
    parser.add_argument("--csv_potts4", type=str, help="4-Potts rank analysis CSV (critical mode)")
    parser.add_argument("--t_min", type=float, default=0.05,
                        help="Min reduced temperature for fitting (default: 0.05)")
    parser.add_argument("--t_max", type=float, default=0.8,
                        help="Max reduced temperature for fitting (default: 0.8)")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Run bootstrap confidence intervals")
    parser.add_argument("--figs_dir", type=str, default="figs/rank_exponent",
                        help="Output directory for figures")
    args = parser.parse_args()

    console = Console()
    console.print("[bold green]Rank-Exponent Duality Analysis (Result 3)[/bold green]")

    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Collect CSV paths
    csv_map = {}
    if args.csv:
        csv_map[args.q] = args.csv
    if args.csv_ising:
        csv_map[2] = args.csv_ising
    if args.csv_potts3:
        csv_map[3] = args.csv_potts3
    if args.csv_potts4:
        csv_map[4] = args.csv_potts4

    if not csv_map:
        console.print("[red]Please specify at least one CSV file.[/red]")
        return

    all_results = {}
    bootstrap_results = {}

    for q, csv_path in sorted(csv_map.items()):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            console.print(f"[red]CSV not found:[/red] {csv_path}")
            continue

        Tc = POTTS_TC[q]
        console.print(f"\n[bold]{MODEL_LABELS[q]}:[/bold] Tc = {Tc:.4f}")

        df = pd.read_csv(csv_path)
        console.print(f"  Loaded {len(df)} rows, {df['T'].nunique()} temperatures")

        # Fit both sides separately and combined
        side_results = {}
        for side in ["both", "above", "below"]:
            result = fit_rank_exponent(df, Tc, side=side, t_min=args.t_min, t_max=args.t_max)
            side_results[side] = result
            if result:
                console.print(
                    f"  {side:>5s}: φ = {result['phi']:.3f} ± {result['phi_err']:.3f}, "
                    f"R² = {result['R2']:.3f}, A = {result['A']:.2f}, "
                    f"N_pts = {len(result['t_data'])}"
                )
            else:
                console.print(f"  {side:>5s}: insufficient data or fit failed")

        all_results[q] = side_results

        # Bootstrap
        if args.bootstrap:
            console.print(f"  Running bootstrap (1000 samples)...")
            boot = bootstrap_exponent(df, Tc, side="both", t_min=args.t_min, t_max=args.t_max)
            bootstrap_results[q] = boot
            if boot:
                console.print(
                    f"  Bootstrap: φ = {boot['phi_median']:.3f} "
                    f"[{boot['phi_ci_low']:.3f}, {boot['phi_ci_high']:.3f}] (95% CI), "
                    f"σ = {boot['phi_std']:.3f}"
                )

    # Generate plots
    console.print("\n[bold cyan]Generating plots[/bold cyan]")

    fig_path = plot_rank_exponent(all_results, figs_dir)
    console.print(f"[green]Rank-exponent plot:[/green] {fig_path}")

    if bootstrap_results:
        fig_path = plot_bootstrap_distribution(bootstrap_results, figs_dir)
        console.print(f"[green]Bootstrap distributions:[/green] {fig_path}")

    # Universality check
    if len(all_results) >= 2:
        console.print("\n[bold cyan]Universality Check[/bold cyan]")
        console.print(f"{'Model':>12s}  {'φ (both)':>12s}  {'φ (above)':>12s}  {'φ (below)':>12s}")
        console.print("-" * 55)
        for q in sorted(all_results.keys()):
            vals = []
            for side in ["both", "above", "below"]:
                r = all_results[q].get(side)
                if r:
                    vals.append(f"{r['phi']:.3f}±{r['phi_err']:.3f}")
                else:
                    vals.append("N/A")
            console.print(f"{MODEL_LABELS[q]:>12s}  {vals[0]:>12s}  {vals[1]:>12s}  {vals[2]:>12s}")

    console.print(f"\n[bold green]Analysis complete.[/bold green]")


if __name__ == "__main__":
    main()
