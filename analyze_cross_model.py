"""
Cross-model comparison of effective rank at criticality (Result 1).

Loads rank analysis CSV files from multiple models (Ising, 3-Potts, 4-Potts)
and produces:
  - Fig 1a: T/Tc normalized eRank(T) overlay for all models
  - Fig 1b: Bar chart of measured eRank_min vs CFT operator count predictions
  - Summary table

This is the core analysis for "Autoregressive neural networks discover
the operator content of CFTs" — showing that eRank_min counts the number
of relevant scaling operators at the critical point.

Usage:
    python analyze_cross_model.py --csv_ising <path> --csv_potts3 <path> [--csv_potts4 <path>]
    python analyze_cross_model.py  # Interactive mode
"""

import os
os.environ['VATD_NO_MHC'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from rich.console import Console
import argparse

from potts_exact_partition import (
    POTTS_TC,
    CENTRAL_CHARGES,
    RELEVANT_OPERATORS,
)


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────


def load_and_extract_erank_min(csv_path, Tc, layer_filter=None):
    """
    Load rank analysis CSV and extract eRank minimum near Tc.

    Args:
        csv_path: Path to rank_analysis CSV
        Tc: Critical temperature for this model
        layer_filter: Optional layer name to restrict to (e.g. "layer_0")

    Returns:
        dict with keys: erank_min, T_at_min, erank_values, temperatures
    """
    df = pd.read_csv(csv_path)

    # Average eRank across layers (unless filtered)
    if layer_filter:
        df = df[df["layer"] == layer_filter]

    avg_erank = df.groupby("T")["channel_erank"].mean().sort_index()
    temperatures = avg_erank.index.values
    erank_values = avg_erank.values

    # Find minimum eRank
    min_idx = np.argmin(erank_values)
    erank_min = erank_values[min_idx]
    T_at_min = temperatures[min_idx]

    # Also compute eRank at nearest T to Tc
    Tc_idx = np.argmin(np.abs(temperatures - Tc))
    erank_at_Tc = erank_values[Tc_idx]
    T_nearest_Tc = temperatures[Tc_idx]

    return {
        "erank_min": erank_min,
        "T_at_min": T_at_min,
        "erank_at_Tc": erank_at_Tc,
        "T_nearest_Tc": T_nearest_Tc,
        "temperatures": temperatures,
        "erank_values": erank_values,
    }


def load_per_layer_erank_min(csv_path, Tc):
    """
    Extract eRank minimum for each layer separately.

    Returns dict: {layer_name: {erank_min, T_at_min, ...}}
    """
    df = pd.read_csv(csv_path)
    results = {}
    for layer in sorted(df["layer"].unique()):
        ld = df[df["layer"] == layer].sort_values("T")
        erank_values = ld["channel_erank"].values
        temperatures = ld["T"].values
        min_idx = np.argmin(erank_values)

        results[layer] = {
            "erank_min": erank_values[min_idx],
            "T_at_min": temperatures[min_idx],
        }
    return results


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────


MODEL_COLORS = {
    2: "#1f77b4",   # Blue for Ising
    3: "#ff7f0e",   # Orange for 3-Potts
    4: "#2ca02c",   # Green for 4-Potts
    12: "#d62728",  # Red for 12-Clock
    36: "#9467bd",  # Purple for 36-Clock
}
MODEL_LABELS = {
    2: "Ising ($q=2$)",
    3: "3-Potts ($q=3$)",
    4: "4-Potts ($q=4$)",
    12: "12-Clock ($q=12$)",
    36: "36-Clock ($q=36$)",
}


def plot_cross_model_comparison(results, figs_dir):
    """
    Main publication figure: 2-panel comparison.

    Args:
        results: dict {q: {erank_min, ..., temperatures, erank_values}}
        figs_dir: output directory

    Fig 1a: T/Tc normalized eRank overlay
    Fig 1b: Bar chart eRank_min vs predicted operator count
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel (a): eRank vs T/Tc ──
    for q, data in sorted(results.items()):
        Tc = POTTS_TC[q]
        T_over_Tc = data["temperatures"] / Tc
        ax1.plot(
            T_over_Tc, data["erank_values"],
            "o-", color=MODEL_COLORS[q], markersize=3, linewidth=1.5,
            label=MODEL_LABELS[q], alpha=0.85,
        )
        # Mark the minimum
        T_min_over_Tc = data["T_at_min"] / Tc
        ax1.axvline(T_min_over_Tc, color=MODEL_COLORS[q], ls=":", alpha=0.3, lw=0.8)

    ax1.axvline(1.0, color="red", ls="--", alpha=0.5, lw=1.0, label="$T/T_c = 1$")
    ax1.set_xlabel("$T / T_c$", fontsize=12)
    ax1.set_ylabel("Effective Rank (layer-averaged)", fontsize=11)
    ax1.set_title("(a) eRank vs normalized temperature", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.15)
    ax1.set_xlim(0.3, 3.0)

    # ── Panel (b): Bar chart ──
    q_values = sorted(results.keys())
    x = np.arange(len(q_values))
    width = 0.35

    # Measured eRank_min
    measured = [results[q]["erank_min"] for q in q_values]
    # Predicted operator counts
    predicted = [RELEVANT_OPERATORS[q] for q in q_values]

    bars_measured = ax2.bar(
        x - width / 2, measured, width,
        label="Measured eRank$_{\\rm min}$",
        color=[MODEL_COLORS[q] for q in q_values], alpha=0.8,
        edgecolor="black", linewidth=0.5,
    )
    bars_predicted = ax2.bar(
        x + width / 2, predicted, width,
        label="CFT predicted $\\#$ operators",
        color="lightgray", alpha=0.8,
        edgecolor="black", linewidth=0.5,
        hatch="//",
    )

    # Add value labels on bars
    for bar in bars_measured:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)
    for bar in bars_predicted:
        h = bar.get_height()
        ax2.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("Count / Rank", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_LABELS[q] for q in q_values])
    ax2.set_title("(b) eRank$_{\\rm min}$ vs CFT operator count", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.15, axis="y")

    plt.tight_layout()
    path = Path(figs_dir) / "cross_model_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_layer_comparison(per_layer_results, figs_dir):
    """
    Supplementary figure: eRank_min per layer for each model.

    Shows which layers track operator content most closely.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    q_values = sorted(per_layer_results.keys())
    all_layers = set()
    for q in q_values:
        all_layers.update(per_layer_results[q].keys())
    layers = sorted(all_layers, key=lambda l: (0, int(l.split("_")[1])) if l.startswith("layer_") else (1, 0))

    x = np.arange(len(layers))
    width = 0.8 / len(q_values)

    for i, q in enumerate(q_values):
        erank_mins = [per_layer_results[q].get(l, {}).get("erank_min", np.nan) for l in layers]
        offset = (i - len(q_values) / 2 + 0.5) * width
        ax.bar(
            x + offset, erank_mins, width,
            label=MODEL_LABELS[q], color=MODEL_COLORS[q], alpha=0.8,
            edgecolor="black", linewidth=0.5,
        )
        # Horizontal line for predicted operator count
        ax.axhline(RELEVANT_OPERATORS[q], color=MODEL_COLORS[q], ls="--", alpha=0.4, lw=1)

    layer_labels = [l.replace("layer_", "Block ").replace("final", "Final") for l in layers]
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=30, ha="right")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("eRank$_{\\rm min}$", fontsize=11)
    ax.set_title("eRank minimum per layer across models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15, axis="y")

    plt.tight_layout()
    path = Path(figs_dir) / "cross_model_per_layer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def print_summary_table(results, console):
    """Print summary comparison table."""
    console.print("\n[bold cyan]Cross-Model Comparison Summary[/bold cyan]")
    console.print(
        f"{'Model':>12s}  {'q':>3s}  {'c':>5s}  {'#ops (pred)':>11s}  "
        f"{'eRank_min':>10s}  {'T_min':>7s}  {'T_min/Tc':>8s}  {'Ratio':>8s}"
    )
    console.print("-" * 80)

    for q in sorted(results.keys()):
        data = results[q]
        Tc = POTTS_TC[q]
        c = CENTRAL_CHARGES[q]
        n_ops = RELEVANT_OPERATORS[q]
        ratio = data["erank_min"] / n_ops

        console.print(
            f"{MODEL_LABELS[q]:>12s}  {q:>3d}  {c:>5.1f}  {n_ops:>11d}  "
            f"{data['erank_min']:>10.2f}  {data['T_at_min']:>7.3f}  "
            f"{data['T_at_min']/Tc:>8.3f}  {ratio:>8.3f}"
        )

    # Monotonicity check
    q_values = sorted(results.keys())
    erank_mins = [results[q]["erank_min"] for q in q_values]
    is_monotone = all(erank_mins[i] <= erank_mins[i + 1] for i in range(len(erank_mins) - 1))
    console.print(f"\nMonotonicity (q↑ → eRank_min↑): {'[green]PASS[/green]' if is_monotone else '[red]FAIL[/red]'}")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model comparison of eRank at criticality"
    )
    parser.add_argument("--csv_ising", type=str, help="Path to Ising rank analysis CSV")
    parser.add_argument("--csv_potts3", type=str, help="Path to 3-Potts rank analysis CSV")
    parser.add_argument("--csv_potts4", type=str, help="Path to 4-Potts rank analysis CSV")
    parser.add_argument("--figs_dir", type=str, default="figs/cross_model",
                        help="Output directory for figures")
    args = parser.parse_args()

    console = Console()
    console.print("[bold green]Cross-Model eRank Comparison (Result 1)[/bold green]")

    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    per_layer_results = {}

    # Load available models
    csv_map = {}
    if args.csv_ising:
        csv_map[2] = args.csv_ising
    if args.csv_potts3:
        csv_map[3] = args.csv_potts3
    if args.csv_potts4:
        csv_map[4] = args.csv_potts4

    if not csv_map:
        # Interactive: try to find CSVs in runs/
        console.print("[yellow]No CSVs specified. Searching runs/ for rank_analysis files...[/yellow]")
        from glob import glob
        csvs = glob("runs/*/*/rank_analysis*.csv")
        if csvs:
            for csv in csvs:
                console.print(f"  Found: {csv}")
            console.print("\nPlease specify CSV paths with --csv_ising, --csv_potts3, --csv_potts4")
        else:
            console.print("[red]No rank analysis CSVs found in runs/[/red]")
        return

    for q, csv_path in csv_map.items():
        csv_path = Path(csv_path)
        if not csv_path.exists():
            console.print(f"[red]CSV not found:[/red] {csv_path}")
            continue

        Tc = POTTS_TC[q]
        console.print(f"\nLoading q={q} ({MODEL_LABELS[q]}): {csv_path}")
        console.print(f"  Tc = {Tc:.4f}")

        results[q] = load_and_extract_erank_min(csv_path, Tc)
        per_layer_results[q] = load_per_layer_erank_min(csv_path, Tc)

        console.print(
            f"  eRank_min = {results[q]['erank_min']:.2f} "
            f"at T = {results[q]['T_at_min']:.3f} (T/Tc = {results[q]['T_at_min']/Tc:.3f})"
        )

    if len(results) < 2:
        console.print("[red]Need at least 2 models for comparison.[/red]")
        return

    # Generate plots
    console.print("\n[bold cyan]Generating comparison figures[/bold cyan]")

    fig_path = plot_cross_model_comparison(results, figs_dir)
    console.print(f"[green]Main comparison:[/green] {fig_path}")

    fig_path = plot_per_layer_comparison(per_layer_results, figs_dir)
    console.print(f"[green]Per-layer comparison:[/green] {fig_path}")

    # Summary
    print_summary_table(results, console)

    console.print(f"\n[bold green]Analysis complete.[/bold green]")
    console.print(f"Figures saved to: {figs_dir}/")


if __name__ == "__main__":
    main()
