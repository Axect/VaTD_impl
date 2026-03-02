"""
Unified PRL Paper Figure Generation.

Reads analysis CSV outputs from all experiments and generates
publication-quality figures for a 4-page PRL paper:

  Fig 1 (Result 1): Multi-model operator counting
    (a) T/Tc normalized eRank overlay: Ising, 3-Potts, 4-Potts
    (b) Bar chart: measured eRank_min vs CFT operator count

  Fig 2 (Result 2): Neural entanglement entropy
    (a) S_E(l) data + Calabrese-Cardy fit at Tc
    (b) Extracted central charge c across models/layers

  Fig 3 (Result 3, optional/supplemental): Rank-exponent duality
    Log-log plot: eRank - eRank_min vs |T-Tc|/Tc + power-law fit

Usage:
    python analyze_prl_paper.py \\
        --rank_ising <csv> --rank_potts3 <csv> \\
        --entanglement_ising <csv> \\
        [--rank_exponent_ising <csv>] \\
        [--output_dir figs/prl_paper]
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

from potts_exact_partition import (
    POTTS_TC,
    CENTRAL_CHARGES,
    RELEVANT_OPERATORS,
)
from analyze_cross_model import load_and_extract_erank_min, MODEL_COLORS, MODEL_LABELS
from analyze_entanglement import calabrese_cardy
from analyze_rank_exponent import power_law, fit_rank_exponent


# ──────────────────────────────────────────────────────────────
# PRL Style Settings
# ──────────────────────────────────────────────────────────────

PRL_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,  # Set True if LaTeX is available
}


def apply_prl_style():
    plt.rcParams.update(PRL_STYLE)


# ──────────────────────────────────────────────────────────────
# Figure 1: Multi-Model Operator Counting
# ──────────────────────────────────────────────────────────────


def generate_figure_1(rank_csvs, output_dir):
    """
    Fig 1: 2-panel comparison of eRank across models.

    Args:
        rank_csvs: dict {q: csv_path}
        output_dir: output path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))  # PRL column width

    results = {}
    for q, csv_path in sorted(rank_csvs.items()):
        Tc = POTTS_TC[q]
        results[q] = load_and_extract_erank_min(csv_path, Tc)

    # ── (a) eRank vs T/Tc ──
    for q, data in sorted(results.items()):
        Tc = POTTS_TC[q]
        T_over_Tc = data["temperatures"] / Tc
        ax1.plot(
            T_over_Tc, data["erank_values"],
            "o-", color=MODEL_COLORS[q], markersize=2, linewidth=1.2,
            label=MODEL_LABELS[q], alpha=0.85,
        )

    ax1.axvline(1.0, color="red", ls="--", alpha=0.5, lw=0.8)
    ax1.set_xlabel("$T / T_c$")
    ax1.set_ylabel("Effective rank (layer-avg)")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.15)
    ax1.set_xlim(0.3, 3.0)
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, fontsize=11,
             fontweight="bold", va="top")

    # ── (b) Bar chart ──
    q_values = sorted(results.keys())
    x = np.arange(len(q_values))
    width = 0.35

    measured = [results[q]["erank_min"] for q in q_values]
    predicted = [RELEVANT_OPERATORS[q] for q in q_values]

    ax2.bar(x - width / 2, measured, width,
            label="Measured eRank$_{\\rm min}$",
            color=[MODEL_COLORS[q] for q in q_values], alpha=0.8,
            edgecolor="black", linewidth=0.5)
    ax2.bar(x + width / 2, predicted, width,
            label="CFT $\\#$ operators",
            color="lightgray", alpha=0.8,
            edgecolor="black", linewidth=0.5, hatch="//")

    for i, (m, p) in enumerate(zip(measured, predicted)):
        ax2.annotate(f"{m:.1f}", xy=(x[i] - width / 2, m), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7)
        ax2.annotate(f"{int(p)}", xy=(x[i] + width / 2, p), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_LABELS[q] for q in q_values], fontsize=8)
    ax2.set_ylabel("Count / Rank")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.15, axis="y")
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontsize=11,
             fontweight="bold", va="top")

    plt.tight_layout()
    path = Path(output_dir) / "fig1_operator_counting.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# Figure 2: Neural Entanglement Entropy
# ──────────────────────────────────────────────────────────────


def generate_figure_2(entanglement_csvs, output_dir):
    """
    Fig 2: 2-panel entanglement entropy.

    Args:
        entanglement_csvs: dict {q: csv_path}
        output_dir: output path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

    all_c_fits = []

    for q, csv_path in sorted(entanglement_csvs.items()):
        df = pd.read_csv(csv_path)
        Tc = POTTS_TC[q]
        c_exact = CENTRAL_CHARGES.get(q)

        # ── (a) S_E(l) at Tc ──
        df_Tc = df[df["T_over_Tc"].between(0.95, 1.05)]
        if len(df_Tc) == 0:
            nearest_idx = (df["T_over_Tc"] - 1.0).abs().idxmin()
            T_nearest = df.loc[nearest_idx, "T"]
            df_Tc = df[df["T"] == T_nearest]

        # Use layer_0 for clearest signal
        for layer in sorted(df_Tc["layer"].unique())[:1]:
            ld = df_Tc[df_Tc["layer"] == layer].sort_values("l")
            L = ld["l"].max() * 2  # Infer L from max(l) = L/2
            ax1.plot(
                ld["l"], ld["S_E"],
                "o-", color=MODEL_COLORS[q], markersize=3, linewidth=1.0,
                label=f"{MODEL_LABELS[q]}", alpha=0.8,
            )

            # Fit and overlay CC curve
            c_fit = ld["c_fit"].iloc[0]
            if not np.isnan(c_fit):
                try:
                    popt, _ = curve_fit(
                        lambda l_f, c, const: calabrese_cardy(l_f, L, c, const),
                        ld["l"].values.astype(float), ld["S_E"].values,
                        p0=[c_fit, ld["S_E"].mean()],
                        bounds=([0.0, -100], [10.0, 100]),
                    )
                    l_smooth = np.linspace(0.5, L / 2, 100)
                    ax1.plot(l_smooth, calabrese_cardy(l_smooth, L, popt[0], popt[1]),
                             "--", color=MODEL_COLORS[q], alpha=0.5, linewidth=1.0)
                except Exception:
                    pass

        # ── Collect c_fits for panel (b) ──
        c_summary = df.groupby(["T_over_Tc", "layer"]).first()[["c_fit", "c_err", "R2"]].reset_index()
        c_summary["q"] = q
        all_c_fits.append(c_summary)

    ax1.set_xlabel("Subsystem size $\\ell$")
    ax1.set_ylabel("$S_E(\\ell)$")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.15)
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, fontsize=11,
             fontweight="bold", va="top")

    # ── (b) Extracted c values ──
    if all_c_fits:
        all_c_df = pd.concat(all_c_fits, ignore_index=True)

        for q in sorted(all_c_df["q"].unique()):
            qd = all_c_df[(all_c_df["q"] == q) & (all_c_df["R2"] > 0.5)]
            if len(qd) == 0:
                continue
            # Average across layers at each T
            avg_c = qd.groupby("T_over_Tc").agg({"c_fit": "mean", "c_err": "mean"}).reset_index()
            ax2.errorbar(
                avg_c["T_over_Tc"], avg_c["c_fit"],
                yerr=avg_c["c_err"].clip(upper=1.0),
                fmt="o-", color=MODEL_COLORS[q], markersize=3,
                linewidth=1.0, capsize=2,
                label=f"{MODEL_LABELS[q]} ($c_{{\\rm exact}}={CENTRAL_CHARGES[q]}$)",
            )

        # Exact values
        for q in sorted(all_c_df["q"].unique()):
            c_exact = CENTRAL_CHARGES.get(q)
            if c_exact:
                ax2.axhline(c_exact, color=MODEL_COLORS[q], ls=":", alpha=0.4, lw=1)

    ax2.axvline(1.0, color="gray", ls="--", alpha=0.4, lw=0.8)
    ax2.set_xlabel("$T / T_c$")
    ax2.set_ylabel("Central charge $c$")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.15)
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontsize=11,
             fontweight="bold", va="top")

    plt.tight_layout()
    path = Path(output_dir) / "fig2_entanglement_entropy.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# Figure 3: Rank-Exponent Duality (Supplemental)
# ──────────────────────────────────────────────────────────────


def generate_figure_3(rank_exponent_csvs, output_dir):
    """
    Fig 3: Log-log power-law plot.

    Args:
        rank_exponent_csvs: dict {q: csv_path}
        output_dir: output path
    """
    fig, ax = plt.subplots(figsize=(4, 3.5))

    for q, csv_path in sorted(rank_exponent_csvs.items()):
        df = pd.read_csv(csv_path)
        Tc = POTTS_TC[q]

        result = fit_rank_exponent(df, Tc, side="both")
        if result is None:
            continue

        ax.loglog(
            result["t_data"], result["erank_data"],
            "o", color=MODEL_COLORS[q], markersize=3, alpha=0.7,
        )

        t_fit = np.logspace(
            np.log10(result["t_data"].min()),
            np.log10(result["t_data"].max()),
            100,
        )
        ax.loglog(
            t_fit, power_law(t_fit, result["phi"], result["A"]),
            "-", color=MODEL_COLORS[q], linewidth=1.2, alpha=0.8,
            label=f"{MODEL_LABELS[q]}: $\\phi={result['phi']:.2f}$",
        )

    ax.set_xlabel("$|T - T_c| / T_c$")
    ax.set_ylabel("$\\Delta$ eRank")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, which="both")

    plt.tight_layout()
    path = Path(output_dir) / "fig3_rank_exponent.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="PRL Paper: Unified Publication Figure Generation"
    )
    # Result 1: Rank analysis CSVs
    parser.add_argument("--rank_ising", type=str, help="Ising rank analysis CSV")
    parser.add_argument("--rank_potts3", type=str, help="3-Potts rank analysis CSV")
    parser.add_argument("--rank_potts4", type=str, help="4-Potts rank analysis CSV")

    # Result 2: Entanglement analysis CSVs
    parser.add_argument("--entanglement_ising", type=str, help="Ising entanglement CSV")
    parser.add_argument("--entanglement_potts3", type=str, help="3-Potts entanglement CSV")
    parser.add_argument("--entanglement_potts4", type=str, help="4-Potts entanglement CSV")

    # Result 3: Rank exponent CSVs (critical mode rank analysis)
    parser.add_argument("--exponent_ising", type=str, help="Ising critical rank CSV")
    parser.add_argument("--exponent_potts3", type=str, help="3-Potts critical rank CSV")
    parser.add_argument("--exponent_potts4", type=str, help="4-Potts critical rank CSV")

    parser.add_argument("--output_dir", type=str, default="figs/prl_paper",
                        help="Output directory for figures")
    args = parser.parse_args()

    console = Console()
    apply_prl_style()

    console.print("[bold green]PRL Paper Figure Generation[/bold green]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Operator Counting ──
    rank_csvs = {}
    if args.rank_ising:
        rank_csvs[2] = args.rank_ising
    if args.rank_potts3:
        rank_csvs[3] = args.rank_potts3
    if args.rank_potts4:
        rank_csvs[4] = args.rank_potts4

    if len(rank_csvs) >= 2:
        console.print("\n[bold cyan]Generating Figure 1: Operator Counting[/bold cyan]")
        fig_path = generate_figure_1(rank_csvs, output_dir)
        console.print(f"[green]Fig 1:[/green] {fig_path}")
    else:
        console.print("[yellow]Skipping Fig 1: need >= 2 rank analysis CSVs[/yellow]")

    # ── Figure 2: Entanglement Entropy ──
    entanglement_csvs = {}
    if args.entanglement_ising:
        entanglement_csvs[2] = args.entanglement_ising
    if args.entanglement_potts3:
        entanglement_csvs[3] = args.entanglement_potts3
    if args.entanglement_potts4:
        entanglement_csvs[4] = args.entanglement_potts4

    if entanglement_csvs:
        console.print("\n[bold cyan]Generating Figure 2: Entanglement Entropy[/bold cyan]")
        fig_path = generate_figure_2(entanglement_csvs, output_dir)
        console.print(f"[green]Fig 2:[/green] {fig_path}")
    else:
        console.print("[yellow]Skipping Fig 2: no entanglement CSVs provided[/yellow]")

    # ── Figure 3: Rank-Exponent Duality ──
    exponent_csvs = {}
    if args.exponent_ising:
        exponent_csvs[2] = args.exponent_ising
    if args.exponent_potts3:
        exponent_csvs[3] = args.exponent_potts3
    if args.exponent_potts4:
        exponent_csvs[4] = args.exponent_potts4

    if exponent_csvs:
        console.print("\n[bold cyan]Generating Figure 3: Rank-Exponent Duality[/bold cyan]")
        fig_path = generate_figure_3(exponent_csvs, output_dir)
        console.print(f"[green]Fig 3:[/green] {fig_path}")
    else:
        console.print("[yellow]Skipping Fig 3: no exponent CSVs provided[/yellow]")

    console.print(f"\n[bold green]All figures saved to {output_dir}/[/bold green]")


if __name__ == "__main__":
    main()
