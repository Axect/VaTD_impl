#!/usr/bin/env python3
"""
Priority 0 Diagnostics for LatticeGPT v0.21 eRank Low-T Dip Analysis.

Four zero-GPU-cost diagnostics that resolve whether the eRank dip at
T/Tc ≈ 0.53–0.60 is under-training vs converged-but-degenerate:

  D1: ΔF(T) — Free energy error profile (v0.20 vs v0.21)
  D2: IS weight reconstruction — Adaptive sampling bias
  D3: Exact Cv(T) Schottky check — Physics baseline
  D4: REINFORCE variance comparison — Gradient quality

Usage:
    # All diagnostics
    python analyze_dip_diagnostic.py --device cuda:0

    # Specific diagnostics only
    python analyze_dip_diagnostic.py --device cuda:0 --diagnostics d1 d3

    # More samples for higher precision
    python analyze_dip_diagnostic.py --device cuda:0 --batch_size 2000
"""

import os
os.environ['VATD_NO_MHC'] = '1'

import torch
import numpy as np
import pandas as pd
import math
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'nature'])
except ImportError:
    pass

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn,
)

from util import load_model
from main import create_ising_energy_fn
from vatd_exact_partition import logZ as exact_logZ, CRITICAL_TEMPERATURE as Tc
from analyze_rank import temperature_grid

console = Console()

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

V20_PROJECT = "Ising_VaTD_v0.20"
V20_GROUP = "SpinGPT_lr1e-1_e250_0ac001"
V21_PROJECT = "Ising_VaTD_v0.21"
V21_GROUP = "SpinGPT_lr1e-1_e300_2b4582"
SEED = "42"

OUTPUT_DIR = Path(
    "outputs/latticegpt_v021_erank_low_t_dip_cause_analysis_20260303_v1/diagnostics"
)

# Dip location in T/Tc units
DIP_T_OVER_TC = 0.55  # T/Tc ≈ 0.53–0.60
DIP_T = DIP_T_OVER_TC * Tc  # ≈ 1.25


# ──────────────────────────────────────────────────────────────
# D1: Free Energy Error Profile
# ──────────────────────────────────────────────────────────────


def compute_free_energy_error(model, energy_fn, L, T_values, batch_size, device):
    """
    Compute ΔF(T) = F_model(T) - F_exact(T) per site for each temperature.

    F_model = <log_prob + β·E> / N  (variational free energy per site)
    F_exact = -logZ / N             (Onsager exact)
    """
    N = L * L
    results = []

    model.eval()
    with torch.no_grad():
        with Progress(
            SpinnerColumn(), TextColumn("[bold blue]D1: ΔF(T)"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("", total=len(T_values))
            for T_val in T_values:
                beta = 1.0 / float(T_val)
                T_tensor = torch.tensor([float(T_val)], device=device)
                beta_tensor = torch.tensor([beta], device=device).unsqueeze(-1)

                samples = model.sample(batch_size=batch_size, T=T_tensor)
                log_prob = model.log_prob(samples, T=T_tensor)  # [B, 1]
                energy = energy_fn(samples)  # [B, 1]

                # Per-site variational free energy for each sample
                f_model_per_sample = (log_prob + beta_tensor * energy) / N  # [B, 1]
                f_model_mean = f_model_per_sample.mean().item()
                f_model_std = f_model_per_sample.std().item() / math.sqrt(batch_size)

                # Exact per-site free energy
                exact_logz_total = exact_logZ(n=L, j=1.0, beta=torch.tensor(beta)).item()
                f_exact = -exact_logz_total / N

                delta_f = f_model_mean - f_exact

                results.append({
                    "T": T_val,
                    "beta": beta,
                    "T_over_Tc": T_val / Tc,
                    "F_model": f_model_mean,
                    "F_exact": f_exact,
                    "delta_F": delta_f,
                    "delta_F_std": f_model_std,
                })
                progress.advance(task)

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────
# D2: IS Weight Reconstruction
# ──────────────────────────────────────────────────────────────


def reconstruct_is_weights(model, energy_fn, L, device,
                           beta_min=0.1, beta_max=2.0, n_bins=32,
                           kappa=1.0, eps=1e-6, batch_size=256):
    """
    Reconstruct the adaptive sampling distribution by measuring
    REINFORCE weight variance at each temperature bin.

    Returns DataFrame with bin info, variance, adaptive probs, IS weights.
    """
    # Log-spaced bin edges matching v0.21 config
    log_edges = np.linspace(np.log(beta_min), np.log(beta_max), n_bins + 1)
    bin_edges = np.exp(log_edges)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean

    results = []

    model.eval()
    with torch.no_grad():
        with Progress(
            SpinnerColumn(), TextColumn("[bold blue]D2: IS weights"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("", total=n_bins)
            variances = np.zeros(n_bins)

            for k, beta_k in enumerate(bin_centers):
                T_k = 1.0 / float(beta_k)
                T_tensor = torch.tensor([float(T_k)], device=device)
                beta_tensor = torch.tensor([float(beta_k)], device=device).unsqueeze(-1)

                samples = model.sample(batch_size=batch_size, T=T_tensor)
                log_prob = model.log_prob(samples, T=T_tensor)  # [B, 1]
                energy = energy_fn(samples)  # [B, 1]

                reinforce_weight = log_prob.detach() + beta_tensor * energy  # [B, 1]
                var_k = reinforce_weight.var().item()
                variances[k] = var_k

                results.append({
                    "bin": k,
                    "beta_center": beta_k,
                    "T_center": T_k,
                    "T_over_Tc": T_k / Tc,
                    "reinforce_var": var_k,
                })
                progress.advance(task)

    # Compute adaptive probabilities: p_k ∝ (var_k + eps)^kappa
    raw = (variances + eps) ** kappa
    p_adaptive = raw / raw.sum()

    # Uniform probability
    p_uniform = np.ones(n_bins) / n_bins

    # IS weights: w_k = p_uniform / p_adaptive
    is_weights = p_uniform / p_adaptive

    # Effective sample size: n_eff/n = (Σ w_k)² / (n · Σ w_k²)
    n_eff_ratio = (is_weights.sum()) ** 2 / (n_bins * (is_weights ** 2).sum())

    df = pd.DataFrame(results)
    df["p_adaptive"] = p_adaptive
    df["p_uniform"] = p_uniform
    df["is_weight"] = is_weights

    return df, n_eff_ratio


# ──────────────────────────────────────────────────────────────
# D3: Exact Specific Heat (Schottky Check)
# ──────────────────────────────────────────────────────────────


def exact_cv_fine(L=16, T_min=0.5, T_max=3.0, n_points=200, J=1.0):
    """
    Compute exact Cv/N via central finite differences on Onsager logZ.

    Cv/N = β² · d²(logZ)/dβ² / N

    Uses 2nd-order central differences: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
    """
    N = L * L
    T_values = np.linspace(T_min, T_max, n_points)
    results = []

    h = 1e-5  # step for finite differences in β

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]D3: Exact Cv"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("", total=n_points)
        for T_val in T_values:
            beta = 1.0 / T_val

            logZ_p = exact_logZ(n=L, j=J, beta=torch.tensor(beta + h, dtype=torch.float64)).item()
            logZ_0 = exact_logZ(n=L, j=J, beta=torch.tensor(beta, dtype=torch.float64)).item()
            logZ_m = exact_logZ(n=L, j=J, beta=torch.tensor(beta - h, dtype=torch.float64)).item()

            d2_logZ_dbeta2 = (logZ_p - 2 * logZ_0 + logZ_m) / (h ** 2)
            cv_per_site = beta ** 2 * d2_logZ_dbeta2 / N

            results.append({
                "T": T_val,
                "beta": beta,
                "T_over_Tc": T_val / Tc,
                "Cv_per_site": cv_per_site,
            })
            progress.advance(task)

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────
# D4: REINFORCE Variance Comparison
# ──────────────────────────────────────────────────────────────


def reinforce_variance_profile(model, energy_fn, L, T_values, batch_size, device):
    """
    Compute REINFORCE weight statistics per temperature.

    Returns variance, mean, and gradient SNR = |mean| / sqrt(var).
    """
    results = []

    model.eval()
    with torch.no_grad():
        with Progress(
            SpinnerColumn(), TextColumn("[bold blue]D4: REINFORCE var"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("", total=len(T_values))
            for T_val in T_values:
                beta = 1.0 / float(T_val)
                T_tensor = torch.tensor([float(T_val)], device=device)
                beta_tensor = torch.tensor([beta], device=device).unsqueeze(-1)

                samples = model.sample(batch_size=batch_size, T=T_tensor)
                log_prob = model.log_prob(samples, T=T_tensor)  # [B, 1]
                energy = energy_fn(samples)  # [B, 1]

                reinforce_weight = log_prob.detach() + beta_tensor * energy  # [B, 1]
                var = reinforce_weight.var().item()
                mean = reinforce_weight.mean().item()
                snr = abs(mean) / math.sqrt(var) if var > 0 else float('inf')

                results.append({
                    "T": T_val,
                    "beta": beta,
                    "T_over_Tc": T_val / Tc,
                    "reinforce_var": var,
                    "reinforce_mean": mean,
                    "gradient_snr": snr,
                })
                progress.advance(task)

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────


def plot_delta_f(df_v20, df_v21, output_dir):
    """Plot 1: ΔF comparison (v0.20 left, v0.21 right, shared y-axis)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    for ax, df, label in [(ax1, df_v20, "v0.20"), (ax2, df_v21, "v0.21")]:
        x = df["T_over_Tc"]
        y = df["delta_F"]
        yerr = df["delta_F_std"]

        ax.plot(x, y, "o-", markersize=2, linewidth=1)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
        ax.axvline(1.0, color="red", ls="--", alpha=0.5, lw=0.8, label="$T_c$")
        ax.axvline(DIP_T_OVER_TC, color="orange", ls=":", alpha=0.7, lw=0.8,
                   label=f"dip ($T/T_c \\approx {DIP_T_OVER_TC}$)")
        ax.set_xlabel("$T / T_c$")
        ax.set_title(label)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.15)

    ax1.set_ylabel("$\\Delta F$ per site")

    fig.tight_layout()
    fig.savefig(output_dir / "delta_f_comparison.png", dpi=300)
    fig.savefig(output_dir / "delta_f_comparison.pdf")
    plt.close(fig)
    console.print(f"  [green]Saved:[/green] delta_f_comparison.png/pdf")


def plot_is_weights(is_data, n_eff_ratio, output_dir):
    """Plot 2: Adaptive sampling probability and IS weights."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4.5), sharex=True)

    bins = is_data["bin"].values
    t_over_tc = is_data["T_over_Tc"].values

    # Top: adaptive sampling probability
    colors = ["orange" if DIP_T_OVER_TC - 0.1 <= t <= DIP_T_OVER_TC + 0.1
              else "steelblue" for t in t_over_tc]
    ax1.bar(bins, is_data["p_adaptive"], color=colors, alpha=0.8, width=0.8)
    ax1.axhline(1.0 / len(bins), color="gray", ls="--", alpha=0.5, lw=0.8,
                label="Uniform")
    ax1.set_ylabel("$p_{\\mathrm{adaptive}}$")
    ax1.legend(fontsize=6)
    ax1.grid(True, alpha=0.15)

    # Bottom: IS weights
    ax2.bar(bins, is_data["is_weight"], color=colors, alpha=0.8, width=0.8)
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.5, lw=0.8)
    ax2.set_ylabel("IS weight $w_k$")
    ax2.set_xlabel("Bin index ($\\beta$ log-spaced)")
    ax2.grid(True, alpha=0.15)

    # Annotate n_eff/n
    fig.suptitle(f"$n_{{\\mathrm{{eff}}}}/n = {n_eff_ratio:.3f}$", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / "is_weight_distribution.png", dpi=300)
    fig.savefig(output_dir / "is_weight_distribution.pdf")
    plt.close(fig)
    console.print(f"  [green]Saved:[/green] is_weight_distribution.png/pdf")


def plot_cv_schottky(cv_data, rank_csv_path, output_dir):
    """Plot 3: Exact Cv/N with Schottky and dip marks, optional eRank overlay."""
    fig, ax1 = plt.subplots(figsize=(5, 3.5))

    ax1.plot(cv_data["T"], cv_data["Cv_per_site"], "k-", linewidth=1, label="Exact $C_v/N$")
    ax1.axvline(Tc, color="red", ls="--", alpha=0.5, lw=0.8, label="$T_c$")
    ax1.axvline(DIP_T, color="orange", ls=":", alpha=0.7, lw=0.8,
                label=f"$T_{{\\mathrm{{dip}}}} \\approx {DIP_T:.2f}$")

    # Schottky estimate: T_Schottky ≈ 8J / ln(L²) for L=16
    L = 16
    T_schottky = 8.0 / np.log(L * L)
    ax1.axvline(T_schottky, color="purple", ls="-.", alpha=0.5, lw=0.8,
                label=f"$T_{{\\mathrm{{Schottky}}}} \\approx {T_schottky:.2f}$")

    ax1.set_xlabel("$T$")
    ax1.set_ylabel("$C_v / N$")
    ax1.grid(True, alpha=0.15)

    # Overlay eRank from rank analysis CSV if available
    if rank_csv_path and Path(rank_csv_path).exists():
        rank_df = pd.read_csv(rank_csv_path)
        # Average channel eRank across layers
        erank_avg = rank_df.groupby("T")["channel_erank"].mean().reset_index()
        erank_avg = erank_avg[erank_avg["T"] <= 3.0]  # match Cv range

        ax2 = ax1.twinx()
        ax2.plot(erank_avg["T"], erank_avg["channel_erank"], "s--",
                 color="tab:blue", markersize=2, linewidth=0.8, alpha=0.7,
                 label="eRank (avg layers)")
        ax2.set_ylabel("Channel eRank", color="tab:blue")
        ax2.tick_params(axis='y', labelcolor="tab:blue")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=5, loc="upper right")
    else:
        ax1.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(output_dir / "cv_schottky_check.png", dpi=300)
    fig.savefig(output_dir / "cv_schottky_check.pdf")
    plt.close(fig)
    console.print(f"  [green]Saved:[/green] cv_schottky_check.png/pdf")


def plot_reinforce_variance(df_v20, df_v21, output_dir):
    """Plot 4: REINFORCE variance and gradient SNR comparison."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

    # Top: variance (log scale)
    ax1.semilogy(df_v20["T_over_Tc"], df_v20["reinforce_var"], "o-",
                 markersize=2, linewidth=1, label="v0.20")
    ax1.semilogy(df_v21["T_over_Tc"], df_v21["reinforce_var"], "s-",
                 markersize=2, linewidth=1, label="v0.21")
    ax1.axvline(1.0, color="red", ls="--", alpha=0.5, lw=0.8)
    ax1.axvspan(DIP_T_OVER_TC - 0.05, DIP_T_OVER_TC + 0.05,
                color="orange", alpha=0.1)
    ax1.set_ylabel("REINFORCE variance")
    ax1.legend(fontsize=6)
    ax1.grid(True, alpha=0.15)

    # Bottom: gradient SNR
    ax2.plot(df_v20["T_over_Tc"], df_v20["gradient_snr"], "o-",
             markersize=2, linewidth=1, label="v0.20")
    ax2.plot(df_v21["T_over_Tc"], df_v21["gradient_snr"], "s-",
             markersize=2, linewidth=1, label="v0.21")
    ax2.axvline(1.0, color="red", ls="--", alpha=0.5, lw=0.8)
    ax2.axvspan(DIP_T_OVER_TC - 0.05, DIP_T_OVER_TC + 0.05,
                color="orange", alpha=0.1)
    ax2.set_xlabel("$T / T_c$")
    ax2.set_ylabel("Gradient SNR $|\\mu| / \\sigma$")
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(output_dir / "reinforce_variance_comparison.png", dpi=300)
    fig.savefig(output_dir / "reinforce_variance_comparison.pdf")
    plt.close(fig)
    console.print(f"  [green]Saved:[/green] reinforce_variance_comparison.png/pdf")


def plot_summary(df_v20_fe, df_v21_fe, is_data, n_eff_ratio,
                 cv_data, df_v20_rv, df_v21_rv, output_dir):
    """Plot 5: 4-panel diagnostic summary."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    # Panel A: ΔF comparison
    ax = axes[0, 0]
    if df_v20_fe is not None:
        ax.plot(df_v20_fe["T_over_Tc"], df_v20_fe["delta_F"], "o-",
                markersize=2, linewidth=1, label="v0.20")
    if df_v21_fe is not None:
        ax.plot(df_v21_fe["T_over_Tc"], df_v21_fe["delta_F"], "s-",
                markersize=2, linewidth=1, label="v0.21")
    ax.axvline(1.0, color="red", ls="--", alpha=0.5, lw=0.8)
    ax.axvline(DIP_T_OVER_TC, color="orange", ls=":", alpha=0.7, lw=0.8)
    ax.set_ylabel("$\\Delta F$ per site")
    ax.set_title("(a) Free energy error", fontsize=8)
    if df_v20_fe is not None or df_v21_fe is not None:
        ax.legend(fontsize=5)
    ax.grid(True, alpha=0.15)

    # Panel B: IS weight distribution
    ax = axes[0, 1]
    if is_data is not None:
        t_over_tc = is_data["T_over_Tc"].values
        colors = ["orange" if DIP_T_OVER_TC - 0.1 <= t <= DIP_T_OVER_TC + 0.1
                  else "steelblue" for t in t_over_tc]
        ax.bar(is_data["bin"], is_data["p_adaptive"], color=colors, alpha=0.8, width=0.8)
        ax.axhline(1.0 / len(is_data), color="gray", ls="--", alpha=0.5, lw=0.8)
        ax.set_title(f"(b) Adaptive $p_k$ ($n_{{\\mathrm{{eff}}}}/n={n_eff_ratio:.3f}$)",
                     fontsize=8)
    else:
        ax.set_title("(b) IS weights (skipped)", fontsize=8)
    ax.set_ylabel("$p_{\\mathrm{adaptive}}$")
    ax.grid(True, alpha=0.15)

    # Panel C: Exact Cv
    ax = axes[1, 0]
    if cv_data is not None:
        ax.plot(cv_data["T_over_Tc"], cv_data["Cv_per_site"], "k-", linewidth=1)
        ax.axvline(1.0, color="red", ls="--", alpha=0.5, lw=0.8)
        ax.axvline(DIP_T_OVER_TC, color="orange", ls=":", alpha=0.7, lw=0.8)
    ax.set_xlabel("$T / T_c$")
    ax.set_ylabel("$C_v / N$")
    ax.set_title("(c) Exact specific heat", fontsize=8)
    ax.grid(True, alpha=0.15)

    # Panel D: REINFORCE variance
    ax = axes[1, 1]
    if df_v20_rv is not None:
        ax.semilogy(df_v20_rv["T_over_Tc"], df_v20_rv["reinforce_var"], "o-",
                    markersize=2, linewidth=1, label="v0.20")
    if df_v21_rv is not None:
        ax.semilogy(df_v21_rv["T_over_Tc"], df_v21_rv["reinforce_var"], "s-",
                    markersize=2, linewidth=1, label="v0.21")
    ax.axvline(1.0, color="red", ls="--", alpha=0.5, lw=0.8)
    ax.axvspan(DIP_T_OVER_TC - 0.05, DIP_T_OVER_TC + 0.05,
               color="orange", alpha=0.1)
    ax.set_xlabel("$T / T_c$")
    ax.set_ylabel("REINFORCE variance")
    ax.set_title("(d) Gradient variance", fontsize=8)
    if df_v20_rv is not None or df_v21_rv is not None:
        ax.legend(fontsize=5)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(output_dir / "diagnostic_summary.png", dpi=300)
    fig.savefig(output_dir / "diagnostic_summary.pdf")
    plt.close(fig)
    console.print(f"  [green]Saved:[/green] diagnostic_summary.png/pdf")


# ──────────────────────────────────────────────────────────────
# Decision Summary
# ──────────────────────────────────────────────────────────────


def print_decision_summary(df_v20_fe, df_v21_fe, is_data, n_eff_ratio,
                           cv_data, df_v20_rv, df_v21_rv):
    """Print the decision gate summary."""
    console.print("\n[bold yellow]═══ DECISION GATE SUMMARY ═══[/bold yellow]\n")

    # D1: ΔF at dip vs Tc
    if df_v21_fe is not None:
        # Find ΔF closest to dip temperature
        dip_idx = (df_v21_fe["T_over_Tc"] - DIP_T_OVER_TC).abs().idxmin()
        delta_f_dip = df_v21_fe.loc[dip_idx, "delta_F"]

        # Find ΔF closest to Tc
        tc_idx = (df_v21_fe["T_over_Tc"] - 1.0).abs().idxmin()
        delta_f_tc = df_v21_fe.loc[tc_idx, "delta_F"]

        ratio = abs(delta_f_dip / delta_f_tc) if abs(delta_f_tc) > 1e-10 else float('inf')

        console.print(f"  [bold]D1 — ΔF(T_dip) = {delta_f_dip:.6f}[/bold]")
        console.print(f"       ΔF(T_c)   = {delta_f_tc:.6f}")
        console.print(f"       |ΔF(dip)/ΔF(Tc)| = {ratio:.2f}")

        if ratio > 3.0:
            console.print("       [red]→ UNDER-TRAINING at dip region (ΔF(dip) >> ΔF(Tc))[/red]")
        elif ratio > 1.5:
            console.print("       [yellow]→ Moderate under-training at dip region[/yellow]")
        else:
            console.print("       [green]→ ΔF is comparable → converged representation[/green]")

        # Also compare v0.20 vs v0.21
        if df_v20_fe is not None:
            dip_idx_20 = (df_v20_fe["T_over_Tc"] - DIP_T_OVER_TC).abs().idxmin()
            delta_f_dip_20 = df_v20_fe.loc[dip_idx_20, "delta_F"]
            console.print(f"\n       v0.20 ΔF(T_dip) = {delta_f_dip_20:.6f}")
            console.print(f"       v0.21 ΔF(T_dip) = {delta_f_dip:.6f}")
            if abs(delta_f_dip) > 2 * abs(delta_f_dip_20):
                console.print("       [red]→ v0.21 significantly worse at dip[/red]")

    # D2: IS weights
    if is_data is not None and n_eff_ratio is not None:
        console.print(f"\n  [bold]D2 — n_eff/n = {n_eff_ratio:.4f}[/bold]")
        if n_eff_ratio < 0.1:
            console.print("       [red]→ SEVERE adaptive sampling bias (n_eff/n < 0.1)[/red]")
        elif n_eff_ratio < 0.5:
            console.print("       [yellow]→ Moderate adaptive sampling bias[/yellow]")
        else:
            console.print("       [green]→ Mild bias, adaptive sampling OK[/green]")

        # Check if dip region is under-sampled
        dip_bins = is_data[(is_data["T_over_Tc"] >= DIP_T_OVER_TC - 0.1) &
                           (is_data["T_over_Tc"] <= DIP_T_OVER_TC + 0.1)]
        if len(dip_bins) > 0:
            dip_prob = dip_bins["p_adaptive"].mean()
            avg_prob = 1.0 / len(is_data)
            console.print(f"       Dip region p_adaptive = {dip_prob:.4f} "
                          f"(uniform = {avg_prob:.4f})")
            if dip_prob < 0.5 * avg_prob:
                console.print("       [red]→ Dip region UNDER-SAMPLED by adaptive sampler[/red]")

    # D3: Schottky check
    if cv_data is not None:
        L = 16
        T_schottky = 8.0 / np.log(L * L)
        console.print(f"\n  [bold]D3 — T_Schottky = {T_schottky:.3f}, "
                      f"T_dip = {DIP_T:.3f}[/bold]")
        if abs(T_schottky - DIP_T) / DIP_T < 0.2:
            console.print("       [yellow]→ Dip near Schottky peak — "
                          "possible finite-size physics[/yellow]")
        else:
            console.print("       [green]→ Dip NOT near Schottky peak — "
                          "likely training artifact[/green]")

    # D4: REINFORCE variance
    if df_v21_rv is not None:
        dip_idx = (df_v21_rv["T_over_Tc"] - DIP_T_OVER_TC).abs().idxmin()
        var_dip = df_v21_rv.loc[dip_idx, "reinforce_var"]
        snr_dip = df_v21_rv.loc[dip_idx, "gradient_snr"]

        tc_idx = (df_v21_rv["T_over_Tc"] - 1.0).abs().idxmin()
        var_tc = df_v21_rv.loc[tc_idx, "reinforce_var"]
        snr_tc = df_v21_rv.loc[tc_idx, "gradient_snr"]

        console.print(f"\n  [bold]D4 — REINFORCE variance[/bold]")
        console.print(f"       At dip: var = {var_dip:.2f}, SNR = {snr_dip:.4f}")
        console.print(f"       At Tc:  var = {var_tc:.2f}, SNR = {snr_tc:.4f}")

        if var_dip > 10 * var_tc:
            console.print("       [red]→ Extremely high variance at dip — "
                          "gradient signal drowned[/red]")
        elif snr_dip < 0.1 * snr_tc:
            console.print("       [red]→ Very low SNR at dip — poor learning[/red]")

        if df_v20_rv is not None:
            dip_idx_20 = (df_v20_rv["T_over_Tc"] - DIP_T_OVER_TC).abs().idxmin()
            var_dip_20 = df_v20_rv.loc[dip_idx_20, "reinforce_var"]
            console.print(f"\n       v0.20 var(dip) = {var_dip_20:.2f}")
            console.print(f"       v0.21 var(dip) = {var_dip:.2f}")

    console.print("\n[bold yellow]═══════════════════════════════[/bold yellow]")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Priority 0 Diagnostics for LatticeGPT v0.21 eRank Dip"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Samples per temperature for D1/D4 (default: 1000)")
    parser.add_argument("--diagnostics", nargs="+", default=["d1", "d2", "d3", "d4"],
                        help="Which diagnostics to run (default: all)")
    parser.add_argument("--rank_csv", type=str, default=None,
                        help="Path to rank_analysis CSV for Cv overlay "
                             "(default: auto-detect from v0.21 runs/)")
    args = parser.parse_args()

    device = args.device
    diagnostics = [d.lower() for d in args.diagnostics]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Output directory:[/bold] {OUTPUT_DIR}")
    console.print(f"[bold]Device:[/bold] {device}")
    console.print(f"[bold]Diagnostics:[/bold] {diagnostics}")
    console.print(f"[bold]Batch size:[/bold] {args.batch_size}")

    # ── Temperature grid ──
    # Match the 40-point grid from rank_analysis
    T_values = temperature_grid(T_min=0.5, T_max=10.0, n_coarse=25, n_critical=15, Tc=Tc)
    console.print(f"[bold]Temperature grid:[/bold] {len(T_values)} points, "
                  f"T ∈ [{T_values[0]:.2f}, {T_values[-1]:.2f}]")

    # ── Auto-detect rank CSV ──
    rank_csv = args.rank_csv
    if rank_csv is None:
        default_path = Path(f"runs/{V21_PROJECT}/{V21_GROUP}/rank_analysis_{SEED}.csv")
        if default_path.exists():
            rank_csv = str(default_path)

    # ── Load models ──
    need_model = any(d in diagnostics for d in ["d1", "d2", "d4"])

    model_v20 = model_v21 = energy_fn = None
    L = 16

    if need_model:
        console.print("\n[bold cyan]Loading v0.20 model...[/bold cyan]")
        model_v20, _ = load_model(V20_PROJECT, V20_GROUP, SEED)
        model_v20 = model_v20.to(device)
        model_v20.eval()
        L = model_v20.size[0] if hasattr(model_v20, 'size') else 16

        console.print("[bold cyan]Loading v0.21 model...[/bold cyan]")
        model_v21, _ = load_model(V21_PROJECT, V21_GROUP, SEED)
        model_v21 = model_v21.to(device)
        model_v21.eval()

        energy_fn = create_ising_energy_fn(L=L, d=2, device=device)

    # ── Results storage ──
    df_v20_fe = df_v21_fe = None
    is_data = None
    n_eff_ratio = None
    cv_data = None
    df_v20_rv = df_v21_rv = None

    # ── D1: Free energy error ──
    if "d1" in diagnostics:
        console.print("\n[bold cyan]═══ D1: Free Energy Error Profile ═══[/bold cyan]")

        console.print("  Computing v0.20...")
        df_v20_fe = compute_free_energy_error(
            model_v20, energy_fn, L, T_values, args.batch_size, device
        )
        df_v20_fe.to_csv(OUTPUT_DIR / "free_energy_error_v020.csv", index=False)

        console.print("  Computing v0.21...")
        df_v21_fe = compute_free_energy_error(
            model_v21, energy_fn, L, T_values, args.batch_size, device
        )
        df_v21_fe.to_csv(OUTPUT_DIR / "free_energy_error_v021.csv", index=False)

        plot_delta_f(df_v20_fe, df_v21_fe, OUTPUT_DIR)

    # ── D2: IS weight reconstruction ──
    if "d2" in diagnostics:
        console.print("\n[bold cyan]═══ D2: IS Weight Reconstruction ═══[/bold cyan]")

        # Use v0.21 config parameters
        is_data, n_eff_ratio = reconstruct_is_weights(
            model_v21, energy_fn, L, device,
            beta_min=0.1, beta_max=2.0, n_bins=32,
            kappa=1.0, eps=1e-6, batch_size=256,
        )
        is_data.to_csv(OUTPUT_DIR / "is_weight_reconstruction.csv", index=False)
        console.print(f"  n_eff/n = {n_eff_ratio:.4f}")

        plot_is_weights(is_data, n_eff_ratio, OUTPUT_DIR)

    # ── D3: Exact Cv Schottky check ──
    if "d3" in diagnostics:
        console.print("\n[bold cyan]═══ D3: Exact Cv (Schottky Check) ═══[/bold cyan]")

        cv_data = exact_cv_fine(L=L, T_min=0.5, T_max=3.0, n_points=200)
        cv_data.to_csv(OUTPUT_DIR / "cv_schottky.csv", index=False)

        plot_cv_schottky(cv_data, rank_csv, OUTPUT_DIR)

    # ── D4: REINFORCE variance ──
    if "d4" in diagnostics:
        console.print("\n[bold cyan]═══ D4: REINFORCE Variance Comparison ═══[/bold cyan]")

        console.print("  Computing v0.20...")
        df_v20_rv = reinforce_variance_profile(
            model_v20, energy_fn, L, T_values, args.batch_size, device
        )
        df_v20_rv.to_csv(OUTPUT_DIR / "reinforce_variance_v020.csv", index=False)

        console.print("  Computing v0.21...")
        df_v21_rv = reinforce_variance_profile(
            model_v21, energy_fn, L, T_values, args.batch_size, device
        )
        df_v21_rv.to_csv(OUTPUT_DIR / "reinforce_variance_v021.csv", index=False)

        plot_reinforce_variance(df_v20_rv, df_v21_rv, OUTPUT_DIR)

    # ── Summary plot ──
    has_any = any(x is not None for x in [df_v20_fe, df_v21_fe, is_data, cv_data,
                                           df_v20_rv, df_v21_rv])
    if has_any:
        console.print("\n[bold cyan]═══ Generating Summary Plot ═══[/bold cyan]")
        plot_summary(df_v20_fe, df_v21_fe, is_data, n_eff_ratio,
                     cv_data, df_v20_rv, df_v21_rv, OUTPUT_DIR)

    # ── Decision gate ──
    print_decision_summary(df_v20_fe, df_v21_fe, is_data, n_eff_ratio,
                           cv_data, df_v20_rv, df_v21_rv)

    console.print(f"\n[bold green]All outputs saved to:[/bold green] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
