"""
Experiment 2: Low-Rank Compression Test

Tests the Low-Rank Hypothesis (Nature Physics 2024) by SVD-truncating model
weights and measuring free energy degradation across temperatures.

Key prediction: degradation(T, k) should peak near Tc ≈ 2.269, where
the model needs maximum effective rank to represent critical correlations.

Degradation is measured as:
    D(T, k) = E_{q_full}[log q_full(x|T) - log q_k(x|T)] / N
            = KL(q_full || q_k) / N  ≥ 0

Usage:
    # Interactive mode
    python analyze_compression.py

    # Command-line mode
    python analyze_compression.py --project Ising_VaTD_v0.15 \
        --group DiscretePixelCNN_lr1e-3_e500_f2d43d --seed 42 --device cuda:0

    # Quick mode (fewer temperatures and samples)
    python analyze_compression.py --project Ising_VaTD_v0.15 \
        --group DiscretePixelCNN_lr1e-3_e500_f2d43d --seed 42 --quick

    # Include per-block sensitivity analysis
    python analyze_compression.py --project Ising_VaTD_v0.15 \
        --group DiscretePixelCNN_lr1e-3_e500_f2d43d --seed 42 --per_layer

    # Replot from existing CSV
    python analyze_compression.py --replot runs/Ising_VaTD_v0.15/GROUP/compression_42.csv
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

import copy
from pathlib import Path
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
import argparse

from util import select_project, select_group, select_seed, select_device, load_model
from main import create_ising_energy_fn
from vatd_exact_partition import CRITICAL_TEMPERATURE
from analyze_rank import temperature_grid, exact_specific_heat


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

DEFAULT_RANK_FRACTIONS = [0.9, 0.75, 0.5, 0.25, 0.1]
PER_LAYER_RANK_FRACTION = 0.5

LAYER_CMAP = plt.cm.viridis
RANK_CMAP = plt.cm.plasma


# ──────────────────────────────────────────────────────────────
# Weight SVD Analysis
# ──────────────────────────────────────────────────────────────


def effective_rank(sv):
    """Shannon entropy-based effective rank."""
    sv = sv[sv > 1e-10]
    if len(sv) == 0:
        return 1.0
    p = sv / sv.sum()
    H = -(p * torch.log(p)).sum()
    return torch.exp(H).item()


def get_weight_svd(model):
    """
    Compute SVD of all Conv2d weights using effective weight (W * mask).

    Returns dict: {layer_name: {shape, singular_values, full_rank, erank}}
    """
    results = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            W = module.weight.data
            if hasattr(module, "mask"):
                W = W * module.mask
            shape = W.shape
            W_2d = W.reshape(shape[0], -1)
            _, S, _ = torch.linalg.svd(W_2d, full_matrices=False)
            results[name] = {
                "shape": shape,
                "singular_values": S.cpu(),
                "full_rank": len(S),
                "erank": effective_rank(S),
            }
    return results


# ──────────────────────────────────────────────────────────────
# SVD Truncation
# ──────────────────────────────────────────────────────────────


def truncate_model(model, rank_fraction):
    """
    Create a deepcopy with all Conv2d weights SVD-truncated.

    Uses effective weight (W * mask) for SVD to avoid including
    masked positions that carry no learned information.

    Args:
        model: DiscretePixelCNN
        rank_fraction: float in (0, 1] — fraction of singular values to keep

    Returns:
        (truncated_model, info_dict)
    """
    model_copy = copy.deepcopy(model)
    info = {}

    for name, module in model_copy.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            W = module.weight.data
            mask = module.mask if hasattr(module, "mask") else None
            W_eff = W * mask if mask is not None else W
            shape = W_eff.shape

            W_2d = W_eff.reshape(shape[0], -1)
            U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)

            full_rank = len(S)
            k = max(1, int(rank_fraction * full_rank))

            W_trunc_2d = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
            module.weight.data = W_trunc_2d.reshape(shape)

            energy_total = (S**2).sum().item()
            energy_kept = (S[:k] ** 2).sum().item()

            info[name] = {
                "full_rank": full_rank,
                "kept_rank": k,
                "energy_frac": energy_kept / max(energy_total, 1e-10),
            }

    return model_copy, info


def truncate_block(model, block_idx, rank_fraction):
    """
    Truncate only a specific residual block's Conv2d layers.

    Used for per-block sensitivity analysis: which blocks matter
    most at which temperatures?
    """
    model_copy = copy.deepcopy(model)

    block = model_copy.masked_conv.hidden_convs[block_idx]
    for module in block.modules():
        if isinstance(module, torch.nn.Conv2d):
            W = module.weight.data
            mask = module.mask if hasattr(module, "mask") else None
            W_eff = W * mask if mask is not None else W
            shape = W_eff.shape

            W_2d = W_eff.reshape(shape[0], -1)
            U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
            k = max(1, int(rank_fraction * len(S)))

            W_trunc_2d = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
            module.weight.data = W_trunc_2d.reshape(shape)

    return model_copy


# ──────────────────────────────────────────────────────────────
# Sample Generation & Log-Prob Evaluation
# ──────────────────────────────────────────────────────────────


def generate_samples(model, temperatures, batch_size, device, console=None):
    """
    Generate samples from the full model at each temperature.

    Returns:
        samples_dict: {T: tensor(B, 1, H, W)}
        energies_dict: {T: tensor(B, 1)}
    """
    if console is None:
        console = Console()

    L = model.size[0]
    energy_fn = create_ising_energy_fn(L=L, d=2, device=device)

    samples_dict = {}
    energies_dict = {}
    model.eval()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Sampling", total=len(temperatures))

        for T_val in temperatures:
            progress.update(task, description=f"Sampling T={T_val:.3f}")
            T_tensor = torch.full((batch_size,), T_val, device=device)

            with torch.no_grad():
                s = model.sample(batch_size=batch_size, T=T_tensor)
                E = energy_fn(s)

            samples_dict[T_val] = s.cpu()
            energies_dict[T_val] = E.cpu()
            progress.advance(task)

    return samples_dict, energies_dict


def evaluate_log_probs(model, samples_dict, device):
    """
    Evaluate log q(x|T) for each temperature on given samples.

    Returns dict: {T: tensor(B, 1)}
    """
    log_probs = {}
    model.eval()

    for T_val, samples in samples_dict.items():
        samples_dev = samples.to(device)
        T_tensor = torch.full((samples_dev.shape[0],), T_val, device=device)
        with torch.no_grad():
            lp = model.log_prob(samples_dev, T=T_tensor)
        log_probs[T_val] = lp.cpu()

    return log_probs


# ──────────────────────────────────────────────────────────────
# Main Analysis Loop
# ──────────────────────────────────────────────────────────────


def run_compression_analysis(
    model,
    temperatures,
    rank_fractions,
    device,
    batch_size=200,
    console=None,
    per_layer=False,
):
    """
    Full compression analysis pipeline.

    1. Generate samples from full model (slow, done once)
    2. Evaluate full model log_probs (baseline)
    3. For each rank_fraction: truncate → evaluate → compute degradation
    4. Optional: per-block sensitivity analysis

    Returns:
        df: DataFrame with global compression results
        df_layer: DataFrame with per-block results (or None)
    """
    if console is None:
        console = Console()

    L = model.size[0]
    N = L * L

    # ── Step 1: Generate samples ──
    console.print("\n[bold cyan]Step 1: Generating samples from full model[/bold cyan]")
    samples_dict, energies_dict = generate_samples(
        model, temperatures, batch_size, device, console
    )

    # ── Step 2: Full model baseline ──
    console.print("[bold cyan]Step 2: Evaluating full model (baseline)[/bold cyan]")
    full_log_probs = evaluate_log_probs(model, samples_dict, device)

    # Precompute full model F(T)
    full_F = {}
    for T_val in temperatures:
        beta = 1.0 / T_val
        lp_mean = full_log_probs[T_val].mean().item()
        E_mean = energies_dict[T_val].mean().item()
        full_F[T_val] = (lp_mean + beta * E_mean) / N

    # Record full model results (rank_fraction = 1.0)
    records = []
    for T_val in temperatures:
        beta = 1.0 / T_val
        lp_mean = full_log_probs[T_val].mean().item()
        E_mean = energies_dict[T_val].mean().item()

        records.append(
            {
                "T": T_val,
                "beta": beta,
                "T_over_Tc": T_val / CRITICAL_TEMPERATURE,
                "rank_fraction": 1.0,
                "log_q_per_site": lp_mean / N,
                "beta_E_per_site": beta * E_mean / N,
                "F_per_site": full_F[T_val],
                "degradation": 0.0,
            }
        )

    # ── Step 3: Truncated models ──
    console.print("[bold cyan]Step 3: Evaluating truncated models[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Compression", total=len(rank_fractions))

        for rf in rank_fractions:
            progress.update(task, description=f"rank_frac={rf:.2f}")

            model_trunc, trunc_info = truncate_model(model, rf)
            model_trunc = model_trunc.to(device)
            model_trunc.eval()

            trunc_log_probs = evaluate_log_probs(model_trunc, samples_dict, device)

            for T_val in temperatures:
                beta = 1.0 / T_val
                lp_trunc = trunc_log_probs[T_val].mean().item()
                lp_full = full_log_probs[T_val].mean().item()
                E_mean = energies_dict[T_val].mean().item()

                F_trunc = (lp_trunc + beta * E_mean) / N
                # Degradation = KL(q_full || q_trunc) / N ≥ 0
                degradation = (lp_full - lp_trunc) / N

                records.append(
                    {
                        "T": T_val,
                        "beta": beta,
                        "T_over_Tc": T_val / CRITICAL_TEMPERATURE,
                        "rank_fraction": rf,
                        "log_q_per_site": lp_trunc / N,
                        "beta_E_per_site": beta * E_mean / N,
                        "F_per_site": F_trunc,
                        "degradation": degradation,
                    }
                )

            del model_trunc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            progress.advance(task)

    df = pd.DataFrame(records)

    # ── Step 4: Per-block analysis (optional) ──
    df_layer = None
    if per_layer:
        console.print(
            "\n[bold cyan]Step 4: Per-block sensitivity analysis "
            f"(rank_frac={PER_LAYER_RANK_FRACTION})[/bold cyan]"
        )
        num_blocks = len(model.masked_conv.hidden_convs)
        layer_records = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Per-block", total=num_blocks)

            for bi in range(num_blocks):
                progress.update(task, description=f"Block {bi}")

                model_trunc = truncate_block(model, bi, PER_LAYER_RANK_FRACTION)
                model_trunc = model_trunc.to(device)
                model_trunc.eval()

                trunc_log_probs = evaluate_log_probs(
                    model_trunc, samples_dict, device
                )

                for T_val in temperatures:
                    lp_trunc = trunc_log_probs[T_val].mean().item()
                    lp_full = full_log_probs[T_val].mean().item()

                    layer_records.append(
                        {
                            "T": T_val,
                            "T_over_Tc": T_val / CRITICAL_TEMPERATURE,
                            "block": bi,
                            "degradation": (lp_full - lp_trunc) / N,
                        }
                    )

                del model_trunc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                progress.advance(task)

        df_layer = pd.DataFrame(layer_records)

    return df, df_layer


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────


def _block_label(name):
    """Convert 'masked_conv.hidden_convs.3.0' → 'B3.1×1↓'."""
    parts = name.split(".")
    if "hidden_convs" in name:
        block_idx = parts[2]
        sub_idx = int(parts[3])
        # Sequential indices: 0=1×1↓, 1=GELU, 2=k×k, 3=GELU, 4=1×1↑
        sub_names = {0: "1×1↓", 2: "k×k", 4: "1×1↑"}
        return f"B{block_idx}.{sub_names.get(sub_idx, sub_idx)}"
    elif "first_conv" in name:
        return "Input (A)"
    elif "first_fc" in name:
        return "FC₁"
    elif "final_fc" in name:
        return "FC_out"
    elif "hidden_fcs" in name:
        idx = parts[2]
        return f"FC_{int(idx)+2}"
    return name


def plot_weight_spectra(svd_info, figs_dir):
    """Plot normalized singular value decay for each conv layer."""
    # Group by residual block — plot one line per block's k×k conv
    kxk_layers = {
        k: v
        for k, v in svd_info.items()
        if "hidden_convs" in k and k.endswith(".2")  # k×k conv (middle)
    }

    if not kxk_layers:
        # Fallback: plot all layers
        kxk_layers = svd_info

    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(kxk_layers)
    colors = LAYER_CMAP(np.linspace(0.15, 0.85, n))

    for ci, (name, info) in enumerate(sorted(kxk_layers.items())):
        S = info["singular_values"]
        S_norm = S / S[0]
        label = _block_label(name)
        ax.semilogy(
            range(1, len(S_norm) + 1),
            S_norm.numpy(),
            color=colors[ci],
            linewidth=1.5,
            alpha=0.8,
            label=f"{label} (eR={info['erank']:.1f})",
        )

    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("$\\sigma_i / \\sigma_1$")
    ax.set_title("Weight SVD Spectra (k×k Conv per Block)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.15)
    ax.set_ylim(1e-4, 1.5)

    plt.tight_layout()
    path = Path(figs_dir) / "weight_svd_spectra.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_compression_results(df, L, figs_dir, df_layer=None):
    """
    2×2 figure (+ optional per-block figure):
      [0,0] Degradation D(T, k) curves for each rank fraction
      [0,1] Degradation heatmap (rank_fraction × T)
      [1,0] Max degradation vs exact Cv (dual axis)
      [1,1] Free energy F(T) curves for each rank fraction
    """
    Tc = CRITICAL_TEMPERATURE
    temperatures = np.sort(df["T"].unique())
    rank_fractions = sorted(df["rank_fraction"].unique())
    # Exclude 1.0 for truncation analysis
    trunc_fractions = [rf for rf in rank_fractions if rf < 1.0]

    n_rf = len(trunc_fractions)
    rf_colors = RANK_CMAP(np.linspace(0.15, 0.85, n_rf))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── [0,0] Degradation curves ──
    ax = axes[0, 0]
    for ci, rf in enumerate(trunc_fractions):
        rd = df[df["rank_fraction"] == rf].sort_values("T")
        ax.plot(
            rd["T"],
            rd["degradation"],
            "o-",
            color=rf_colors[ci],
            markersize=3,
            linewidth=1.5,
            label=f"k={rf:.0%}",
            alpha=0.85,
        )
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8, label="$T_c$")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Degradation $D(T,k)$ [nats/site]")
    ax.set_title("Log-Prob Degradation vs Temperature")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    # ── [0,1] Degradation heatmap ──
    ax = axes[0, 1]
    if len(trunc_fractions) >= 2:
        pivot = df[df["rank_fraction"] < 1.0].pivot_table(
            values="degradation", index="rank_fraction", columns="T"
        )
        pivot = pivot.sort_index(ascending=False)  # High fraction at top

        T_vals = pivot.columns.values
        im = ax.pcolormesh(
            np.log10(T_vals),
            np.arange(len(trunc_fractions)),
            pivot.values,
            cmap="inferno",
            shading="nearest",
        )
        ax.axvline(
            np.log10(Tc), color="cyan", ls="--", alpha=0.7, lw=1, label="$T_c$"
        )
        ax.set_xlabel("$\\log_{10}(T)$")
        ax.set_ylabel("Rank Fraction")
        ax.set_yticks(np.arange(len(trunc_fractions)))
        ax.set_yticklabels([f"{rf:.0%}" for rf in sorted(trunc_fractions, reverse=True)])
        plt.colorbar(im, ax=ax, label="Degradation [nats/site]")
        ax.set_title("Compression Sensitivity Heatmap")
        ax.legend(fontsize=7, loc="upper right")

    # ── [1,0] Max degradation vs Cv ──
    ax = axes[1, 0]
    # Use strongest truncation for clearest signal
    strongest_rf = min(trunc_fractions)
    rd = df[df["rank_fraction"] == strongest_rf].sort_values("T")
    T_arr = rd["T"].values
    deg_arr = rd["degradation"].values

    ax.plot(
        T_arr,
        deg_arr,
        "b-o",
        markersize=4,
        linewidth=2,
        label=f"Degradation (k={strongest_rf:.0%})",
    )
    ax.axvline(Tc, color="gray", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Degradation [nats/site]", color="blue")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.15)

    ax2 = ax.twinx()
    Cv = exact_specific_heat(L, temperatures)
    ax2.plot(
        temperatures,
        Cv,
        "r-",
        linewidth=2,
        alpha=0.6,
        label="Exact $C_v$ (Onsager)",
    )
    ax2.set_ylabel("$C_v$ / site", color="red")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.set_title("Compression Sensitivity vs Specific Heat")

    # Inset: normalized shape comparison near Tc
    inset = ax.inset_axes([0.42, 0.08, 0.55, 0.42])
    T_crit_lo, T_crit_hi = 1.5, 4.0
    crit_mask = (T_arr >= T_crit_lo) & (T_arr <= T_crit_hi)

    if crit_mask.sum() > 3:
        T_crit = T_arr[crit_mask]
        deg_crit = deg_arr[crit_mask]

        Cv_crit_mask = (temperatures >= T_crit_lo) & (temperatures <= T_crit_hi)
        Cv_crit = Cv[Cv_crit_mask]

        # Normalize for shape comparison
        deg_norm = (deg_crit - deg_crit.min()) / (
            deg_crit.max() - deg_crit.min() + 1e-10
        )
        Cv_norm = (Cv_crit - Cv_crit.min()) / (Cv_crit.max() - Cv_crit.min() + 1e-10)

        inset.plot(
            T_crit, deg_norm, "b-o", markersize=3, linewidth=1.5, label="Degradation"
        )
        inset.plot(
            temperatures[Cv_crit_mask],
            Cv_norm,
            "r-",
            linewidth=1.5,
            alpha=0.7,
            label="$C_v$",
        )
        inset.axvline(Tc, color="gray", ls="--", alpha=0.4, lw=0.7)
        inset.set_xlabel("$T$", fontsize=7)
        inset.set_ylabel("Normalized", fontsize=7)
        inset.tick_params(labelsize=6)
        inset.legend(fontsize=5, loc="upper right")
        inset.set_title("Critical region", fontsize=7)

    # ── [1,1] Relative degradation: D / |log_q_full| ──
    ax = axes[1, 1]
    for ci, rf in enumerate(trunc_fractions):
        rd = df[df["rank_fraction"] == rf].sort_values("T")
        rd_full = df[df["rank_fraction"] == 1.0].sort_values("T")
        # Merge to get full log_q at same temperatures
        merged = rd.merge(
            rd_full[["T", "log_q_per_site"]],
            on="T",
            suffixes=("", "_full"),
        )
        rel_deg = merged["degradation"] / merged["log_q_per_site_full"].abs().clip(lower=1e-8)

        ax.plot(
            merged["T"],
            rel_deg * 100,  # percent
            "o-",
            color=rf_colors[ci],
            markersize=3,
            linewidth=1.5,
            label=f"k={rf:.0%}",
            alpha=0.85,
        )
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8, label="$T_c$")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Relative Degradation [%]")
    ax.set_title("$D(T,k) \\,/\\, |\\log q_{\\mathrm{full}}|$")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"Low-Rank Compression Test: 2D Ising ($L={L}$, $T_c \\approx {Tc:.3f}$)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "compression_test.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Per-block heatmap (separate figure) ──
    path_layer = None
    if df_layer is not None and not df_layer.empty:
        path_layer = plot_per_block_sensitivity(df_layer, L, figs_dir)

    return path, path_layer


def plot_per_block_sensitivity(df_layer, L, figs_dir):
    """
    Per-block sensitivity heatmap: degradation when truncating
    each block individually at 50% rank.

    Cross-validates with Exp 1 (activation rank): blocks with high
    activation rank near Tc should also show high compression sensitivity.
    """
    Tc = CRITICAL_TEMPERATURE

    blocks = sorted(df_layer["block"].unique())
    temperatures = np.sort(df_layer["T"].unique())

    fig, ax = plt.subplots(figsize=(10, 4))

    pivot = df_layer.pivot_table(
        values="degradation", index="block", columns="T"
    )
    pivot = pivot.reindex(sorted(pivot.index))

    T_vals = pivot.columns.values
    im = ax.pcolormesh(
        np.log10(T_vals),
        np.arange(len(blocks)),
        pivot.values,
        cmap="inferno",
        shading="nearest",
    )
    ax.axvline(np.log10(Tc), color="cyan", ls="--", alpha=0.7, lw=1, label="$T_c$")
    ax.set_xlabel("$\\log_{10}(T)$")
    ax.set_ylabel("Residual Block")
    ax.set_yticks(np.arange(len(blocks)))
    ax.set_yticklabels([str(b) for b in blocks])
    plt.colorbar(im, ax=ax, label="Degradation [nats/site]")
    ax.legend(fontsize=8)

    ax.set_title(
        f"Per-Block Compression Sensitivity ($L={L}$, "
        f"rank={PER_LAYER_RANK_FRACTION:.0%})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "compression_per_block.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Low-Rank Compression Test"
    )
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Samples per temperature (default: 200)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer temperatures and samples",
    )
    parser.add_argument(
        "--per_layer",
        action="store_true",
        help="Include per-block sensitivity analysis",
    )
    parser.add_argument(
        "--replot",
        type=str,
        default=None,
        help="Regenerate plots from existing CSV",
    )
    args = parser.parse_args()

    console = Console()
    Tc = CRITICAL_TEMPERATURE

    console.print(
        "[bold green]Experiment 2: Low-Rank Compression Test[/bold green]"
    )
    console.print(f"Critical Temperature: Tc = {Tc:.4f}")

    # ── Replot mode ──
    if args.replot:
        csv_path = Path(args.replot)
        if not csv_path.exists():
            console.print(f"[red]CSV not found:[/red] {csv_path}")
            return

        console.print(f"[yellow]Replot mode:[/yellow] reading {csv_path}")
        df = pd.read_csv(csv_path)

        # Try to load per-block CSV
        layer_csv = csv_path.parent / csv_path.name.replace(
            "compression_", "compression_per_block_"
        )
        df_layer = pd.read_csv(layer_csv) if layer_csv.exists() else None

        # Infer L
        config_path = csv_path.parent / "config.yaml"
        if config_path.exists():
            from config import RunConfig

            config = RunConfig.from_yaml(str(config_path))
            L = config.net_config.get("size", 16)
            if not isinstance(L, int):
                L = L[0]
        else:
            L = 16

        figs_dir = Path(f"figs/{csv_path.parent.name}")
        figs_dir.mkdir(parents=True, exist_ok=True)

        fig_path, fig_layer_path = plot_compression_results(
            df, L, figs_dir, df_layer
        )
        console.print(f"[green]Main plot:[/green] {fig_path}")
        if fig_layer_path:
            console.print(f"[green]Per-block plot:[/green] {fig_layer_path}")
        console.print("[bold green]Replot complete.[/bold green]")
        return

    # ── Model selection ──
    project = args.project if args.project else select_project()
    group_name = args.group if args.group else select_group(project)
    seed = args.seed if args.seed else select_seed(project, group_name)
    device = args.device if args.device else select_device()

    console.print(f"\n[bold]Loading:[/bold] {project}/{group_name}/{seed}")
    model, config = load_model(project, group_name, seed)
    model = model.to(device)
    model.eval()

    L = model.size[0]
    num_layers = len(model.masked_conv.hidden_convs)
    num_params = sum(p.numel() for p in model.parameters())
    console.print(
        f"Lattice: {L}x{L}, Params: {num_params:,}, "
        f"Residual blocks: {num_layers}, "
        f"Hidden channels: {model.masked_conv.hidden_channels}"
    )

    # ── Output directories ──
    figs_dir = Path(f"figs/{group_name}")
    figs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(f"runs/{project}/{group_name}")

    # ── Weight SVD spectra ──
    console.print("\n[bold cyan]Weight SVD Analysis[/bold cyan]")
    svd_info = get_weight_svd(model)

    for name in sorted(svd_info.keys()):
        info = svd_info[name]
        console.print(
            f"  {_block_label(name):>12s}  "
            f"shape={str(list(info['shape'])):>20s}  "
            f"rank={info['full_rank']:>3d}  "
            f"eRank={info['erank']:.1f}"
        )

    fig_path = plot_weight_spectra(svd_info, figs_dir)
    console.print(f"[green]Weight spectra:[/green] {fig_path}")

    # ── Temperature grid ──
    if args.quick:
        temps = temperature_grid(T_min=0.8, T_max=6.0, n_coarse=12, n_critical=8)
        batch_size = 100
        rank_fractions = [0.75, 0.5, 0.25]
        console.print("[yellow]Quick mode: reduced grid & rank fractions[/yellow]")
    else:
        temps = temperature_grid(T_min=0.5, T_max=10.0, n_coarse=25, n_critical=15)
        batch_size = args.batch_size
        rank_fractions = DEFAULT_RANK_FRACTIONS

    console.print(
        f"Temperature grid: {len(temps)} points, "
        f"T in [{temps.min():.3f}, {temps.max():.3f}]"
    )
    console.print(f"Samples per temperature: {batch_size}")
    console.print(f"Rank fractions: {rank_fractions}")

    # ── Compression analysis ──
    df, df_layer = run_compression_analysis(
        model,
        temps,
        rank_fractions,
        device,
        batch_size=batch_size,
        console=console,
        per_layer=args.per_layer,
    )

    # ── Save results ──
    csv_path = output_dir / f"compression_{seed}.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]Data saved:[/green] {csv_path}")

    if df_layer is not None:
        layer_csv = output_dir / f"compression_per_block_{seed}.csv"
        df_layer.to_csv(layer_csv, index=False)
        console.print(f"[green]Per-block data:[/green] {layer_csv}")

    # ── Plots ──
    console.print("\n[bold cyan]Generating plots[/bold cyan]")
    fig_path, fig_layer_path = plot_compression_results(
        df, L, figs_dir, df_layer
    )
    console.print(f"[green]Main plot:[/green] {fig_path}")
    if fig_layer_path:
        console.print(f"[green]Per-block plot:[/green] {fig_layer_path}")

    # ── Summary ──
    console.print("\n[bold cyan]Summary: Peak degradation by rank fraction[/bold cyan]")
    console.print(
        f"{'Rank Frac':>10s}  {'Peak D':>10s}  {'at T':>7s}  "
        f"{'T/Tc':>6s}  {'Min D':>9s}"
    )
    console.print("-" * 55)

    for rf in rank_fractions:
        rd = df[df["rank_fraction"] == rf]
        if rd.empty:
            continue
        peak_idx = rd["degradation"].idxmax()
        peak_T = rd.loc[peak_idx, "T"]
        peak_D = rd.loc[peak_idx, "degradation"]
        min_D = rd["degradation"].min()

        console.print(
            f"{rf:>10.0%}  {peak_D:>10.4f}  {peak_T:>7.3f}  "
            f"{peak_T / Tc:>6.3f}  {min_D:>9.4f}"
        )

    console.print(f"\n[bold green]Analysis complete.[/bold green]")
    console.print(f"Figures: {figs_dir}/")
    console.print(f"Data:    {csv_path}")


if __name__ == "__main__":
    main()
