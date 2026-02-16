"""
Low-Rank Analysis of DiscretePixelCNN Internal Representations.

Tests the Low-Rank Hypothesis (Nature Physics 2024) by measuring the effective rank
of intermediate feature maps across the 2D Ising model phase transition.

Key prediction: effective_rank(T) should show NON-MONOTONIC behavior peaking
near Tc ≈ 2.269, mirroring the specific heat divergence.

Usage:
    # Interactive mode
    python analyze_rank.py

    # Command-line mode
    python analyze_rank.py --project Ising_VaTD_v0.15 \
        --group DiscretePixelCNN_lr1e-3_e500_f2d43d --seed 42 --device cuda:0

    # Quick mode (fewer temperatures and samples)
    python analyze_rank.py --project Ising_VaTD_v0.15 \
        --group DiscretePixelCNN_lr1e-3_e500_f2d43d --seed 42 --quick
"""

import os
os.environ['VATD_NO_MHC'] = '1'  # Prevent mHC.cu CUDA extension from loading

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

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
from vatd_exact_partition import logZ as exact_logZ, CRITICAL_TEMPERATURE


# ──────────────────────────────────────────────────────────────
# Rank Metrics
# ──────────────────────────────────────────────────────────────


def effective_rank(singular_values):
    """
    Effective rank via Shannon entropy of normalized singular values.

    Roy & Bhattacharya (2007): erank(A) = exp(H(p))
    where p_i = σ_i / Σ σ_j and H(p) = -Σ p_i log(p_i).

    Returns a continuous scalar in [1, min(m,n)].
    """
    sv = singular_values[singular_values > 1e-10]
    if len(sv) == 0:
        return 1.0
    p = sv / sv.sum()
    H = -(p * torch.log(p)).sum()
    return torch.exp(H).item()


def stable_rank(singular_values):
    """
    Stable rank: ||A||_F² / ||A||_2² = Σσ_i² / σ_max².

    More robust to noise than effective rank.
    """
    sv = singular_values[singular_values > 1e-10]
    if len(sv) == 0:
        return 1.0
    return (sv**2).sum().item() / (sv[0] ** 2).item()


def participation_ratio(singular_values):
    """
    Participation ratio: (Σσ_i²)² / Σσ_i⁴.

    Counts how many singular values "participate" in the spectrum.
    """
    sv = singular_values[singular_values > 1e-10]
    if len(sv) == 0:
        return 1.0
    sv2 = sv**2
    return (sv2.sum() ** 2 / (sv2**2).sum()).item()


def numerical_rank(singular_values, threshold=0.99):
    """
    Numerical rank: minimum k such that Σᵢ₌₁ᵏ σᵢ² / Σ σⱼ² ≥ threshold.

    The only discrete (integer) rank metric. Directly answers
    "how many components carry threshold% of the energy?"
    """
    sv = singular_values[singular_values > 1e-10]
    if len(sv) == 0:
        return 1
    energy = sv**2
    total = energy.sum()
    cumsum = torch.cumsum(energy, dim=0)
    mask = cumsum >= threshold * total
    if mask.any():
        return (mask.float().argmax() + 1).item()
    return len(sv)



def renyi_rank(singular_values, alpha=2.0):
    """
    Rényi rank: exp(H_α(p)) where H_α = 1/(1-α) · ln(Σ pᵢ^α).

    Uses L1-normalized singular values: p_i = σ_i / Σσ_j (same convention
    as eRank). α=2 gives the "collision rank", more sensitive to dominant
    modes than Shannon-based eRank (α→1).
    """
    sv = singular_values[singular_values > 1e-10]
    if len(sv) == 0:
        return 1.0
    p = sv / sv.sum()
    H_alpha = (1.0 / (1.0 - alpha)) * torch.log((p**alpha).sum())
    return torch.exp(H_alpha).item()



def nuclear_rank(singular_values):
    """
    Nuclear rank: Σσᵢ / σ_max  (L1/L∞ ratio).

    The L1 analog of stable rank (which uses L2²/L∞²). Counts how many
    singular values "participate" by magnitude rather than energy.
    Always nuclear_rank ≥ stable_rank (Cauchy-Schwarz).

    From Thibeault et al., Nature Physics (2024).
    """
    sv = singular_values[singular_values > 1e-10]
    if len(sv) == 0:
        return 1.0
    return (sv.sum() / sv[0]).item()


def elbow_rank(singular_values):
    """
    Elbow rank: index of maximum perpendicular distance from the diagonal y = 1 - x.

    Given normalized cumulative singular values, finds the "elbow" point where
    the spectrum transitions from signal to noise. This is the geometric method
    from the Thibeault et al. (2024) codebase.

    Returns integer rank (1-indexed).
    """
    sv = singular_values[singular_values > 1e-10]
    if len(sv) <= 1:
        return 1
    # Normalize SVs to [0, 1] range on x-axis
    n = len(sv)
    x = torch.arange(1, n + 1, dtype=torch.float64) / n  # [1/n, 2/n, ..., 1]
    # Cumulative energy fraction
    energy = sv**2
    y = torch.cumsum(energy, dim=0) / energy.sum()
    # Distance from diagonal y = x (the "null" line for uniform spectrum)
    # For the elbow method: distance from line connecting (0,0) to (1,1)
    # d = |y_i - x_i| / sqrt(2)
    dist = (y - x).abs()
    return (dist.argmax() + 1).item()


def optimal_hard_threshold(singular_values, N, M):
    """
    Gavish & Donoho (2014) optimal hard threshold for singular values.

    Returns the number of singular values above the optimal threshold
    λ* = ω(β) · σ_median · √max(N,M), where β = min(N,M)/max(N,M)
    and ω(β) is the optimal threshold coefficient.

    σ_median is the median singular value (robust noise estimator).
    For unknown noise level, σ_noise = median(σ) / √(μ_β), where
    μ_β is the median of the Marchenko-Pastur distribution.

    Reference: Gavish & Donoho, "The optimal hard threshold for
    singular values is 4/√3", IEEE Trans. Inf. Theory (2014).
    """
    sv = singular_values[singular_values > 1e-10]
    if len(sv) <= 1:
        return 1

    beta = min(N, M) / max(N, M)

    # Optimal threshold coefficient ω(β) from Gavish & Donoho
    # Approximation valid for all β ∈ (0, 1]:
    # ω(β) ≈ 0.56·β³ - 0.95·β² + 1.82·β + 1.43
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43

    # Estimate noise level from median SV
    sv_median = sv.median().item()

    # Median of MP distribution (approximation for β → 0 is √(2β))
    # More accurate: μ_β ≈ √(2β + β²/3) for moderate β
    mu_beta = np.sqrt(2 * beta + beta**2 / 3)
    if mu_beta < 1e-12:
        return len(sv)

    sigma_noise = sv_median / (mu_beta * np.sqrt(max(N, M)))

    # Threshold
    threshold = omega * sigma_noise * np.sqrt(max(N, M))

    # Count SVs above threshold
    count = (sv > threshold).sum().item()
    return max(count, 1)


# ──────────────────────────────────────────────────────────────
# Exact Specific Heat (Onsager)
# ──────────────────────────────────────────────────────────────


def exact_specific_heat(L, temperatures, J=1.0):
    """
    Exact specific heat per site from Onsager's partition function.

    Cv/N = β²/N · ∂²logZ/∂β²

    Computed via central finite differences of exact logZ.
    """
    N = L * L
    db = 1e-5
    Cv = np.zeros(len(temperatures))

    for i, T in enumerate(temperatures):
        beta = 1.0 / T
        lZ_p = exact_logZ(L, J, beta + db)
        lZ_m = exact_logZ(L, J, beta - db)
        lZ_0 = exact_logZ(L, J, beta)

        if isinstance(lZ_p, torch.Tensor):
            lZ_p = lZ_p.item()
        if isinstance(lZ_m, torch.Tensor):
            lZ_m = lZ_m.item()
        if isinstance(lZ_0, torch.Tensor):
            lZ_0 = lZ_0.item()

        d2logZ = (lZ_p - 2 * lZ_0 + lZ_m) / (db**2)
        Cv[i] = beta**2 * d2logZ / N

    return Cv


# ──────────────────────────────────────────────────────────────
# Activation Collection via Hooks
# ──────────────────────────────────────────────────────────────


def collect_activations(model, samples, T_val, device):
    """
    Collect intermediate activations from all residual blocks using forward pre-hooks.

    Pre-hook on hidden_convs[i] captures the input to block i,
    which is the output after residual addition at block i-1.
    Pre-hook on first_fc captures the state after ALL residual blocks.

    Args:
        model: DiscretePixelCNN (eval mode)
        samples: [B, 1, H, W] in {-1, +1}
        T_val: scalar temperature
        device: torch device

    Returns:
        dict {layer_idx: Tensor[B, C, H, W], 'final': Tensor[B, C, H, W]}
    """
    captured = {}
    hooks = []

    # Pre-hooks on each hidden conv block
    for i, conv in enumerate(model.masked_conv.hidden_convs):

        def make_hook(idx):
            def hook_fn(module, inp):
                captured[idx] = inp[0].detach().cpu()

            return hook_fn

        hooks.append(conv.register_forward_pre_hook(make_hook(i)))

    # Pre-hook on first_fc → captures state after last residual block + skip fusion
    def final_hook(module, inp):
        captured["final"] = inp[0].detach().cpu()

    hooks.append(model.masked_conv.first_fc.register_forward_pre_hook(final_hook))

    # Trigger forward pass via log_prob
    T_tensor = torch.full((samples.shape[0],), T_val, device=device)
    with torch.no_grad():
        model.log_prob(samples, T=T_tensor)

    for h in hooks:
        h.remove()

    return captured


# ──────────────────────────────────────────────────────────────
# Temperature Grid
# ──────────────────────────────────────────────────────────────


def temperature_grid(T_min=0.5, T_max=10.0, n_coarse=25, n_critical=15):
    """
    Generate temperature grid with extra density near Tc.

    Merges a log-spaced coarse grid with a linear dense grid
    around [0.8·Tc, 1.2·Tc].
    """
    Tc = CRITICAL_TEMPERATURE
    T_coarse = np.logspace(np.log10(T_min), np.log10(T_max), n_coarse)
    T_dense = np.linspace(0.8 * Tc, 1.2 * Tc, n_critical)
    T_all = np.unique(np.concatenate([T_coarse, T_dense]))
    T_all.sort()
    return T_all


# ──────────────────────────────────────────────────────────────
# Main Analysis Loop
# ──────────────────────────────────────────────────────────────


def run_analysis(model, temperatures, device, batch_size=200, n_batches=3, console=None):
    """
    For each temperature: generate samples → collect activations → compute rank metrics.

    Returns a DataFrame with columns:
        T, beta, T_over_Tc, layer,
        channel_erank, channel_stable_rank, channel_pr,
        channel_numerical_rank_99, channel_numerical_rank_95,
        channel_renyi2,
        channel_nuclear_rank, channel_elbow_rank, channel_opt_threshold,
        spatial_erank, spatial_stable_rank, spatial_pr,
        spatial_numerical_rank_99, spatial_numerical_rank_95,
        spatial_renyi2,
        spatial_nuclear_rank, spatial_elbow_rank, spatial_opt_threshold
    """
    if console is None:
        console = Console()

    num_layers = len(model.masked_conv.hidden_convs)
    layer_keys = list(range(num_layers)) + ["final"]

    records = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Rank analysis", total=len(temperatures))

        for T_val in temperatures:
            beta_val = 1.0 / T_val
            progress.update(task, description=f"T={T_val:.3f} (β={beta_val:.3f})")

            # Generate samples at this temperature
            all_samples = []
            T_tensor = torch.full((batch_size,), T_val, device=device)
            model.eval()

            with torch.no_grad():
                for _ in range(n_batches):
                    s = model.sample(batch_size=batch_size, T=T_tensor)
                    all_samples.append(s)

            all_samples = torch.cat(all_samples, dim=0)  # [N, 1, H, W]

            # Collect activations via single forward pass
            activations = collect_activations(model, all_samples, T_val, device)

            # Compute rank metrics per layer
            for lk in layer_keys:
                if lk not in activations:
                    continue

                act = activations[lk]  # [N, C, H, W]
                N, C, H, W = act.shape

                # ── Channel rank: average over spatial dims → [N, C] ──
                act_ch = act.mean(dim=(-2, -1))
                act_ch = act_ch - act_ch.mean(dim=0, keepdim=True)
                _, S_ch, _ = torch.linalg.svd(act_ch, full_matrices=False)

                # ── Spatial rank: average over channels → [N, H*W] ──
                act_sp = act.mean(dim=1).reshape(N, H * W)
                act_sp = act_sp - act_sp.mean(dim=0, keepdim=True)
                _, S_sp, _ = torch.linalg.svd(act_sp, full_matrices=False)

                layer_name = f"layer_{lk}" if isinstance(lk, int) else lk

                records.append(
                    {
                        "T": T_val,
                        "beta": beta_val,
                        "T_over_Tc": T_val / CRITICAL_TEMPERATURE,
                        "layer": layer_name,
                        "channel_erank": effective_rank(S_ch),
                        "channel_stable_rank": stable_rank(S_ch),
                        "channel_pr": participation_ratio(S_ch),
                        "channel_numerical_rank_99": numerical_rank(S_ch, 0.99),
                        "channel_numerical_rank_95": numerical_rank(S_ch, 0.95),
                        "channel_renyi2": renyi_rank(S_ch, 2.0),
                        "channel_nuclear_rank": nuclear_rank(S_ch),
                        "channel_elbow_rank": elbow_rank(S_ch),
                        "channel_opt_threshold": optimal_hard_threshold(S_ch, N, C),
                        "spatial_erank": effective_rank(S_sp),
                        "spatial_stable_rank": stable_rank(S_sp),
                        "spatial_pr": participation_ratio(S_sp),
                        "spatial_numerical_rank_99": numerical_rank(S_sp, 0.99),
                        "spatial_numerical_rank_95": numerical_rank(S_sp, 0.95),
                        "spatial_renyi2": renyi_rank(S_sp, 2.0),
                        "spatial_nuclear_rank": nuclear_rank(S_sp),
                        "spatial_elbow_rank": elbow_rank(S_sp),
                        "spatial_opt_threshold": optimal_hard_threshold(S_sp, N, H * W),
                    }
                )

            progress.advance(task)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

LAYER_CMAP = plt.cm.viridis


def _layer_sort_key(name):
    if name.startswith("layer_"):
        return (0, int(name.split("_")[1]))
    return (1, 0)


def _layer_label(name):
    return name.replace("layer_", "Block ").replace("final", "Final")


def plot_rank_vs_temperature(df, L, figs_dir):
    """
    3×2 figure:
      [0,0] Channel effective rank vs T per layer
      [0,1] Spatial effective rank vs T per layer
      [1,0] Numerical Rank vs T (99% and 95% thresholds)
      [1,1] Nuclear Rank vs T per layer
      [2,0] Avg channel erank vs T + exact Cv overlay
      [2,1] Heatmap of channel erank [layer × T]
    """
    temperatures = np.sort(df["T"].unique())
    Cv = exact_specific_heat(L, temperatures)

    layers = sorted(df["layer"].unique(), key=_layer_sort_key)
    n_layers = len(layers)
    colors = LAYER_CMAP(np.linspace(0.15, 0.85, n_layers))

    has_extended = "channel_numerical_rank_99" in df.columns

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    Tc = CRITICAL_TEMPERATURE

    # ── [0,0] Channel effective rank ──
    ax = axes[0, 0]
    for ci, layer in enumerate(layers):
        ld = df[df["layer"] == layer].sort_values("T")
        ax.plot(
            ld["T"], ld["channel_erank"],
            "o-", color=colors[ci], markersize=3, linewidth=1.2,
            label=_layer_label(layer), alpha=0.85,
        )
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Effective Rank (channel)")
    ax.set_title("Channel Effective Rank")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.15)

    # ── [0,1] Spatial effective rank ──
    ax = axes[0, 1]
    for ci, layer in enumerate(layers):
        ld = df[df["layer"] == layer].sort_values("T")
        ax.plot(
            ld["T"], ld["spatial_erank"],
            "o-", color=colors[ci], markersize=3, linewidth=1.2,
            label=_layer_label(layer), alpha=0.85,
        )
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Effective Rank (spatial)")
    ax.set_title("Spatial Effective Rank")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.15)

    # ── [1,0] Numerical Rank vs T ──
    ax = axes[1, 0]
    if has_extended:
        for ci, layer in enumerate(layers):
            ld = df[df["layer"] == layer].sort_values("T")
            ax.plot(
                ld["T"], ld["channel_numerical_rank_99"],
                "o-", color=colors[ci], markersize=3, linewidth=1.2,
                label=f"{_layer_label(layer)} (99%)", alpha=0.85,
            )
            ax.plot(
                ld["T"], ld["channel_numerical_rank_95"],
                "s--", color=colors[ci], markersize=2, linewidth=0.8,
                alpha=0.5,
            )
        ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
        ax.set_xlabel("Temperature $T$")
        ax.set_ylabel("Numerical Rank $k$")
        ax.set_title("Numerical Rank (solid: 99%, dashed: 95%)")
        ax.set_xscale("log")
        ax.legend(fontsize=5, ncol=2)
        ax.grid(True, alpha=0.15)
    else:
        ax.set_visible(False)

    # ── [1,1] Nuclear Rank vs T ──
    ax = axes[1, 1]
    if has_extended and "channel_nuclear_rank" in df.columns:
        for ci, layer in enumerate(layers):
            ld = df[df["layer"] == layer].sort_values("T")
            ax.plot(
                ld["T"], ld["channel_nuclear_rank"],
                "o-", color=colors[ci], markersize=3, linewidth=1.2,
                label=_layer_label(layer), alpha=0.85,
            )
        ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
        ax.set_xlabel("Temperature $T$")
        ax.set_ylabel("Nuclear Rank ($\\Sigma\\sigma_i / \\sigma_{max}$)")
        ax.set_title("Nuclear Rank (L1/L$\\infty$)")
        ax.set_xscale("log")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.15)
    else:
        ax.set_visible(False)

    # ── [2,0] eRank vs Cv (dual axis) ──
    ax = axes[2, 0]
    avg_rank = df.groupby("T")["channel_erank"].mean().sort_index()
    T_arr = avg_rank.index.values
    rank_arr = avg_rank.values

    ax.plot(
        T_arr, rank_arr,
        "b-o", markersize=4, linewidth=2, label="Avg Channel eRank",
    )
    ax.axvline(Tc, color="gray", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Effective Rank", color="blue")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.15)

    ax2 = ax.twinx()
    ax2.plot(
        temperatures, Cv,
        "r-", linewidth=2, alpha=0.6, label="Exact $C_v$ (Onsager)",
    )
    ax2.set_ylabel("$C_v$ / site", color="red")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title("eRank (entropy-like) vs $C_v$ ($\\sim d$S$/dT$)")

    # ── [2,1] Heatmap ──
    ax = axes[2, 1]
    numeric_layers = [l for l in layers if l.startswith("layer_")]
    if numeric_layers:
        pivot = df[df["layer"].isin(numeric_layers)].pivot_table(
            values="channel_erank", index="layer", columns="T",
        )
        pivot = pivot.reindex(sorted(pivot.index, key=_layer_sort_key))

        T_vals = pivot.columns.values
        im = ax.pcolormesh(
            np.log10(T_vals),
            np.arange(len(numeric_layers)),
            pivot.values,
            cmap="magma", shading="nearest",
        )
        ax.axvline(np.log10(Tc), color="cyan", ls="--", alpha=0.7, lw=1)
        ax.set_xlabel("$\\log_{10}(T)$")
        ax.set_ylabel("Residual Block")
        ax.set_yticks(np.arange(len(numeric_layers)))
        ax.set_yticklabels(
            [l.replace("layer_", "") for l in sorted(numeric_layers, key=_layer_sort_key)]
        )
        plt.colorbar(im, ax=ax, label="Effective Rank")
        ax.set_title("Channel eRank Heatmap")

    fig.suptitle(
        f"Low-Rank Analysis: 2D Ising Model ($L={L}$, $T_c \\approx {Tc:.3f}$)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "rank_vs_temperature.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_derank_dt(df, L, figs_dir, suffix=""):
    """
    Standalone figure comparing d(eRank)/dT with exact specific heat Cv.

    The effective rank is an entropy-like quantity, so its temperature
    derivative should peak at Tc, mirroring the specific heat Cv = dE/dT.
    Both are normalized to [0,1] for shape comparison.

    Args:
        suffix: filename suffix (e.g. "_critical") to distinguish modes.
    """
    from scipy.ndimage import gaussian_filter1d

    Tc = CRITICAL_TEMPERATURE

    # Average channel eRank across all layers
    avg_rank = df.groupby("T")["channel_erank"].mean().sort_index()
    T_arr = avg_rank.index.values
    rank_arr = avg_rank.values

    # Exact Cv over the same T range (dense for smooth curve)
    T_cv = np.linspace(T_arr.min(), T_arr.max(), 500)
    Cv = exact_specific_heat(L, T_cv)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Compute d(eRank)/dT — use lighter smoothing when data is dense
    d_rank_raw = np.gradient(rank_arr, T_arr)
    # Adapt sigma to grid density: fewer points → more smoothing
    sigma = max(1, min(3, len(T_arr) // 15))
    d_rank = gaussian_filter1d(d_rank_raw, sigma=sigma)

    # Normalize both for shape comparison
    d_norm = (d_rank - d_rank.min()) / (d_rank.max() - d_rank.min() + 1e-10)
    Cv_norm = (Cv - Cv.min()) / (Cv.max() - Cv.min() + 1e-10)

    ax.plot(T_arr, d_norm, "b-o", markersize=3, linewidth=1.5,
            label="$d$(eRank)/$dT$ (normalized)")
    ax.plot(T_cv, Cv_norm, "r-", linewidth=2, alpha=0.7,
            label="Exact $C_v$ (Onsager, normalized)")
    ax.axvline(Tc, color="gray", ls="--", alpha=0.5, lw=1,
               label=f"$T_c = {Tc:.3f}$")

    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel("Normalized value", fontsize=12)
    ax.set_title(
        f"$d$(eRank)/$dT$ vs Specific Heat $C_v$  ($L={L}$)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_xlim(T_arr.min() - 0.05, min(T_arr.max() + 0.05, 6.0))
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    path = Path(figs_dir) / f"derank_dt_vs_Cv{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_singular_value_spectra(
    model, selected_temperatures, device, batch_size, n_batches, L, figs_dir, console,
):
    """
    Scree plots at 3 representative temperatures: high-T, Tc, low-T.

    Shows how the singular value spectrum sharpens (low rank) or flattens
    (high rank) across the phase transition.
    """
    Tc = CRITICAL_TEMPERATURE
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, T_val in enumerate(selected_temperatures):
        ax = axes[idx]
        beta_val = 1.0 / T_val
        console.print(f"  SVD spectrum at T={T_val:.3f} (β={beta_val:.3f})")

        # Generate samples
        T_tensor = torch.full((batch_size,), T_val, device=device)
        all_samples = []
        with torch.no_grad():
            for _ in range(n_batches):
                all_samples.append(model.sample(batch_size=batch_size, T=T_tensor))
        all_samples = torch.cat(all_samples, dim=0)

        # Collect activations
        activations = collect_activations(model, all_samples, T_val, device)

        sorted_keys = sorted(
            activations.keys(),
            key=lambda x: (0, x) if isinstance(x, int) else (1, 0),
        )
        n_keys = len(sorted_keys)
        colors = LAYER_CMAP(np.linspace(0.15, 0.85, n_keys))

        for ci, lk in enumerate(sorted_keys):
            act = activations[lk]
            N, C, H, W = act.shape
            act_flat = act.mean(dim=(-2, -1))
            act_flat = act_flat - act_flat.mean(dim=0, keepdim=True)
            _, S, _ = torch.linalg.svd(act_flat, full_matrices=False)
            S_norm = S / S[0]

            label = f"Block {lk}" if isinstance(lk, int) else "Final"
            ax.semilogy(
                range(1, len(S_norm) + 1), S_norm.numpy(),
                color=colors[ci], alpha=0.75, linewidth=1.5, label=label,
            )

        phase = (
            "Disordered" if T_val > Tc * 1.1
            else "Ordered" if T_val < Tc * 0.9
            else "Critical"
        )
        ax.set_title(f"$T={T_val:.2f}$ ({phase})\n$\\beta={beta_val:.3f}$")
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("$\\sigma_i / \\sigma_1$")
        ax.legend(fontsize=6, ncol=2)
        ax.set_ylim(1e-4, 1.5)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Singular Value Spectra: Channel Activations ($L={L}$)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "singular_value_spectra.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_extended_metrics(df, L, figs_dir):
    """
    2×2 figure for the Nature Physics 2024 extended rank metrics:
      [0,0] Rényi Rank (α=2) vs T per layer
      [0,1] Optimal Hard Threshold (Gavish & Donoho) vs T per layer
      [1,0] All 5 rank metrics (normalized) overlaid
      [1,1] Metric correlation heatmap near Tc
    """
    if "channel_renyi2" not in df.columns:
        return None

    layers = sorted(df["layer"].unique(), key=_layer_sort_key)
    n_layers = len(layers)
    colors = LAYER_CMAP(np.linspace(0.15, 0.85, n_layers))
    Tc = CRITICAL_TEMPERATURE

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── [0,0] Rényi Rank (α=2) vs T ──
    ax = axes[0, 0]
    for ci, layer in enumerate(layers):
        ld = df[df["layer"] == layer].sort_values("T")
        ax.plot(
            ld["T"], ld["channel_renyi2"],
            "o-", color=colors[ci], markersize=3, linewidth=1.2,
            label=_layer_label(layer), alpha=0.85,
        )
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Rényi Rank ($\\alpha=2$)")
    ax.set_title("Rényi Rank (collision rank)")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.15)

    # ── [0,1] Optimal Hard Threshold (Gavish & Donoho) vs T ──
    ax = axes[0, 1]
    if "channel_opt_threshold" in df.columns:
        for ci, layer in enumerate(layers):
            ld = df[df["layer"] == layer].sort_values("T")
            ax.plot(
                ld["T"], ld["channel_opt_threshold"],
                "o-", color=colors[ci], markersize=3, linewidth=1.2,
                label=_layer_label(layer), alpha=0.85,
            )
        ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
        ax.set_xlabel("Temperature $T$")
        ax.set_ylabel("Rank (Gavish-Donoho)")
        ax.set_title("Optimal Hard Threshold (RMT)")
        ax.set_xscale("log")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.15)
    else:
        ax.set_visible(False)

    # ── [1,0] All rank metrics (normalized to [0,1]) overlaid ──
    ax = axes[1, 0]
    metric_cols = {
        "channel_erank": "eRank (Shannon)",
        "channel_stable_rank": "Stable Rank",
        "channel_nuclear_rank": "Nuclear Rank",
        "channel_pr": "Participation Ratio",
        "channel_renyi2": "Rényi Rank (α=2)",
        "channel_numerical_rank_99": "Numerical Rank (99%)",
        "channel_elbow_rank": "Elbow Rank",
        "channel_opt_threshold": "Opt. Threshold (G&D)",
    }
    # Filter to columns that exist in the DataFrame
    metric_cols = {k: v for k, v in metric_cols.items() if k in df.columns}
    n_metrics = len(metric_cols)
    metric_styles = ["o-", "s-", "^-", "D-", "v-", "P-", "X-", "h-"][:n_metrics]
    metric_colors = plt.cm.tab10(np.linspace(0, 0.7, n_metrics))

    avg_metrics = df.groupby("T")[list(metric_cols.keys())].mean().sort_index()
    T_arr = avg_metrics.index.values

    for (col, label), style, mc in zip(metric_cols.items(), metric_styles, metric_colors):
        vals = avg_metrics[col].values.astype(float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        norm_vals = (vals - vmin) / (vmax - vmin + 1e-10)
        ax.plot(
            T_arr, norm_vals,
            style, color=mc, markersize=3, linewidth=1.2,
            label=label, alpha=0.85,
        )
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Normalized rank [0, 1]")
    ax.set_title("All Rank Metrics (layer-averaged, normalized)")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    # ── [1,1] Metric correlation heatmap near Tc ──
    ax = axes[1, 1]
    near_Tc = df[(df["T"] > 0.8 * Tc) & (df["T"] < 1.2 * Tc)]
    corr_cols = [
        "channel_erank", "channel_stable_rank", "channel_nuclear_rank",
        "channel_pr", "channel_renyi2", "channel_numerical_rank_99",
        "channel_elbow_rank", "channel_opt_threshold",
    ]
    corr_labels = [
        "eRank", "sRank", "nRank", "PR", "Rényi₂", "NumRk₉₉",
        "Elbow", "Opt.Thr",
    ]
    available_cols = [c for c in corr_cols if c in near_Tc.columns]
    available_labels = [corr_labels[corr_cols.index(c)] for c in available_cols]

    if len(near_Tc) > 2 and len(available_cols) > 1:
        corr_data = near_Tc[available_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(corr_data) > 2:
            corr_matrix = corr_data.corr().values
            im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(available_labels)))
            ax.set_xticklabels(available_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(available_labels)))
            ax.set_yticklabels(available_labels, fontsize=7)
            for i in range(len(available_labels)):
                for j in range(len(available_labels)):
                    ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                            ha="center", va="center", fontsize=6,
                            color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
            plt.colorbar(im, ax=ax, label="Pearson $r$")
            ax.set_title(f"Metric Correlations near $T_c$ ({0.8*Tc:.2f}–{1.2*Tc:.2f})")
        else:
            ax.text(0.5, 0.5, "Insufficient data\nnear Tc",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title("Metric Correlations near $T_c$")
    else:
        ax.text(0.5, 0.5, "Insufficient data\nnear Tc",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title("Metric Correlations near $T_c$")

    fig.suptitle(
        f"Extended Rank Metrics: Nature Physics 2024 ($L={L}$, $T_c \\approx {Tc:.3f}$)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "extended_rank_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Low-Rank Analysis of PixelCNN Internal Representations"
    )
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--batch_size", type=int, default=200,
        help="Batch size for sample generation (default: 200)",
    )
    parser.add_argument(
        "--n_batches", type=int, default=3,
        help="Number of batches per temperature (default: 3)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer temperatures and smaller batches",
    )
    parser.add_argument(
        "--critical", action="store_true",
        help="Critical mode: dense uniform T grid around Tc for smooth d(eRank)/dT",
    )
    parser.add_argument(
        "--replot", type=str, default=None,
        help="Skip sampling; regenerate plots from existing CSV file",
    )
    args = parser.parse_args()

    console = Console()
    Tc = CRITICAL_TEMPERATURE

    console.print("[bold green]Low-Rank Analysis of PixelCNN Representations[/bold green]")
    console.print(f"Critical Temperature: $T_c$ = {Tc:.4f}")

    # ── Replot mode: skip sampling, just regenerate plots ──
    if args.replot:
        csv_path = Path(args.replot)
        if not csv_path.exists():
            console.print(f"[red]CSV not found:[/red] {csv_path}")
            return

        console.print(f"[yellow]Replot mode:[/yellow] reading {csv_path}")
        df = pd.read_csv(csv_path)

        # Infer L from group config
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

        fig_path = plot_rank_vs_temperature(df, L, figs_dir)
        console.print(f"[green]Main plot:[/green] {fig_path}")

        fig_path = plot_derank_dt(df, L, figs_dir)
        console.print(f"[green]d(eRank)/dT plot:[/green] {fig_path}")

        ext_path = plot_extended_metrics(df, L, figs_dir)
        if ext_path:
            console.print(f"[green]Extended metrics plot:[/green] {ext_path}")

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

    # Use pure PyTorch for MHC fusion (avoids mHC.cu CUDA kernel issues)
    if hasattr(model, 'use_pytorch_mhc'):
        model.use_pytorch_mhc()

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

    # ── Temperature grid ──
    mode_tag = "full"
    if args.critical:
        Tc = CRITICAL_TEMPERATURE
        # Dense uniform grid: smooth finite differences for d(eRank)/dT
        temps = np.linspace(0.5 * Tc, 2.0 * Tc, 60)
        batch_size, n_batches = args.batch_size, args.n_batches
        mode_tag = "critical"
        console.print(
            f"[magenta]Critical mode: {len(temps)} temps in "
            f"[{temps.min():.3f}, {temps.max():.3f}], "
            f"batch={batch_size}, {n_batches} batches[/magenta]"
        )
    elif args.quick:
        temps = temperature_grid(T_min=0.8, T_max=6.0, n_coarse=12, n_critical=8)
        batch_size, n_batches = 100, 2
        mode_tag = "quick"
        console.print("[yellow]Quick mode: 20 temps, batch=100, 2 batches[/yellow]")
    else:
        temps = temperature_grid(T_min=0.5, T_max=10.0, n_coarse=25, n_critical=15)
        batch_size, n_batches = args.batch_size, args.n_batches

    console.print(
        f"Temperature grid: {len(temps)} points, "
        f"T in [{temps.min():.3f}, {temps.max():.3f}]"
    )
    console.print(f"Samples per temperature: {batch_size * n_batches}")

    # ── Phase 1: Rank analysis ──
    console.print("\n[bold cyan]Phase 1: Effective rank analysis[/bold cyan]")
    df = run_analysis(model, temps, device, batch_size, n_batches, console)

    csv_suffix = f"_{mode_tag}" if mode_tag != "full" else ""
    csv_path = output_dir / f"rank_analysis{csv_suffix}_{seed}.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]Data saved:[/green] {csv_path}")

    # ── Phase 2: Plots ──
    console.print("\n[bold cyan]Phase 2: Generating plots[/bold cyan]")

    fig_path = plot_derank_dt(df, L, figs_dir, suffix=f"_{mode_tag}" if mode_tag != "full" else "")
    console.print(f"[green]d(eRank)/dT plot:[/green] {fig_path}")

    ext_path = plot_extended_metrics(df, L, figs_dir)
    if ext_path:
        console.print(f"[green]Extended metrics plot:[/green] {ext_path}")

    if mode_tag != "critical":
        # Full/quick mode: also generate 3x2 overview and scree plots
        fig_path = plot_rank_vs_temperature(df, L, figs_dir)
        console.print(f"[green]Main plot:[/green] {fig_path}")

        # Select 3 representative temperatures for scree plots
        selected_T = [
            temps[temps > 2.0 * Tc].min(),    # high T (disordered)
            temps[np.argmin(np.abs(temps - Tc))],  # near Tc
            temps[temps < 0.6 * Tc].max(),    # low T (ordered)
        ]
        console.print(
            f"Scree plot temperatures: "
            f"{', '.join(f'T={t:.3f}' for t in selected_T)}"
        )

        fig_path = plot_singular_value_spectra(
            model, selected_T, device, batch_size, n_batches, L, figs_dir, console,
        )
        console.print(f"[green]Scree plot:[/green] {fig_path}")

    # ── Summary ──
    console.print("\n[bold cyan]Summary: Peak effective rank by layer[/bold cyan]")
    console.print(f"{'Layer':>10s}  {'Peak eRank':>10s}  {'at T':>7s}  "
                  f"{'T/Tc':>6s}  {'Min eRank':>9s}  {'Ratio':>6s}")
    console.print("-" * 60)

    for layer in sorted(df["layer"].unique(), key=_layer_sort_key):
        ld = df[df["layer"] == layer]
        peak_idx = ld["channel_erank"].idxmax()
        peak_T = ld.loc[peak_idx, "T"]
        peak_rank = ld.loc[peak_idx, "channel_erank"]
        min_rank = ld["channel_erank"].min()
        ratio = peak_rank / max(min_rank, 1e-6)

        console.print(
            f"{_layer_label(layer):>10s}  {peak_rank:>10.2f}  {peak_T:>7.3f}  "
            f"{peak_T / Tc:>6.3f}  {min_rank:>9.2f}  {ratio:>6.2f}x"
        )

    # Extended metrics summary (if available)
    if "channel_renyi2" in df.columns:
        console.print("\n[bold cyan]Extended Metrics at T nearest Tc[/bold cyan]")
        T_nearest_Tc = df.iloc[(df["T"] - Tc).abs().argsort().iloc[0]]["T"]
        near_df = df[df["T"] == T_nearest_Tc]
        console.print(f"T = {T_nearest_Tc:.4f} (T/Tc = {T_nearest_Tc / Tc:.4f})")
        console.print(
            f"{'Layer':>10s}  {'Rényi₂':>8s}  {'NumRk99':>8s}  "
            f"{'nRank':>7s}  {'Elbow':>6s}  {'OptThr':>7s}"
        )
        console.print("-" * 56)
        for layer in sorted(near_df["layer"].unique(), key=_layer_sort_key):
            row = near_df[near_df["layer"] == layer].iloc[0]
            console.print(
                f"{_layer_label(layer):>10s}  {row['channel_renyi2']:>8.2f}  "
                f"{int(row['channel_numerical_rank_99']):>8d}  "
                f"{row['channel_nuclear_rank']:>7.2f}"
                f"  {int(row['channel_elbow_rank']):>6d}"
                f"  {int(row['channel_opt_threshold']):>7d}"
            )

    console.print(f"\n[bold green]Analysis complete.[/bold green]")
    console.print(f"Figures: {figs_dir}/")
    console.print(f"Data:    {csv_path}")


if __name__ == "__main__":
    main()
