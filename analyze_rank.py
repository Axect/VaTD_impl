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


def get_critical_temperature(config):
    """Determine critical temperature from model config (Ising, Potts, or Clock)."""
    net_config = config.net_config if hasattr(config, 'net_config') else {}
    q = net_config.get("category", 2)
    model_type = net_config.get("model_type", "")
    if model_type == "clock":
        from clock import CLOCK_TC
        return CLOCK_TC.get(q, 0.89), q
    return get_critical_temperature_from_q(q)


def get_critical_temperature_from_q(q):
    """Get critical temperature from number of states q."""
    if q == 2:
        from vatd_exact_partition import CRITICAL_TEMPERATURE
        return CRITICAL_TEMPERATURE, q
    else:
        from potts_exact_partition import critical_temperature
        return critical_temperature(q), q


# ──────────────────────────────────────────────────────────────
# Rank Metrics (unified two-parameter Rényi-normalization framework)
# ──────────────────────────────────────────────────────────────
from unified_rank_metrics import (
    # Core unified function
    renyi_effective_rank,
    # Named grid wrappers (α, norm) → metric
    effective_rank,           # (α=1,  L1)   = eRank
    von_neumann_effective_rank,  # (α=1,  L2sq) = exp(S_vN)
    stable_rank,              # (α=∞,  L2sq)
    participation_ratio,      # (α=2,  L2sq)
    nuclear_rank,             # (α=∞,  L1)
    renyi2_rank,              # (α=2,  L1)
    # Raw entropies (no exponentiation)
    shannon_entropy,
    von_neumann_entropy,
    # Standalone metrics (not in grid)
    numerical_rank,
    elbow_rank,
    optimal_hard_threshold,
    spectral_gap_ratio,
    # RMT spectral unfolding (BBP outlier detection)
    marchenko_pastur_outlier_count,
    # Full grid computation
    compute_full_grid,
    compute_entropy_grid,
)

# Backward-compatible alias for existing code
def renyi_rank(singular_values, alpha=2.0):
    return renyi_effective_rank(singular_values, alpha=alpha, norm="L1")


# ──────────────────────────────────────────────────────────────
# Exact Specific Heat (Onsager)
# ──────────────────────────────────────────────────────────────


def exact_specific_heat(L, temperatures, J=1.0, q=2):
    """
    Exact specific heat per site from Onsager's partition function.

    Only available for q=2 (Ising). Returns None for Potts (q > 2).

    Cv/N = β²/N · ∂²logZ/∂β²
    Computed via central finite differences of exact logZ.
    """
    if q > 2:
        return None

    from vatd_exact_partition import logZ as exact_logZ

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
    Collect intermediate activations from all blocks using forward hooks.

    Dispatches to model-specific collector based on architecture:
      - PixelCNN: hooks on masked_conv.hidden_convs[i] and first_fc
      - Transformer (LatticeGPT): hooks on backbone.blocks[i] output

    Args:
        model: DiscretePixelCNN or LatticeGPT (eval mode)
        samples: [B, 1, H, W] in physical representation
        T_val: scalar temperature
        device: torch device

    Returns:
        dict {layer_idx: Tensor[B, C, H, W], 'final': Tensor[B, C, H, W]}
    """
    if hasattr(model, 'backbone'):
        return _collect_activations_transformer(model, samples, T_val, device)
    else:
        return _collect_activations_pixelcnn(model, samples, T_val, device)


def _collect_activations_pixelcnn(model, samples, T_val, device):
    """PixelCNN: pre-hooks on hidden_convs[i] and first_fc."""
    captured = {}
    hooks = []

    for i, conv in enumerate(model.masked_conv.hidden_convs):

        def make_hook(idx):
            def hook_fn(module, inp):
                captured[idx] = inp[0].detach().cpu()

            return hook_fn

        hooks.append(conv.register_forward_pre_hook(make_hook(i)))

    def final_hook(module, inp):
        captured["final"] = inp[0].detach().cpu()

    hooks.append(model.masked_conv.first_fc.register_forward_pre_hook(final_hook))

    T_tensor = torch.full((samples.shape[0],), T_val, device=device)
    with torch.no_grad():
        model.log_prob(samples, T=T_tensor)

    for h in hooks:
        h.remove()

    return captured


def _collect_activations_transformer(model, samples, T_val, device):
    """
    Transformer (LatticeGPT): hooks on backbone.blocks[i] output.

    Captures [B, L, d_model] and reshapes to [B, d_model, H, W]
    for compatibility with the SVD analysis pipeline.
    """
    captured = {}
    hooks = []
    H, W = model.size

    for i, block in enumerate(model.backbone.blocks):

        def make_hook(idx):
            def hook_fn(module, inp, out):
                # out may be (tensor, kv_cache) tuple if block returns KV-cache
                act = out[0] if isinstance(out, tuple) else out
                act = act.detach().cpu()
                B, L, D = act.shape
                captured[idx] = act.permute(0, 2, 1).reshape(B, D, H, W)

            return hook_fn

        hooks.append(block.register_forward_hook(make_hook(i)))

    # Final: after backbone.final_norm
    def final_hook(module, inp, out):
        act = out.detach().cpu()
        B, L, D = act.shape
        captured["final"] = act.permute(0, 2, 1).reshape(B, D, H, W)

    hooks.append(model.backbone.final_norm.register_forward_hook(final_hook))

    T_tensor = torch.full((samples.shape[0],), T_val, device=device)
    with torch.no_grad():
        model.log_prob(samples, T=T_tensor)

    for h in hooks:
        h.remove()

    return captured


# ──────────────────────────────────────────────────────────────
# Temperature Grid
# ──────────────────────────────────────────────────────────────


def temperature_grid(T_min=0.5, T_max=10.0, n_coarse=25, n_critical=15, Tc=None):
    """
    Generate temperature grid with extra density near Tc.

    Merges a log-spaced coarse grid with a linear dense grid
    around [0.8·Tc, 1.2·Tc].
    """
    if Tc is None:
        from vatd_exact_partition import CRITICAL_TEMPERATURE
        Tc = CRITICAL_TEMPERATURE
    T_coarse = np.logspace(np.log10(T_min), np.log10(T_max), n_coarse)
    T_dense = np.linspace(0.8 * Tc, 1.2 * Tc, n_critical)
    T_all = np.unique(np.concatenate([T_coarse, T_dense]))
    T_all.sort()
    return T_all


# ──────────────────────────────────────────────────────────────
# Main Analysis Loop
# ──────────────────────────────────────────────────────────────


def run_analysis(model, temperatures, device, batch_size=200, n_batches=3, console=None, Tc=None):
    """
    For each temperature: generate samples → collect activations → compute rank metrics.

    Returns a DataFrame with columns:
        T, beta, T_over_Tc, layer,
        channel_erank, channel_vn_erank, channel_stable_rank, channel_pr,
        channel_numerical_rank_99, channel_numerical_rank_95,
        channel_renyi2,
        channel_nuclear_rank, channel_elbow_rank, channel_opt_threshold,
        channel_H_shannon, channel_S_vN, channel_sgr_max, channel_sgr_k3,
        channel_mp_outliers,
        spatial_erank, spatial_vn_erank, spatial_stable_rank, spatial_pr,
        spatial_numerical_rank_99, spatial_numerical_rank_95,
        spatial_renyi2,
        spatial_nuclear_rank, spatial_elbow_rank, spatial_opt_threshold,
        spatial_H_shannon, spatial_S_vN, spatial_sgr_max, spatial_sgr_k3,
        spatial_mp_outliers
    """
    if console is None:
        console = Console()

    if hasattr(model, 'backbone'):
        # Transformer (LatticeGPT): blocks in backbone
        num_layers = len(model.backbone.blocks)
    else:
        # PixelCNN: hidden conv blocks
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
                        "T_over_Tc": T_val / Tc if Tc else T_val,
                        "layer": layer_name,
                        # ── Channel metrics (unified framework) ──
                        "channel_erank": effective_rank(S_ch),
                        "channel_vn_erank": von_neumann_effective_rank(S_ch),
                        "channel_stable_rank": stable_rank(S_ch),
                        "channel_pr": participation_ratio(S_ch),
                        "channel_numerical_rank_99": numerical_rank(S_ch, 0.99),
                        "channel_numerical_rank_95": numerical_rank(S_ch, 0.95),
                        "channel_renyi2": renyi2_rank(S_ch),
                        "channel_nuclear_rank": nuclear_rank(S_ch),
                        "channel_elbow_rank": elbow_rank(S_ch),
                        "channel_opt_threshold": optimal_hard_threshold(S_ch, N, C),
                        "channel_H_shannon": shannon_entropy(S_ch, norm="L1"),
                        "channel_S_vN": von_neumann_entropy(S_ch),
                        "channel_sgr_max": spectral_gap_ratio(S_ch, k=None),
                        "channel_sgr_k3": spectral_gap_ratio(S_ch, k=3),
                        "channel_mp_outliers": marchenko_pastur_outlier_count(S_ch, N, C),
                        # ── Spatial metrics (unified framework) ──
                        "spatial_erank": effective_rank(S_sp),
                        "spatial_vn_erank": von_neumann_effective_rank(S_sp),
                        "spatial_stable_rank": stable_rank(S_sp),
                        "spatial_pr": participation_ratio(S_sp),
                        "spatial_numerical_rank_99": numerical_rank(S_sp, 0.99),
                        "spatial_numerical_rank_95": numerical_rank(S_sp, 0.95),
                        "spatial_renyi2": renyi2_rank(S_sp),
                        "spatial_nuclear_rank": nuclear_rank(S_sp),
                        "spatial_elbow_rank": elbow_rank(S_sp),
                        "spatial_opt_threshold": optimal_hard_threshold(S_sp, N, H * W),
                        "spatial_H_shannon": shannon_entropy(S_sp, norm="L1"),
                        "spatial_S_vN": von_neumann_entropy(S_sp),
                        "spatial_sgr_max": spectral_gap_ratio(S_sp, k=None),
                        "spatial_sgr_k3": spectral_gap_ratio(S_sp, k=3),
                        "spatial_mp_outliers": marchenko_pastur_outlier_count(S_sp, N, H * W),
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


def plot_rank_vs_temperature(df, L, figs_dir, Tc=None, q=2, model_type=""):
    """
    3×2 figure:
      [0,0] Channel effective rank vs T per layer
      [0,1] Spatial effective rank vs T per layer
      [1,0] Numerical Rank vs T (99% and 95% thresholds)
      [1,1] Nuclear Rank vs T per layer
      [2,0] Avg channel erank vs T + exact Cv overlay
      [2,1] Heatmap of channel erank [layer × T]
    """
    if Tc is None:
        from vatd_exact_partition import CRITICAL_TEMPERATURE
        Tc = CRITICAL_TEMPERATURE

    temperatures = np.sort(df["T"].unique())
    Cv = exact_specific_heat(L, temperatures, q=q)

    layers = sorted(df["layer"].unique(), key=_layer_sort_key)
    n_layers = len(layers)
    colors = LAYER_CMAP(np.linspace(0.15, 0.85, n_layers))

    has_extended = "channel_numerical_rank_99" in df.columns

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

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

    if Cv is not None:
        ax2 = ax.twinx()
        ax2.plot(
            temperatures, Cv,
            "r-", linewidth=2, alpha=0.6, label="Exact $C_v$ (Onsager)",
        )
        ax2.set_ylabel("$C_v$ / site", color="red")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    else:
        ax.legend(fontsize=8, loc="upper left")
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

    if q == 2:
        model_label = "2D Ising"
    elif model_type == "clock":
        model_label = f"{q}-state Clock"
    else:
        model_label = f"{q}-state Potts"
    fig.suptitle(
        f"Low-Rank Analysis: {model_label} ($L={L}$, $T_c \\approx {Tc:.3f}$)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "rank_vs_temperature.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_derank_dt(df, L, figs_dir, suffix="", Tc=None, q=2, model_type=""):
    """
    Standalone figure comparing d(eRank)/dT with exact specific heat Cv.

    The effective rank is an entropy-like quantity, so its temperature
    derivative should peak at Tc, mirroring the specific heat Cv = dE/dT.
    Both are normalized to [0,1] for shape comparison.

    Args:
        suffix: filename suffix (e.g. "_critical") to distinguish modes.
        Tc: critical temperature (auto-detected if None)
        q: number of states (2=Ising, 3/4=Potts)
    """
    from scipy.ndimage import gaussian_filter1d

    if Tc is None:
        from vatd_exact_partition import CRITICAL_TEMPERATURE
        Tc = CRITICAL_TEMPERATURE

    # Average channel eRank across all layers
    avg_rank = df.groupby("T")["channel_erank"].mean().sort_index()
    T_arr = avg_rank.index.values
    rank_arr = avg_rank.values

    # Exact Cv over the same T range (dense for smooth curve) — Ising only
    T_cv = np.linspace(T_arr.min(), T_arr.max(), 500)
    Cv = exact_specific_heat(L, T_cv, q=q)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Compute d(eRank)/dT — use lighter smoothing when data is dense
    d_rank_raw = np.gradient(rank_arr, T_arr)
    # Adapt sigma to grid density: fewer points → more smoothing
    sigma = max(1, min(3, len(T_arr) // 15))
    d_rank = gaussian_filter1d(d_rank_raw, sigma=sigma)

    # Normalize d(eRank)/dT
    d_norm = (d_rank - d_rank.min()) / (d_rank.max() - d_rank.min() + 1e-10)

    ax.plot(T_arr, d_norm, "b-o", markersize=3, linewidth=1.5,
            label="$d$(eRank)/$dT$ (normalized)")

    if Cv is not None:
        Cv_norm = (Cv - Cv.min()) / (Cv.max() - Cv.min() + 1e-10)
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
    Tc=None,
):
    """
    Scree plots at 3 representative temperatures: high-T, Tc, low-T.

    Shows how the singular value spectrum sharpens (low rank) or flattens
    (high rank) across the phase transition.
    """
    if Tc is None:
        from vatd_exact_partition import CRITICAL_TEMPERATURE
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


def plot_extended_metrics(df, L, figs_dir, Tc=None, q=2, model_type=""):
    """
    2×2 figure for the Nature Physics 2024 extended rank metrics:
      [0,0] Rényi Rank (α=2) vs T per layer
      [0,1] Optimal Hard Threshold (Gavish & Donoho) vs T per layer
      [1,0] All 5 rank metrics (normalized) overlaid
      [1,1] Metric correlation heatmap near Tc
    """
    if "channel_renyi2" not in df.columns:
        return None

    if Tc is None:
        from vatd_exact_partition import CRITICAL_TEMPERATURE
        Tc = CRITICAL_TEMPERATURE

    layers = sorted(df["layer"].unique(), key=_layer_sort_key)
    n_layers = len(layers)
    colors = LAYER_CMAP(np.linspace(0.15, 0.85, n_layers))

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
        "channel_erank": "eRank (α=1, L1)",
        "channel_vn_erank": "vN-eRank (α=1, L2²)",
        "channel_stable_rank": "Stable Rank (α=∞, L2²)",
        "channel_nuclear_rank": "Nuclear Rank (α=∞, L1)",
        "channel_pr": "PR (α=2, L2²)",
        "channel_renyi2": "Rényi-2 (α=2, L1)",
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
        "channel_erank", "channel_vn_erank", "channel_stable_rank",
        "channel_nuclear_rank", "channel_pr", "channel_renyi2",
        "channel_numerical_rank_99", "channel_elbow_rank",
        "channel_opt_threshold",
    ]
    corr_labels = [
        "eRank", "vN-eR", "sRank", "nRank", "PR", "Rényi₂",
        "NumRk₉₉", "Elbow", "Opt.Thr",
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

    if q == 2:
        model_label = "2D Ising"
    elif model_type == "clock":
        model_label = f"{q}-state Clock"
    else:
        model_label = f"{q}-state Potts"
    fig.suptitle(
        f"Extended Rank Metrics: {model_label} ($L={L}$, $T_c \\approx {Tc:.3f}$)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "extended_rank_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_unified_framework(df, L, figs_dir, Tc=None, q=2, model_type=""):
    """
    3×2 figure for the unified Rényi-normalization framework metrics:
      [0,0] vN-eRank (α=1, L2²) vs T per layer
      [0,1] eRank vs vN-eRank scatter colored by T/Tc
      [1,0] Shannon entropy H(p) and von Neumann entropy S_vN vs T
      [1,1] Entropy gap: H(p) - S_vN vs T (norm sensitivity)
      [2,0] Spectral Gap Ratio (max & k=3) vs T per layer
      [2,1] Rank gap ΔR = eRank - vN-eRank vs T per layer
    """
    if "channel_vn_erank" not in df.columns:
        return None

    if Tc is None:
        from vatd_exact_partition import CRITICAL_TEMPERATURE
        Tc = CRITICAL_TEMPERATURE

    layers = sorted(df["layer"].unique(), key=_layer_sort_key)
    n_layers = len(layers)
    colors = LAYER_CMAP(np.linspace(0.15, 0.85, n_layers))

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # ── [0,0] vN-eRank vs T per layer ──
    ax = axes[0, 0]
    for ci, layer in enumerate(layers):
        ld = df[df["layer"] == layer].sort_values("T")
        ax.plot(
            ld["T"], ld["channel_vn_erank"],
            "o-", color=colors[ci], markersize=3, linewidth=1.2,
            label=_layer_label(layer), alpha=0.85,
        )
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("vN-eRank $\\exp(S_{\\mathrm{vN}})$")
    ax.set_title("von Neumann Effective Rank ($\\alpha=1$, $L^2$ norm)")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.15)

    # ── [0,1] eRank vs vN-eRank scatter colored by T/Tc ──
    ax = axes[0, 1]
    sc = ax.scatter(
        df["channel_erank"], df["channel_vn_erank"],
        c=df["T_over_Tc"], cmap="coolwarm", s=15, alpha=0.7,
        edgecolors="none", vmin=0.5, vmax=2.0,
    )
    plt.colorbar(sc, ax=ax, label="$T / T_c$")
    # Reference line: eRank = vN-eRank (equality for uniform spectrum)
    lim_max = max(df["channel_erank"].max(), df["channel_vn_erank"].max()) * 1.05
    ax.plot([1, lim_max], [1, lim_max], "k--", alpha=0.3, lw=0.8, label="$y = x$")
    ax.set_xlabel("eRank ($\\alpha=1$, $L^1$)")
    ax.set_ylabel("vN-eRank ($\\alpha=1$, $L^2$)")
    ax.set_title("eRank vs vN-eRank (norm comparison)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    # ── [1,0] Raw entropies vs T (layer-averaged) ──
    ax = axes[1, 0]
    if "channel_H_shannon" in df.columns and "channel_S_vN" in df.columns:
        avg_H = df.groupby("T")["channel_H_shannon"].mean().sort_index()
        avg_SvN = df.groupby("T")["channel_S_vN"].mean().sort_index()
        T_arr = avg_H.index.values

        ax.plot(T_arr, avg_H.values, "b-o", markersize=3, linewidth=1.5,
                label="$H(p)$ (Shannon, $L^1$)")
        ax.plot(T_arr, avg_SvN.values, "r-s", markersize=3, linewidth=1.5,
                label="$S_{\\mathrm{vN}}$ (von Neumann, $L^2$)")
        ax.axvline(Tc, color="gray", ls="--", alpha=0.4, lw=0.8)
        ax.set_xlabel("Temperature $T$")
        ax.set_ylabel("Entropy (nats)")
        ax.set_title("Raw Spectral Entropies (layer-averaged)")
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)
    else:
        ax.set_visible(False)

    # ── [1,1] Entropy gap: H(p) - S_vN vs T ──
    ax = axes[1, 1]
    if "channel_H_shannon" in df.columns and "channel_S_vN" in df.columns:
        avg_gap = (
            df.groupby("T")["channel_H_shannon"].mean()
            - df.groupby("T")["channel_S_vN"].mean()
        ).sort_index()
        T_arr = avg_gap.index.values

        ax.plot(T_arr, avg_gap.values, "g-D", markersize=3, linewidth=1.5,
                label="$\\Delta H = H(p) - S_{\\mathrm{vN}}$")
        ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
        ax.axhline(0, color="gray", ls=":", alpha=0.3)
        ax.set_xlabel("Temperature $T$")
        ax.set_ylabel("$\\Delta H$ (nats)")
        ax.set_title("Entropy Gap: $L^1$ vs $L^2$ Normalization")
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)
    else:
        ax.set_visible(False)

    # ── [2,0] Spectral Gap Ratio vs T per layer ──
    ax = axes[2, 0]
    if "channel_sgr_max" in df.columns:
        for ci, layer in enumerate(layers):
            ld = df[df["layer"] == layer].sort_values("T")
            ax.plot(
                ld["T"], ld["channel_sgr_max"],
                "o-", color=colors[ci], markersize=3, linewidth=1.2,
                label=_layer_label(layer), alpha=0.85,
            )
        ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
        ax.set_xlabel("Temperature $T$")
        ax.set_ylabel("SGR (max gap)")
        ax.set_title("Spectral Gap Ratio (max over $k$)")
        ax.set_xscale("log")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.15)
    else:
        ax.set_visible(False)

    # ── [2,1] Rank gap ΔR = eRank - vN-eRank vs T per layer ──
    ax = axes[2, 1]
    for ci, layer in enumerate(layers):
        ld = df[df["layer"] == layer].sort_values("T")
        delta_r = ld["channel_erank"].values - ld["channel_vn_erank"].values
        ax.plot(
            ld["T"].values, delta_r,
            "o-", color=colors[ci], markersize=3, linewidth=1.2,
            label=_layer_label(layer), alpha=0.85,
        )
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.axhline(0, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("$\\Delta R$ = eRank $-$ vN-eRank")
    ax.set_title("Norm Gap: $L^1$ vs $L^2$ Rank")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.15)

    if q == 2:
        model_label = "2D Ising"
    elif model_type == "clock":
        model_label = f"{q}-state Clock"
    else:
        model_label = f"{q}-state Potts"
    fig.suptitle(
        f"Unified Rényi-Normalization Framework: {model_label} ($L={L}$, $T_c \\approx {Tc:.3f}$)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "unified_framework_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# CFT relevant operator counts per q-state model
CFT_OPERATOR_COUNT = {2: 3, 3: 6, 4: 8}


def plot_mp_outliers(df, L, figs_dir, Tc=None, q=2, model_type=""):
    """
    Marchenko-Pastur outlier count vs temperature.

    SVs exceeding the BBP threshold are RMT outliers — genuine signal
    components.  At Tc the count should match the number of relevant
    CFT operators: 3 (Ising), 6 (3-Potts), 8 (4-Potts).

    Left:  per-layer MP outlier count vs T
    Right: layer-averaged MP outlier count vs T

    Args:
        df: DataFrame from run_analysis() with 'channel_mp_outliers' column.
        L: lattice size.
        figs_dir: output directory for the figure.
        Tc: critical temperature (auto-detected if None).
        q: number of states (2=Ising, 3/4=Potts).

    Returns:
        Path to saved figure, or None if the column is missing.
    """
    if "channel_mp_outliers" not in df.columns:
        return None

    if Tc is None:
        from vatd_exact_partition import CRITICAL_TEMPERATURE
        Tc = CRITICAL_TEMPERATURE

    layers = sorted(df["layer"].unique(), key=_layer_sort_key)
    n_layers = len(layers)
    colors = LAYER_CMAP(np.linspace(0.15, 0.85, n_layers))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: per-layer MP outlier count vs T ──
    ax = axes[0]
    for ci, layer in enumerate(layers):
        ld = df[df["layer"] == layer].sort_values("T")
        ax.plot(
            ld["T"], ld["channel_mp_outliers"],
            "o-", color=colors[ci], markersize=3, linewidth=1.2,
            label=_layer_label(layer), alpha=0.85,
        )
    ax.axvline(Tc, color="red", ls="--", alpha=0.5, lw=1.0,
               label=f"$T_c = {Tc:.3f}$")
    n_cft = CFT_OPERATOR_COUNT.get(q)
    if n_cft is not None:
        ax.axhline(n_cft, color="green", ls=":", alpha=0.6, lw=1.5,
                    label=f"CFT prediction ({n_cft} operators)")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("MP Outlier Count (integer)")
    ax.set_title("Per-Layer MP Outlier Count")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.15)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Right: layer-averaged MP outlier count vs T ──
    ax = axes[1]
    avg_outliers = df.groupby("T")["channel_mp_outliers"].mean().sort_index()
    T_arr = avg_outliers.index.values
    outlier_arr = avg_outliers.values

    ax.plot(T_arr, outlier_arr, "b-o", markersize=4, linewidth=2,
            label="Avg MP Outlier Count")
    ax.axvline(Tc, color="red", ls="--", alpha=0.5, lw=1.0,
               label=f"$T_c = {Tc:.3f}$")
    if n_cft is not None:
        ax.axhline(n_cft, color="green", ls=":", alpha=0.6, lw=1.5,
                    label=f"CFT: {n_cft} relevant operators")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("MP Outlier Count (layer-averaged)")
    ax.set_title("Layer-Averaged MP Outlier Count")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    if q == 2:
        model_label = "2D Ising"
    elif model_type == "clock":
        model_label = f"{q}-state Clock"
    else:
        model_label = f"{q}-state Potts"
    fig.suptitle(
        f"RMT Spectral Unfolding: {model_label} ($L={L}$, $T_c \\approx {Tc:.3f}$)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "mp_outlier_count.png"
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

    console.print("[bold green]Low-Rank Analysis of PixelCNN Representations[/bold green]")

    # ── Replot mode: skip sampling, just regenerate plots ──
    if args.replot:
        csv_path = Path(args.replot)
        if not csv_path.exists():
            console.print(f"[red]CSV not found:[/red] {csv_path}")
            return

        console.print(f"[yellow]Replot mode:[/yellow] reading {csv_path}")
        df = pd.read_csv(csv_path)

        # Infer L and q from group config
        config_path = csv_path.parent / "config.yaml"
        q = 2
        model_type = ""
        if config_path.exists():
            from config import RunConfig
            config = RunConfig.from_yaml(str(config_path))
            L = config.net_config.get("size", 16)
            if not isinstance(L, int):
                L = L[0]
            q = config.net_config.get("category", 2)
            model_type = config.net_config.get("model_type", "")
        else:
            L = 16

        if model_type == "clock":
            from clock import CLOCK_TC
            Tc = CLOCK_TC.get(q, 0.89)
        else:
            Tc, _ = get_critical_temperature_from_q(q)

        figs_dir = Path(f"figs/{csv_path.parent.name}")
        figs_dir.mkdir(parents=True, exist_ok=True)

        fig_path = plot_rank_vs_temperature(df, L, figs_dir, Tc=Tc, q=q, model_type=model_type)
        console.print(f"[green]Main plot:[/green] {fig_path}")

        fig_path = plot_derank_dt(df, L, figs_dir, Tc=Tc, q=q, model_type=model_type)
        console.print(f"[green]d(eRank)/dT plot:[/green] {fig_path}")

        ext_path = plot_extended_metrics(df, L, figs_dir, Tc=Tc, q=q, model_type=model_type)
        if ext_path:
            console.print(f"[green]Extended metrics plot:[/green] {ext_path}")

        uni_path = plot_unified_framework(df, L, figs_dir, Tc=Tc, q=q, model_type=model_type)
        if uni_path:
            console.print(f"[green]Unified framework plot:[/green] {uni_path}")

        mp_path = plot_mp_outliers(df, L, figs_dir, Tc=Tc, q=q, model_type=model_type)
        if mp_path:
            console.print(f"[green]MP outlier plot:[/green] {mp_path}")

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

    # Determine model type and critical temperature
    net_config = config.net_config if hasattr(config, 'net_config') else {}
    model_type = net_config.get("model_type", "")
    Tc, q = get_critical_temperature(config)
    if q == 2:
        model_label = "2D Ising"
    elif model_type == "clock":
        model_label = f"{q}-state Clock"
    else:
        model_label = f"{q}-state Potts"
    console.print(f"Model type: {model_label}")
    console.print(f"Critical Temperature: Tc = {Tc:.4f}")

    # Use pure PyTorch for MHC fusion (avoids mHC.cu CUDA kernel issues)
    if hasattr(model, 'use_pytorch_mhc'):
        model.use_pytorch_mhc()

    L = model.size[0]
    if hasattr(model, 'backbone'):
        num_layers = len(model.backbone.blocks)
        arch_info = f"Transformer blocks: {num_layers}, d_model: {model.hparams.get('d_model', '?')}"
    else:
        num_layers = len(model.masked_conv.hidden_convs)
        arch_info = f"Residual blocks: {num_layers}, Hidden channels: {model.masked_conv.hidden_channels}"
    num_params = sum(p.numel() for p in model.parameters())
    console.print(
        f"Lattice: {L}x{L}, Params: {num_params:,}, {arch_info}"
    )

    # ── Output directories ──
    figs_dir = Path(f"figs/{group_name}")
    figs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(f"runs/{project}/{group_name}")

    # ── Temperature grid ──
    mode_tag = "full"
    if args.critical:
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
        temps = temperature_grid(T_min=0.8, T_max=6.0, n_coarse=12, n_critical=8, Tc=Tc)
        batch_size, n_batches = 100, 2
        mode_tag = "quick"
        console.print("[yellow]Quick mode: 20 temps, batch=100, 2 batches[/yellow]")
    else:
        temps = temperature_grid(T_min=0.5, T_max=10.0, n_coarse=25, n_critical=15, Tc=Tc)
        batch_size, n_batches = args.batch_size, args.n_batches

    console.print(
        f"Temperature grid: {len(temps)} points, "
        f"T in [{temps.min():.3f}, {temps.max():.3f}]"
    )
    console.print(f"Samples per temperature: {batch_size * n_batches}")

    # ── Phase 1: Rank analysis ──
    console.print("\n[bold cyan]Phase 1: Effective rank analysis[/bold cyan]")
    df = run_analysis(model, temps, device, batch_size, n_batches, console, Tc=Tc)

    csv_suffix = f"_{mode_tag}" if mode_tag != "full" else ""
    csv_path = output_dir / f"rank_analysis{csv_suffix}_{seed}.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]Data saved:[/green] {csv_path}")

    # ── Phase 2: Plots ──
    console.print("\n[bold cyan]Phase 2: Generating plots[/bold cyan]")

    fig_path = plot_derank_dt(df, L, figs_dir, suffix=f"_{mode_tag}" if mode_tag != "full" else "", Tc=Tc, q=q, model_type=model_type)
    console.print(f"[green]d(eRank)/dT plot:[/green] {fig_path}")

    ext_path = plot_extended_metrics(df, L, figs_dir, Tc=Tc, q=q, model_type=model_type)
    if ext_path:
        console.print(f"[green]Extended metrics plot:[/green] {ext_path}")

    uni_path = plot_unified_framework(df, L, figs_dir, Tc=Tc, q=q, model_type=model_type)
    if uni_path:
        console.print(f"[green]Unified framework plot:[/green] {uni_path}")

    mp_path = plot_mp_outliers(df, L, figs_dir, Tc=Tc, q=q, model_type=model_type)
    if mp_path:
        console.print(f"[green]MP outlier plot:[/green] {mp_path}")

    if mode_tag != "critical":
        # Full/quick mode: also generate 3x2 overview and scree plots
        fig_path = plot_rank_vs_temperature(df, L, figs_dir, Tc=Tc, q=q, model_type=model_type)
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
            Tc=Tc,
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
            f"{'Layer':>10s}  {'eRank':>8s}  {'vN-eR':>8s}  {'Rényi₂':>8s}  "
            f"{'nRank':>7s}  {'sRank':>7s}  {'OptThr':>7s}"
        )
        console.print("-" * 68)
        for layer in sorted(near_df["layer"].unique(), key=_layer_sort_key):
            row = near_df[near_df["layer"] == layer].iloc[0]
            vn_erank = row.get('channel_vn_erank', float('nan'))
            console.print(
                f"{_layer_label(layer):>10s}  {row['channel_erank']:>8.2f}  "
                f"{vn_erank:>8.2f}  "
                f"{row['channel_renyi2']:>8.2f}  "
                f"{row['channel_nuclear_rank']:>7.2f}"
                f"  {row['channel_stable_rank']:>7.2f}"
                f"  {int(row['channel_opt_threshold']):>7d}"
            )

    console.print(f"\n[bold green]Analysis complete.[/bold green]")
    console.print(f"Figures: {figs_dir}/")
    console.print(f"Data:    {csv_path}")


if __name__ == "__main__":
    main()
