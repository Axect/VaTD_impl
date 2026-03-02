"""
Kernel PCA Low-Rank Analysis of DiscretePixelCNN Internal Representations.

Extends the linear SVD analysis (analyze_rank.py) with nonlinear Kernel PCA
metrics to test whether the Low-Rank Hypothesis (Nature Physics 2024) reveals
additional nonlinear structure near the critical temperature Tc ≈ 2.269.

New metrics:
  - Kernel Effective Rank (RBF, poly2, poly3)
  - Linear-Nonlinear Rank Gap
  - Spectral Gap of the Kernel Gram Matrix

Usage:
    # Interactive mode
    python analyze_kernel_rank.py

    # Command-line mode
    python analyze_kernel_rank.py --project Ising_VaTD_v0.16 \\
        --group <group> --seed 42 --device cuda:0

    # Quick mode
    python analyze_kernel_rank.py --project ... --group ... --seed 42 --quick

    # Replot from existing CSV
    python analyze_kernel_rank.py --replot <csv_path>
"""

import os
os.environ['VATD_NO_MHC'] = '1'

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'nature'])
except ImportError:
    pass

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

from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_distances

from util import select_project, select_group, select_seed, select_device, load_model
from vatd_exact_partition import CRITICAL_TEMPERATURE
from analyze_rank import (
    collect_activations,
    temperature_grid,
    exact_specific_heat,
    effective_rank as svd_effective_rank,
    stable_rank as svd_stable_rank,
    participation_ratio as svd_participation_ratio,
)


# ──────────────────────────────────────────────────────────────
# Kernel PCA Metrics
# ──────────────────────────────────────────────────────────────


def median_heuristic(X):
    """
    Median heuristic for RBF kernel bandwidth.

    gamma = 1 / (2 * sigma^2) where sigma = median(||x_i - x_j||).
    This is the standard default for kernel methods when no prior is available.

    Args:
        X: np.ndarray [N, D]

    Returns:
        gamma: float (sklearn convention: K(x,y) = exp(-gamma * ||x-y||^2))
    """
    dists = pairwise_distances(X, metric='euclidean')
    # Take upper triangle (exclude diagonal zeros)
    triu_idx = np.triu_indices_from(dists, k=1)
    median_dist = np.median(dists[triu_idx])
    if median_dist < 1e-10:
        median_dist = 1.0
    sigma = median_dist
    gamma = 1.0 / (2.0 * sigma ** 2)
    return gamma, sigma


def kernel_effective_rank(eigenvalues):
    """
    Effective rank from kernel PCA eigenvalues via Shannon entropy.

    kernel_erank = exp(H(p)) where p_i = λ_i / Σλ_j for λ_i > 0.
    Analogous to SVD effective rank but in the RKHS feature space.

    Args:
        eigenvalues: np.ndarray of kernel PCA eigenvalues (descending)

    Returns:
        float: kernel effective rank ∈ [1, N]
    """
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) == 0:
        return 1.0
    p = ev / ev.sum()
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def kernel_stable_rank(eigenvalues):
    """
    Stable rank from kernel PCA eigenvalues: Σλ_i² / λ_max².
    """
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) == 0:
        return 1.0
    return float((ev ** 2).sum() / (ev[0] ** 2))


def kernel_participation_ratio(eigenvalues):
    """
    Participation ratio from kernel PCA eigenvalues: (Σλ_i)² / Σλ_i².
    """
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) == 0:
        return 1.0
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def kernel_spectral_gap(eigenvalues):
    """
    Spectral gap: (λ_1 - λ_2) / λ_1.

    A closing gap (→0) at Tc indicates degenerate leading modes,
    analogous to gapless excitations in condensed matter.
    """
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) < 2:
        return 1.0
    return float((ev[0] - ev[1]) / ev[0])


def kernel_spectral_gap_ratio(eigenvalues):
    """
    Spectral gap ratio: λ_2 / λ_1.

    Closer to 1 means nearly degenerate leading eigenvalues.
    """
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) < 2:
        return 0.0
    return float(ev[1] / ev[0])


def compute_kernel_metrics(X, kernel='rbf', gamma=None, degree=2):
    """
    Compute kernel PCA metrics on data matrix X.

    Args:
        X: np.ndarray [N, D], mean-centered
        kernel: str, 'rbf' or 'poly'
        gamma: float, kernel bandwidth (None = auto via median heuristic)
        degree: int, polynomial degree (only for kernel='poly')

    Returns:
        dict with kernel rank metrics and eigenvalues
    """
    N = X.shape[0]

    # Auto bandwidth for RBF
    if kernel == 'rbf' and gamma is None:
        gamma, sigma = median_heuristic(X)
    else:
        sigma = None

    # Fit Kernel PCA — request all components to get full spectrum
    kpca_params = dict(
        kernel=kernel,
        n_components=min(N - 1, X.shape[1], 200),  # cap for speed
        fit_inverse_transform=False,
        eigen_solver='arpack',
    )
    if kernel == 'rbf':
        kpca_params['gamma'] = gamma
    elif kernel == 'poly':
        kpca_params['degree'] = degree
        kpca_params['gamma'] = gamma if gamma is not None else 1.0 / X.shape[1]
        kpca_params['coef0'] = 1.0

    try:
        kpca = KernelPCA(**kpca_params)
        kpca.fit(X)
        eigenvalues = kpca.eigenvalues_
    except Exception:
        # Fallback: compute kernel matrix and eigendecompose manually
        from sklearn.metrics.pairwise import pairwise_kernels
        K = pairwise_kernels(X, metric=kernel, gamma=gamma)
        # Center the kernel matrix
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        eigenvalues = np.linalg.eigvalsh(K_centered)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Ensure descending order and positive
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    return {
        'eigenvalues': eigenvalues,
        'erank': kernel_effective_rank(eigenvalues),
        'stable_rank': kernel_stable_rank(eigenvalues),
        'pr': kernel_participation_ratio(eigenvalues),
        'spectral_gap': kernel_spectral_gap(eigenvalues),
        'spectral_gap_ratio': kernel_spectral_gap_ratio(eigenvalues),
        'gamma': gamma,
        'sigma': sigma,
        'n_components': len(eigenvalues),
    }


# ──────────────────────────────────────────────────────────────
# Main Analysis Loop
# ──────────────────────────────────────────────────────────────


def run_kernel_analysis(
    model, temperatures, device,
    batch_size=200, n_batches=3,
    console=None,
):
    """
    For each temperature: generate samples → collect activations →
    compute both SVD and Kernel PCA rank metrics.

    Returns a DataFrame with SVD and kernel metrics per layer per temperature.
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
        task = progress.add_task("Kernel rank analysis", total=len(temperatures))

        for T_val in temperatures:
            beta_val = 1.0 / T_val
            progress.update(task, description=f"T={T_val:.3f} (β={beta_val:.3f})")

            # Generate samples
            all_samples = []
            T_tensor = torch.full((batch_size,), T_val, device=device)
            model.eval()

            with torch.no_grad():
                for _ in range(n_batches):
                    s = model.sample(batch_size=batch_size, T=T_tensor)
                    all_samples.append(s)

            all_samples = torch.cat(all_samples, dim=0)  # [N, 1, H, W]

            # Collect activations
            activations = collect_activations(model, all_samples, T_val, device)

            for lk in layer_keys:
                if lk not in activations:
                    continue

                act = activations[lk]  # [N, C, H, W]
                N, C, H, W = act.shape

                # ── Channel representation: [N, C] ──
                act_ch = act.mean(dim=(-2, -1))  # spatial average
                act_ch = act_ch - act_ch.mean(dim=0, keepdim=True)  # center

                # SVD metrics (linear)
                _, S_ch, _ = torch.linalg.svd(act_ch, full_matrices=False)
                svd_erank = svd_effective_rank(S_ch)
                svd_srank = svd_stable_rank(S_ch)
                svd_pr = svd_participation_ratio(S_ch)

                # Kernel PCA metrics (nonlinear)
                X_ch = act_ch.numpy().astype(np.float64)

                # RBF kernel
                km_rbf = compute_kernel_metrics(X_ch, kernel='rbf')

                # Polynomial degree 2
                km_poly2 = compute_kernel_metrics(
                    X_ch, kernel='poly', degree=2,
                    gamma=1.0 / X_ch.shape[1],
                )

                # Polynomial degree 3
                km_poly3 = compute_kernel_metrics(
                    X_ch, kernel='poly', degree=3,
                    gamma=1.0 / X_ch.shape[1],
                )

                layer_name = f"layer_{lk}" if isinstance(lk, int) else lk

                records.append({
                    "T": T_val,
                    "beta": beta_val,
                    "T_over_Tc": T_val / CRITICAL_TEMPERATURE,
                    "layer": layer_name,
                    # SVD metrics
                    "svd_erank": svd_erank,
                    "svd_stable_rank": svd_srank,
                    "svd_pr": svd_pr,
                    # Kernel RBF metrics
                    "kernel_rbf_erank": km_rbf['erank'],
                    "kernel_rbf_stable_rank": km_rbf['stable_rank'],
                    "kernel_rbf_pr": km_rbf['pr'],
                    "kernel_rbf_spectral_gap": km_rbf['spectral_gap'],
                    "kernel_rbf_spectral_gap_ratio": km_rbf['spectral_gap_ratio'],
                    "rbf_bandwidth": km_rbf['sigma'],
                    "rbf_gamma": km_rbf['gamma'],
                    # Kernel Poly metrics
                    "kernel_poly2_erank": km_poly2['erank'],
                    "kernel_poly3_erank": km_poly3['erank'],
                    # Rank gaps
                    "rank_gap_rbf": km_rbf['erank'] - svd_erank,
                    "rank_gap_poly2": km_poly2['erank'] - svd_erank,
                    "rank_gap_poly3": km_poly3['erank'] - svd_erank,
                })

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


def plot_kernel_vs_svd_rank(df, L, figs_dir):
    """
    3x2 figure comparing SVD and Kernel PCA rank metrics:
      [0,0] SVD eRank vs T per layer
      [0,1] Kernel RBF eRank vs T per layer
      [1,0] Rank Gap (RBF) vs T per layer
      [1,1] Rank Gap (RBF) avg + Cv overlay
      [2,0] Kernel Poly2 vs Poly3 eRank (layer-averaged)
      [2,1] Kernel eRank heatmap [layer x T]
    """
    temperatures = np.sort(df["T"].unique())
    Cv = exact_specific_heat(L, temperatures)
    Tc = CRITICAL_TEMPERATURE

    layers = sorted(df["layer"].unique(), key=_layer_sort_key)
    n_layers = len(layers)
    colors = LAYER_CMAP(np.linspace(0.15, 0.85, n_layers))

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # ── [0,0] SVD eRank ──
    ax = axes[0, 0]
    for ci, layer in enumerate(layers):
        ld = df[df["layer"] == layer].sort_values("T")
        ax.plot(ld["T"], ld["svd_erank"],
                "o-", color=colors[ci], markersize=3, linewidth=1.2,
                label=_layer_label(layer), alpha=0.85)
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("SVD Effective Rank")
    ax.set_title("Linear SVD eRank")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.15)

    # ── [0,1] Kernel RBF eRank ──
    ax = axes[0, 1]
    for ci, layer in enumerate(layers):
        ld = df[df["layer"] == layer].sort_values("T")
        ax.plot(ld["T"], ld["kernel_rbf_erank"],
                "o-", color=colors[ci], markersize=3, linewidth=1.2,
                label=_layer_label(layer), alpha=0.85)
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Kernel RBF Effective Rank")
    ax.set_title("Nonlinear Kernel eRank (RBF)")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.15)

    # ── [1,0] Rank Gap vs T per layer ──
    ax = axes[1, 0]
    for ci, layer in enumerate(layers):
        ld = df[df["layer"] == layer].sort_values("T")
        ax.plot(ld["T"], ld["rank_gap_rbf"],
                "o-", color=colors[ci], markersize=3, linewidth=1.2,
                label=_layer_label(layer), alpha=0.85)
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.axhline(0, color="gray", ls="-", alpha=0.3, lw=0.5)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Rank Gap (Kernel $-$ SVD)")
    ax.set_title("Linear vs Nonlinear Rank Gap (RBF)")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.15)

    # ── [1,1] Avg Rank Gap + Cv ──
    ax = axes[1, 1]
    avg_gap = df.groupby("T")["rank_gap_rbf"].mean().sort_index()
    T_arr = avg_gap.index.values
    gap_arr = avg_gap.values

    ax.plot(T_arr, gap_arr, "b-o", markersize=4, linewidth=2,
            label="Avg Rank Gap (RBF)")
    ax.axvline(Tc, color="gray", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Rank Gap", color="blue")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.15)

    ax2 = ax.twinx()
    ax2.plot(temperatures, Cv, "r-", linewidth=2, alpha=0.6,
             label="Exact $C_v$ (Onsager)")
    ax2.set_ylabel("$C_v$ / site", color="red")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title("Rank Gap vs Specific Heat")

    # ── [2,0] Poly2 vs Poly3 eRank ──
    ax = axes[2, 0]
    avg_rbf = df.groupby("T")["kernel_rbf_erank"].mean().sort_index()
    avg_poly2 = df.groupby("T")["kernel_poly2_erank"].mean().sort_index()
    avg_poly3 = df.groupby("T")["kernel_poly3_erank"].mean().sort_index()
    avg_svd = df.groupby("T")["svd_erank"].mean().sort_index()

    ax.plot(avg_svd.index, avg_svd.values, "k-o", markersize=3,
            linewidth=1.5, label="SVD (linear)", alpha=0.8)
    ax.plot(avg_rbf.index, avg_rbf.values, "b-s", markersize=3,
            linewidth=1.5, label="RBF", alpha=0.8)
    ax.plot(avg_poly2.index, avg_poly2.values, "g-^", markersize=3,
            linewidth=1.5, label="Poly (d=2)", alpha=0.8)
    ax.plot(avg_poly3.index, avg_poly3.values, "m-v", markersize=3,
            linewidth=1.5, label="Poly (d=3)", alpha=0.8)
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Effective Rank (layer-avg)")
    ax.set_title("Kernel Comparison: SVD vs RBF vs Polynomial")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    # ── [2,1] Kernel eRank heatmap ──
    ax = axes[2, 1]
    numeric_layers = [l for l in layers if l.startswith("layer_")]
    if numeric_layers:
        pivot = df[df["layer"].isin(numeric_layers)].pivot_table(
            values="kernel_rbf_erank", index="layer", columns="T",
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
        plt.colorbar(im, ax=ax, label="Kernel eRank (RBF)")
        ax.set_title("Kernel eRank Heatmap")

    fig.suptitle(
        f"Kernel PCA Low-Rank Analysis: 2D Ising ($L={L}$, $T_c \\approx {Tc:.3f}$)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "kernel_vs_svd_rank.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spectral_gap(df, L, figs_dir):
    """
    2x1 figure for spectral gap analysis:
      [0] Spectral gap (λ₁-λ₂)/λ₁ vs T per layer
      [1] Spectral gap ratio λ₂/λ₁ vs T + Cv overlay
    """
    Tc = CRITICAL_TEMPERATURE
    temperatures = np.sort(df["T"].unique())
    Cv = exact_specific_heat(L, temperatures)

    layers = sorted(df["layer"].unique(), key=_layer_sort_key)
    n_layers = len(layers)
    colors = LAYER_CMAP(np.linspace(0.15, 0.85, n_layers))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── [0] Spectral gap per layer ──
    ax = axes[0]
    for ci, layer in enumerate(layers):
        ld = df[df["layer"] == layer].sort_values("T")
        ax.plot(ld["T"], ld["kernel_rbf_spectral_gap"],
                "o-", color=colors[ci], markersize=3, linewidth=1.2,
                label=_layer_label(layer), alpha=0.85)
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Spectral Gap $(\\lambda_1 - \\lambda_2) / \\lambda_1$")
    ax.set_title("Kernel Spectral Gap")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.15)

    # ── [1] Spectral gap ratio + Cv ──
    ax = axes[1]
    avg_ratio = df.groupby("T")["kernel_rbf_spectral_gap_ratio"].mean().sort_index()
    T_arr = avg_ratio.index.values
    ratio_arr = avg_ratio.values

    ax.plot(T_arr, ratio_arr, "b-o", markersize=4, linewidth=2,
            label="Avg $\\lambda_2 / \\lambda_1$")
    ax.axvline(Tc, color="gray", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("$\\lambda_2 / \\lambda_1$", color="blue")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.15)

    ax2 = ax.twinx()
    ax2.plot(temperatures, Cv, "r-", linewidth=2, alpha=0.6,
             label="Exact $C_v$ (Onsager)")
    ax2.set_ylabel("$C_v$ / site", color="red")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title("Spectral Gap Ratio vs Specific Heat")

    fig.suptitle(
        f"Kernel Spectral Gap: 2D Ising ($L={L}$, $T_c \\approx {Tc:.3f}$)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "kernel_spectral_gap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_kernel_spectra(
    model, selected_temperatures, device, batch_size, n_batches, L, figs_dir, console,
):
    """
    Kernel eigenvalue scree plots at 3 representative temperatures.
    """
    Tc = CRITICAL_TEMPERATURE
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, T_val in enumerate(selected_temperatures):
        ax = axes[idx]
        beta_val = 1.0 / T_val
        console.print(f"  Kernel spectra at T={T_val:.3f} (β={beta_val:.3f})")

        T_tensor = torch.full((batch_size,), T_val, device=device)
        all_samples = []
        with torch.no_grad():
            for _ in range(n_batches):
                all_samples.append(model.sample(batch_size=batch_size, T=T_tensor))
        all_samples = torch.cat(all_samples, dim=0)

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
            X = act_flat.numpy().astype(np.float64)

            km = compute_kernel_metrics(X, kernel='rbf')
            ev = km['eigenvalues']
            if len(ev) > 0:
                ev_norm = ev / ev[0]
            else:
                continue

            label = f"Block {lk}" if isinstance(lk, int) else "Final"
            ax.semilogy(
                range(1, len(ev_norm) + 1), ev_norm,
                color=colors[ci], alpha=0.75, linewidth=1.5, label=label,
            )

        phase = (
            "Disordered" if T_val > Tc * 1.1
            else "Ordered" if T_val < Tc * 0.9
            else "Critical"
        )
        ax.set_title(f"$T={T_val:.2f}$ ({phase})\n$\\beta={beta_val:.3f}$")
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("$\\lambda_i / \\lambda_1$")
        ax.legend(fontsize=6, ncol=2)
        ax.set_ylim(1e-4, 1.5)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Kernel PCA Eigenvalue Spectra ($L={L}$)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "kernel_eigenvalue_spectra.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bandwidth_sensitivity(
    model, temperatures_subset, device, batch_size, n_batches, L, figs_dir, console,
):
    """
    Bandwidth sensitivity analysis: sweep sigma and show kernel eRank is robust.

    Uses a single layer (layer_0) at 5 temperatures spanning the phase transition.
    """
    Tc = CRITICAL_TEMPERATURE
    sigma_multipliers = [0.1, 0.3, 1.0, 3.0, 10.0]

    records = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Bandwidth sweep", total=len(temperatures_subset))

        for T_val in temperatures_subset:
            progress.update(task, description=f"σ sweep at T={T_val:.3f}")

            T_tensor = torch.full((batch_size,), T_val, device=device)
            all_samples = []
            with torch.no_grad():
                for _ in range(n_batches):
                    all_samples.append(model.sample(batch_size=batch_size, T=T_tensor))
            all_samples = torch.cat(all_samples, dim=0)

            activations = collect_activations(model, all_samples, T_val, device)

            # Use first layer
            lk = 0 if 0 in activations else list(activations.keys())[0]
            act = activations[lk]
            N, C, H, W = act.shape
            act_ch = act.mean(dim=(-2, -1))
            act_ch = act_ch - act_ch.mean(dim=0, keepdim=True)
            X = act_ch.numpy().astype(np.float64)

            # Get base sigma via median heuristic
            _, sigma_base = median_heuristic(X)

            for mult in sigma_multipliers:
                sigma = sigma_base * mult
                gamma = 1.0 / (2.0 * sigma ** 2)
                km = compute_kernel_metrics(X, kernel='rbf', gamma=gamma)
                records.append({
                    "T": T_val,
                    "sigma_mult": mult,
                    "sigma": sigma,
                    "kernel_erank": km['erank'],
                    "spectral_gap": km['spectral_gap'],
                })

            progress.advance(task)

    df_bw = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── eRank vs T for each bandwidth ──
    ax = axes[0]
    for mult in sigma_multipliers:
        rd = df_bw[df_bw["sigma_mult"] == mult].sort_values("T")
        ax.plot(rd["T"], rd["kernel_erank"],
                "o-", markersize=4, linewidth=1.5,
                label=f"$\\sigma = {mult}\\times\\sigma_{{med}}$", alpha=0.85)
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Kernel eRank (RBF)")
    ax.set_title("Bandwidth Sensitivity: eRank")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    # ── Spectral gap vs T for each bandwidth ──
    ax = axes[1]
    for mult in sigma_multipliers:
        rd = df_bw[df_bw["sigma_mult"] == mult].sort_values("T")
        ax.plot(rd["T"], rd["spectral_gap"],
                "o-", markersize=4, linewidth=1.5,
                label=f"$\\sigma = {mult}\\times\\sigma_{{med}}$", alpha=0.85)
    ax.axvline(Tc, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Spectral Gap $(\\lambda_1 - \\lambda_2) / \\lambda_1$")
    ax.set_title("Bandwidth Sensitivity: Spectral Gap")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"RBF Bandwidth Sensitivity ($L={L}$, Layer 0)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "bandwidth_sensitivity.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Kernel PCA Low-Rank Analysis of PixelCNN Representations"
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
        "--replot", type=str, default=None,
        help="Regenerate plots from existing CSV file",
    )
    args = parser.parse_args()

    console = Console()
    Tc = CRITICAL_TEMPERATURE

    console.print("[bold green]Kernel PCA Low-Rank Analysis[/bold green]")
    console.print(f"Critical Temperature: Tc = {Tc:.4f}")

    # ── Replot mode ──
    if args.replot:
        csv_path = Path(args.replot)
        if not csv_path.exists():
            console.print(f"[red]CSV not found:[/red] {csv_path}")
            return

        console.print(f"[yellow]Replot mode:[/yellow] reading {csv_path}")
        df = pd.read_csv(csv_path)

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

        fig_path = plot_kernel_vs_svd_rank(df, L, figs_dir)
        console.print(f"[green]Main plot:[/green] {fig_path}")

        fig_path = plot_spectral_gap(df, L, figs_dir)
        console.print(f"[green]Spectral gap plot:[/green] {fig_path}")

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
    if args.quick:
        temps = temperature_grid(T_min=0.8, T_max=6.0, n_coarse=12, n_critical=8)
        batch_size, n_batches = 100, 2
        console.print("[yellow]Quick mode: 20 temps, batch=100, 2 batches[/yellow]")
    else:
        temps = temperature_grid(T_min=0.5, T_max=10.0, n_coarse=25, n_critical=15)
        batch_size, n_batches = args.batch_size, args.n_batches

    console.print(
        f"Temperature grid: {len(temps)} points, "
        f"T in [{temps.min():.3f}, {temps.max():.3f}]"
    )
    console.print(f"Samples per temperature: {batch_size * n_batches}")

    # ── Phase 1: Kernel rank analysis ──
    console.print("\n[bold cyan]Phase 1: SVD + Kernel PCA rank analysis[/bold cyan]")
    df = run_kernel_analysis(model, temps, device, batch_size, n_batches, console)

    csv_path = output_dir / f"kernel_rank_analysis_{seed}.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]Data saved:[/green] {csv_path}")

    # ── Phase 2: Plots ──
    console.print("\n[bold cyan]Phase 2: Generating plots[/bold cyan]")

    fig_path = plot_kernel_vs_svd_rank(df, L, figs_dir)
    console.print(f"[green]Main comparison plot:[/green] {fig_path}")

    fig_path = plot_spectral_gap(df, L, figs_dir)
    console.print(f"[green]Spectral gap plot:[/green] {fig_path}")

    # Scree plots at 3 representative temperatures
    selected_T = [
        temps[temps > 2.0 * Tc].min(),       # high T (disordered)
        temps[np.argmin(np.abs(temps - Tc))], # near Tc
        temps[temps < 0.6 * Tc].max(),        # low T (ordered)
    ]
    console.print(
        f"Scree plot temperatures: "
        f"{', '.join(f'T={t:.3f}' for t in selected_T)}"
    )
    fig_path = plot_kernel_spectra(
        model, selected_T, device, batch_size, n_batches, L, figs_dir, console,
    )
    console.print(f"[green]Kernel spectra plot:[/green] {fig_path}")

    # Bandwidth sensitivity (5 temperatures)
    console.print("\n[bold cyan]Phase 3: Bandwidth sensitivity[/bold cyan]")
    bw_temps = np.array([
        temps[temps < 0.5 * Tc].max(),
        temps[np.argmin(np.abs(temps - 0.8 * Tc))],
        temps[np.argmin(np.abs(temps - Tc))],
        temps[np.argmin(np.abs(temps - 1.2 * Tc))],
        temps[temps > 2.0 * Tc].min(),
    ])
    fig_path = plot_bandwidth_sensitivity(
        model, bw_temps, device, batch_size, n_batches, L, figs_dir, console,
    )
    console.print(f"[green]Bandwidth sensitivity plot:[/green] {fig_path}")

    # ── Summary ──
    console.print("\n[bold cyan]Summary: SVD vs Kernel eRank at T nearest Tc[/bold cyan]")
    T_nearest_Tc = df.iloc[(df["T"] - Tc).abs().argsort().iloc[0]]["T"]
    near_df = df[df["T"] == T_nearest_Tc]
    console.print(f"T = {T_nearest_Tc:.4f} (T/Tc = {T_nearest_Tc / Tc:.4f})")
    console.print(
        f"{'Layer':>10s}  {'SVD eR':>8s}  {'RBF eR':>8s}  "
        f"{'Gap':>8s}  {'SG Ratio':>8s}  {'P2 eR':>8s}  {'P3 eR':>8s}"
    )
    console.print("-" * 68)
    for layer in sorted(near_df["layer"].unique(), key=_layer_sort_key):
        row = near_df[near_df["layer"] == layer].iloc[0]
        console.print(
            f"{_layer_label(layer):>10s}  {row['svd_erank']:>8.2f}  "
            f"{row['kernel_rbf_erank']:>8.2f}  {row['rank_gap_rbf']:>8.2f}  "
            f"{row['kernel_rbf_spectral_gap_ratio']:>8.4f}  "
            f"{row['kernel_poly2_erank']:>8.2f}  {row['kernel_poly3_erank']:>8.2f}"
        )

    # Peak rank gap analysis
    console.print("\n[bold cyan]Peak Rank Gap by Layer[/bold cyan]")
    console.print(
        f"{'Layer':>10s}  {'Peak Gap':>10s}  {'at T':>7s}  "
        f"{'T/Tc':>6s}  {'Min Gap':>9s}"
    )
    console.print("-" * 50)
    for layer in sorted(df["layer"].unique(), key=_layer_sort_key):
        ld = df[df["layer"] == layer]
        peak_idx = ld["rank_gap_rbf"].abs().idxmax()
        peak_T = ld.loc[peak_idx, "T"]
        peak_gap = ld.loc[peak_idx, "rank_gap_rbf"]
        min_gap = ld["rank_gap_rbf"].min()
        console.print(
            f"{_layer_label(layer):>10s}  {peak_gap:>10.2f}  {peak_T:>7.3f}  "
            f"{peak_T / Tc:>6.3f}  {min_gap:>9.2f}"
        )

    console.print(f"\n[bold green]Analysis complete.[/bold green]")
    console.print(f"Figures: {figs_dir}/")
    console.print(f"Data:    {csv_path}")


if __name__ == "__main__":
    main()
