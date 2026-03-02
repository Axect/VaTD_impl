"""
Compression Degradation Comparison: SVD vs Kernel PCA.

Instead of comparing eRank (a descriptive statistic in different spaces),
this script answers the correct question:

    "Given the same number of retained components k, which method
     better preserves the original activation information?"

Both methods are evaluated in the SAME space (input space) via
reconstruction error:

    D_method(T, k) = ||X - X_hat||_F^2 / ||X||_F^2

where X_hat is reconstructed from top-k components.

For SVD:   X_hat = U[:,:k] @ diag(S[:k]) @ V[:,:k].T  (exact)
For KPCA:  X_hat = kpca.inverse_transform(kpca.transform(X))  (approximate pre-image)

Usage:
    python analyze_compression_comparison.py --project Ising_VaTD_v0.18 \\
        --group <group> --seed 42 --device cuda:0

    python analyze_compression_comparison.py ... --quick
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

from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics.pairwise import pairwise_distances

from util import select_project, select_group, select_seed, select_device, load_model
from vatd_exact_partition import CRITICAL_TEMPERATURE
from analyze_rank import (
    collect_activations,
    temperature_grid,
    exact_specific_heat,
)


# ──────────────────────────────────────────────────────────────
# Bandwidth Heuristic
# ──────────────────────────────────────────────────────────────


def median_heuristic(X):
    """Median heuristic for RBF bandwidth: gamma = 1/(2*sigma^2)."""
    dists = pairwise_distances(X, metric='euclidean')
    triu_idx = np.triu_indices_from(dists, k=1)
    median_dist = np.median(dists[triu_idx])
    if median_dist < 1e-10:
        median_dist = 1.0
    sigma = median_dist
    gamma = 1.0 / (2.0 * sigma ** 2)
    return gamma, sigma


# ──────────────────────────────────────────────────────────────
# Reconstruction Degradation
# ──────────────────────────────────────────────────────────────


def svd_reconstruction_degradation(X, k):
    """
    SVD reconstruction degradation at rank k.

    D(k) = ||X - X_k||_F^2 / ||X||_F^2

    where X_k = U[:,:k] @ S[:k] @ V[:,:k].T

    This is exactly 1 - (sum of top-k squared SVs / total).
    But we compute it explicitly for consistency.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    X_hat = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    residual = np.linalg.norm(X - X_hat, 'fro') ** 2
    total = np.linalg.norm(X, 'fro') ** 2
    if total < 1e-12:
        return 0.0
    return float(residual / total)


def kpca_reconstruction_degradation(X, k, kernel='rbf', gamma=None, degree=2):
    """
    Kernel PCA reconstruction degradation at rank k.

    D(k) = ||X - X_hat||_F^2 / ||X||_F^2

    where X_hat = kpca.inverse_transform(kpca.transform(X)).
    The pre-image is approximated by sklearn via kernel ridge regression.

    Returns (degradation, gamma_used).
    """
    N, D = X.shape

    if kernel == 'rbf' and gamma is None:
        gamma, _ = median_heuristic(X)

    kpca_params = dict(
        kernel=kernel,
        n_components=k,
        fit_inverse_transform=True,  # enables inverse_transform
        eigen_solver='arpack',
    )
    if kernel == 'rbf':
        kpca_params['gamma'] = gamma
    elif kernel == 'poly':
        kpca_params['degree'] = degree
        kpca_params['gamma'] = gamma if gamma is not None else 1.0 / D
        kpca_params['coef0'] = 1.0

    kpca = KernelPCA(**kpca_params)
    Z = kpca.fit_transform(X)         # [N, k] in RKHS
    X_hat = kpca.inverse_transform(Z)  # [N, D] pre-image approximation

    residual = np.linalg.norm(X - X_hat, 'fro') ** 2
    total = np.linalg.norm(X, 'fro') ** 2
    if total < 1e-12:
        return 0.0, gamma
    return float(residual / total), gamma


def explained_variance_curves(X, max_k, kernel='rbf', gamma=None, degree=2):
    """
    Compute cumulative explained variance for SVD and Kernel PCA.

    Returns dict with:
        svd_cumvar: [max_k] cumulative explained variance ratio for SVD
        kpca_cumvar: [max_k] cumulative explained variance ratio for KPCA
        svd_singular_values: all SVs
        kpca_eigenvalues: all kernel eigenvalues
    """
    N, D = X.shape

    # SVD
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    sv2 = S ** 2
    svd_cumvar = np.cumsum(sv2) / sv2.sum()

    # Kernel PCA — get full spectrum
    if kernel == 'rbf' and gamma is None:
        gamma, _ = median_heuristic(X)

    kpca_params = dict(
        kernel=kernel,
        n_components=min(N - 1, D, max_k * 2),
        fit_inverse_transform=False,
        eigen_solver='arpack',
    )
    if kernel == 'rbf':
        kpca_params['gamma'] = gamma
    elif kernel == 'poly':
        kpca_params['degree'] = degree
        kpca_params['gamma'] = gamma if gamma is not None else 1.0 / D
        kpca_params['coef0'] = 1.0

    kpca = KernelPCA(**kpca_params)
    kpca.fit(X)
    ev = kpca.eigenvalues_
    ev = ev[ev > 1e-10]
    kpca_cumvar = np.cumsum(ev) / ev.sum()

    return {
        'svd_cumvar': svd_cumvar[:max_k],
        'kpca_cumvar': kpca_cumvar[:max_k],
        'svd_singular_values': S,
        'kpca_eigenvalues': ev,
        'gamma': gamma,
    }


def numerical_rank_at_threshold(cumvar, threshold=0.95):
    """Minimum k such that cumulative variance >= threshold."""
    mask = cumvar >= threshold
    if mask.any():
        return int(np.argmax(mask)) + 1
    return len(cumvar)


# ──────────────────────────────────────────────────────────────
# Main Analysis
# ──────────────────────────────────────────────────────────────


def run_compression_comparison(
    model, temperatures, device,
    batch_size=200, n_batches=3,
    rank_fractions=None,
    console=None,
):
    """
    For each temperature: collect activations → compare SVD vs KPCA
    reconstruction degradation at several rank values.

    Returns two DataFrames:
        df_deg: per (T, layer, method, k) degradation values
        df_numrank: per (T, layer) numerical rank at 95% for SVD vs KPCA
    """
    if console is None:
        console = Console()

    if rank_fractions is None:
        rank_fractions = [1, 2, 3, 5, 10, 20, 30, 50]

    num_layers = len(model.masked_conv.hidden_convs)
    layer_keys = list(range(num_layers)) + ["final"]

    deg_records = []
    numrank_records = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Compression comparison", total=len(temperatures))

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
            all_samples = torch.cat(all_samples, dim=0)

            # Collect activations
            activations = collect_activations(model, all_samples, T_val, device)

            for lk in layer_keys:
                if lk not in activations:
                    continue

                act = activations[lk]  # [N, C, H, W]
                N, C, H, W = act.shape

                # Channel-averaged: [N, C]
                act_ch = act.mean(dim=(-2, -1))
                act_ch = act_ch - act_ch.mean(dim=0, keepdim=True)
                X = act_ch.numpy().astype(np.float64)

                layer_name = f"layer_{lk}" if isinstance(lk, int) else lk
                max_k = min(N - 1, C)

                # ── Explained variance curves ──
                evc = explained_variance_curves(X, max_k=max_k)

                # Numerical rank at 95%
                svd_nr95 = numerical_rank_at_threshold(evc['svd_cumvar'], 0.95)
                kpca_nr95 = numerical_rank_at_threshold(evc['kpca_cumvar'], 0.95)
                svd_nr99 = numerical_rank_at_threshold(evc['svd_cumvar'], 0.99)
                kpca_nr99 = numerical_rank_at_threshold(evc['kpca_cumvar'], 0.99)

                numrank_records.append({
                    'T': T_val,
                    'beta': beta_val,
                    'T_over_Tc': T_val / CRITICAL_TEMPERATURE,
                    'layer': layer_name,
                    'svd_numrank_95': svd_nr95,
                    'kpca_numrank_95': kpca_nr95,
                    'numrank_diff_95': svd_nr95 - kpca_nr95,
                    'svd_numrank_99': svd_nr99,
                    'kpca_numrank_99': kpca_nr99,
                    'numrank_diff_99': svd_nr99 - kpca_nr99,
                })

                # ── Reconstruction degradation at specific k values ──
                gamma_rbf = evc['gamma']

                for k in rank_fractions:
                    if k > max_k:
                        continue

                    # SVD degradation
                    d_svd = svd_reconstruction_degradation(X, k)

                    # Kernel PCA degradation (RBF)
                    try:
                        d_kpca_rbf, _ = kpca_reconstruction_degradation(
                            X, k, kernel='rbf', gamma=gamma_rbf,
                        )
                    except Exception:
                        d_kpca_rbf = np.nan

                    # Kernel PCA degradation (Poly d=2)
                    try:
                        d_kpca_poly2, _ = kpca_reconstruction_degradation(
                            X, k, kernel='poly', degree=2,
                            gamma=1.0 / X.shape[1],
                        )
                    except Exception:
                        d_kpca_poly2 = np.nan

                    deg_records.append({
                        'T': T_val,
                        'beta': beta_val,
                        'T_over_Tc': T_val / CRITICAL_TEMPERATURE,
                        'layer': layer_name,
                        'k': k,
                        'svd_degradation': d_svd,
                        'kpca_rbf_degradation': d_kpca_rbf,
                        'kpca_poly2_degradation': d_kpca_poly2,
                        'advantage_rbf': d_svd - d_kpca_rbf,
                        'advantage_poly2': d_svd - d_kpca_poly2,
                    })

            progress.advance(task)

    df_deg = pd.DataFrame(deg_records)
    df_numrank = pd.DataFrame(numrank_records)
    return df_deg, df_numrank


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


def plot_degradation_comparison(df_deg, df_numrank, L, figs_dir):
    """
    Main figure: 2x2
      [0,0] Degradation at k=5: SVD vs KPCA(RBF) vs T
      [0,1] Advantage (D_SVD - D_KPCA) vs T at several k values
      [1,0] Numerical rank at 95% threshold: SVD vs KPCA vs T
      [1,1] Numerical rank difference + Cv overlay
    """
    Tc = CRITICAL_TEMPERATURE
    temperatures = np.sort(df_numrank["T"].unique())
    Cv = exact_specific_heat(L, temperatures)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ── [0,0] Degradation at k=5 ──
    ax = axes[0, 0]
    k_show = 5
    df_k = df_deg[df_deg["k"] == k_show]
    if not df_k.empty:
        avg_svd = df_k.groupby("T")["svd_degradation"].mean().sort_index()
        avg_kpca = df_k.groupby("T")["kpca_rbf_degradation"].mean().sort_index()

        ax.plot(avg_svd.index, avg_svd.values, 'k-o', markersize=4,
                linewidth=1.5, label=f'SVD ($k={k_show}$)')
        ax.plot(avg_kpca.index, avg_kpca.values, 'b-s', markersize=4,
                linewidth=1.5, label=f'KPCA RBF ($k={k_show}$)')
        ax.axvline(Tc, color='red', ls='--', alpha=0.4, lw=0.8)
        ax.set_xlabel('Temperature $T$')
        ax.set_ylabel('Reconstruction Degradation')
        ax.set_title(f'Degradation at $k={k_show}$ components')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

    # ── [0,1] Advantage curves at several k ──
    ax = axes[0, 1]
    k_values = sorted(df_deg["k"].unique())
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(k_values)))
    for ci, k in enumerate(k_values):
        df_k = df_deg[df_deg["k"] == k]
        if df_k.empty:
            continue
        avg_adv = df_k.groupby("T")["advantage_rbf"].mean().sort_index()
        ax.plot(avg_adv.index, avg_adv.values, 'o-', color=colors[ci],
                markersize=3, linewidth=1.2, label=f'$k={k}$', alpha=0.85)
    ax.axvline(Tc, color='red', ls='--', alpha=0.4, lw=0.8)
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel('$D_{\\mathrm{SVD}} - D_{\\mathrm{KPCA}}$ (advantage)')
    ax.set_title('KPCA Advantage (>0 = KPCA better)')
    ax.set_xscale('log')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.15)

    # ── [1,0] Numerical rank at 95% ──
    ax = axes[1, 0]
    layers = sorted(df_numrank["layer"].unique(), key=_layer_sort_key)
    n_layers = len(layers)
    colors_layer = LAYER_CMAP(np.linspace(0.15, 0.85, n_layers))

    for ci, layer in enumerate(layers):
        ld = df_numrank[df_numrank["layer"] == layer].sort_values("T")
        ax.plot(ld["T"], ld["svd_numrank_95"], 'o-', color=colors_layer[ci],
                markersize=3, linewidth=1.2, alpha=0.85)
        ax.plot(ld["T"], ld["kpca_numrank_95"], 's--', color=colors_layer[ci],
                markersize=3, linewidth=1.0, alpha=0.6)

    # Manual legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', marker='o', ls='-', label='SVD (95%)'),
        Line2D([0], [0], color='gray', marker='s', ls='--', label='KPCA (95%)'),
    ]
    ax.axvline(Tc, color='red', ls='--', alpha=0.4, lw=0.8)
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel('Components for 95% variance')
    ax.set_title('Numerical Rank at 95% (solid=SVD, dashed=KPCA)')
    ax.set_xscale('log')
    ax.legend(handles=legend_elements, fontsize=8)
    ax.grid(True, alpha=0.15)

    # ── [1,1] Numrank difference + Cv ──
    ax = axes[1, 1]
    avg_diff = df_numrank.groupby("T")["numrank_diff_95"].mean().sort_index()
    T_arr = avg_diff.index.values
    diff_arr = avg_diff.values

    ax.plot(T_arr, diff_arr, 'b-o', markersize=4, linewidth=2,
            label='$k_{\\mathrm{SVD}}^{95\\%} - k_{\\mathrm{KPCA}}^{95\\%}$')
    ax.axvline(Tc, color='gray', ls='--', alpha=0.4, lw=0.8)
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel('Numrank difference', color='blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.15)

    ax2 = ax.twinx()
    ax2.plot(temperatures, Cv, 'r-', linewidth=2, alpha=0.6,
             label='Exact $C_v$ (Onsager)')
    ax2.set_ylabel('$C_v$ / site', color='red')

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    ax.set_title('Numerical Rank Difference vs $C_v$')

    fig.suptitle(
        f'Compression Comparison: SVD vs Kernel PCA ($L={L}$, $T_c \\approx {Tc:.3f}$)',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()

    path = Path(figs_dir) / 'compression_comparison.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    return path


def plot_degradation_per_layer(df_deg, L, figs_dir):
    """
    Per-layer degradation heatmaps: SVD vs KPCA at k=5.
    Shows which layers benefit most from nonlinear compression.
    """
    Tc = CRITICAL_TEMPERATURE
    k_show = 5
    df_k = df_deg[df_deg["k"] == k_show]
    if df_k.empty:
        return None

    layers = sorted(df_k["layer"].unique(), key=_layer_sort_key)
    n_layers = len(layers)
    colors = LAYER_CMAP(np.linspace(0.15, 0.85, n_layers))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # [0] SVD degradation per layer
    ax = axes[0]
    for ci, layer in enumerate(layers):
        ld = df_k[df_k["layer"] == layer].sort_values("T")
        ax.plot(ld["T"], ld["svd_degradation"], 'o-', color=colors[ci],
                markersize=3, linewidth=1.2, label=_layer_label(layer), alpha=0.85)
    ax.axvline(Tc, color='red', ls='--', alpha=0.4, lw=0.8)
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel('SVD Degradation')
    ax.set_title(f'SVD Reconstruction Error ($k={k_show}$)')
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    # [1] KPCA degradation per layer
    ax = axes[1]
    for ci, layer in enumerate(layers):
        ld = df_k[df_k["layer"] == layer].sort_values("T")
        ax.plot(ld["T"], ld["kpca_rbf_degradation"], 'o-', color=colors[ci],
                markersize=3, linewidth=1.2, label=_layer_label(layer), alpha=0.85)
    ax.axvline(Tc, color='red', ls='--', alpha=0.4, lw=0.8)
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel('KPCA RBF Degradation')
    ax.set_title(f'KPCA RBF Reconstruction Error ($k={k_show}$)')
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    # [2] Advantage per layer
    ax = axes[2]
    for ci, layer in enumerate(layers):
        ld = df_k[df_k["layer"] == layer].sort_values("T")
        ax.plot(ld["T"], ld["advantage_rbf"], 'o-', color=colors[ci],
                markersize=3, linewidth=1.2, label=_layer_label(layer), alpha=0.85)
    ax.axvline(Tc, color='red', ls='--', alpha=0.4, lw=0.8)
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel('Advantage ($D_{SVD} - D_{KPCA}$)')
    ax.set_title(f'KPCA Advantage per Layer ($k={k_show}$)')
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        f'Per-Layer Compression: SVD vs KPCA ($L={L}$, $k={k_show}$)',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    path = Path(figs_dir) / 'compression_comparison_per_layer.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compression Comparison: SVD vs Kernel PCA"
    )
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--n_batches", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--replot", type=str, default=None)
    args = parser.parse_args()

    console = Console()
    Tc = CRITICAL_TEMPERATURE

    console.print("[bold green]Compression Comparison: SVD vs Kernel PCA[/bold green]")
    console.print(f"Critical Temperature: Tc = {Tc:.4f}")

    # Replot mode
    if args.replot:
        csv_path = Path(args.replot)
        if not csv_path.exists():
            console.print(f"[red]CSV not found:[/red] {csv_path}")
            return

        df_deg = pd.read_csv(csv_path)
        nr_csv = csv_path.parent / csv_path.name.replace("degradation", "numrank")
        df_numrank = pd.read_csv(nr_csv) if nr_csv.exists() else None

        L = 16
        figs_dir = Path(f"figs/{csv_path.parent.name}")
        figs_dir.mkdir(parents=True, exist_ok=True)

        if df_numrank is not None:
            fig_path = plot_degradation_comparison(df_deg, df_numrank, L, figs_dir)
            console.print(f"[green]Main plot:[/green] {fig_path}")

        fig_path = plot_degradation_per_layer(df_deg, L, figs_dir)
        if fig_path:
            console.print(f"[green]Per-layer plot:[/green] {fig_path}")

        console.print("[bold green]Replot complete.[/bold green]")
        return

    # Model selection
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

    # Output directories
    figs_dir = Path(f"figs/{group_name}")
    figs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(f"runs/{project}/{group_name}")

    # Temperature grid
    if args.quick:
        temps = temperature_grid(T_min=0.8, T_max=6.0, n_coarse=12, n_critical=8)
        batch_size, n_batches = 100, 2
        rank_ks = [1, 2, 3, 5, 10, 20]
        console.print("[yellow]Quick mode[/yellow]")
    else:
        temps = temperature_grid(T_min=0.5, T_max=10.0, n_coarse=25, n_critical=15)
        batch_size, n_batches = args.batch_size, args.n_batches
        rank_ks = [1, 2, 3, 5, 10, 20, 30, 50]

    console.print(
        f"Temperature grid: {len(temps)} points, "
        f"T in [{temps.min():.3f}, {temps.max():.3f}]"
    )
    console.print(f"Samples per temperature: {batch_size * n_batches}")
    console.print(f"Rank values to test: {rank_ks}")

    # Run analysis
    console.print("\n[bold cyan]Running compression comparison[/bold cyan]")
    df_deg, df_numrank = run_compression_comparison(
        model, temps, device,
        batch_size=batch_size, n_batches=n_batches,
        rank_fractions=rank_ks,
        console=console,
    )

    # Save
    csv_path = output_dir / f"compression_degradation_{seed}.csv"
    df_deg.to_csv(csv_path, index=False)
    console.print(f"[green]Degradation data:[/green] {csv_path}")

    nr_csv = output_dir / f"compression_numrank_{seed}.csv"
    df_numrank.to_csv(nr_csv, index=False)
    console.print(f"[green]Numrank data:[/green] {nr_csv}")

    # Plots
    console.print("\n[bold cyan]Generating plots[/bold cyan]")

    fig_path = plot_degradation_comparison(df_deg, df_numrank, L, figs_dir)
    console.print(f"[green]Main comparison plot:[/green] {fig_path}")

    fig_path = plot_degradation_per_layer(df_deg, L, figs_dir)
    if fig_path:
        console.print(f"[green]Per-layer plot:[/green] {fig_path}")

    # Summary
    console.print("\n[bold cyan]Summary: Numerical Rank at 95% (layer-averaged)[/bold cyan]")
    console.print(
        f"{'T':>7s}  {'T/Tc':>6s}  {'SVD k95':>8s}  {'KPCA k95':>9s}  "
        f"{'Diff':>6s}  {'Winner':>8s}"
    )
    console.print("-" * 55)
    for T_val in sorted(df_numrank["T"].unique()):
        td = df_numrank[df_numrank["T"] == T_val]
        svd_k = td["svd_numrank_95"].mean()
        kpca_k = td["kpca_numrank_95"].mean()
        diff = svd_k - kpca_k
        winner = "KPCA" if diff > 0 else "SVD" if diff < 0 else "tie"
        console.print(
            f"{T_val:>7.3f}  {T_val / Tc:>6.3f}  {svd_k:>8.1f}  {kpca_k:>9.1f}  "
            f"{diff:>6.1f}  {winner:>8s}"
        )

    # Advantage summary at k=5
    console.print("\n[bold cyan]Advantage at k=5 (D_SVD - D_KPCA, layer-averaged)[/bold cyan]")
    df_k5 = df_deg[df_deg["k"] == 5]
    if not df_k5.empty:
        console.print(f"{'T':>7s}  {'Advantage':>10s}  {'Meaning':>20s}")
        console.print("-" * 42)
        for T_val in sorted(df_k5["T"].unique()):
            td = df_k5[df_k5["T"] == T_val]
            adv = td["advantage_rbf"].mean()
            meaning = "KPCA better" if adv > 0.001 else "SVD better" if adv < -0.001 else "~equal"
            console.print(f"{T_val:>7.3f}  {adv:>10.4f}  {meaning:>20s}")

    console.print(f"\n[bold green]Analysis complete.[/bold green]")
    console.print(f"Figures: {figs_dir}/")
    console.print(f"Data:    {csv_path}")


if __name__ == "__main__":
    main()
