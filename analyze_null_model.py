"""
Phase 0: Random-Weight Null Model eRank Analysis

Tests whether the eRank dip at Tc is an artifact of local connectivity
(convolution's spatial filtering of critical correlations) or a learned
representation of RG-relevant operators.

Three-tier falsification ladder:
  Tier 1: Random convnet shows dip → geometry-only artifact
  Tier 2: Trained shows dip, random does not → learned locality effect
  Tier 3: Low-rank directions align with CFT operators → genuine RG

Usage:
    python analyze_null_model.py --device cuda:0
    python analyze_null_model.py --device cuda:0 --quick
    python analyze_null_model.py --replot <csv_path>
"""

import os
os.environ['VATD_NO_MHC'] = '1'

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn,
)
import argparse

from config import RunConfig
from util import load_model
from ising import swendsen_wang_update
from analyze_rank import (
    collect_activations, temperature_grid,
    exact_specific_heat, get_critical_temperature,
    effective_rank, von_neumann_effective_rank, stable_rank,
    participation_ratio, nuclear_rank, renyi2_rank,
    numerical_rank, elbow_rank, optimal_hard_threshold,
    shannon_entropy, von_neumann_entropy,
    spectral_gap_ratio, marchenko_pastur_outlier_count,
)
from unified_rank_metrics import renyi_effective_rank

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

PIXELCNN_PROJECT = "Ising_VaTD_v0.18"
PIXELCNN_GROUP = "DiscretePixelCNN_lr1e-3_e250_3028fe"
PIXELCNN_SEED = "42"

LATTICEGPT_PROJECT = "Ising_VaTD_v0.21"
LATTICEGPT_GROUP = "LatticeGPT_lr2e-1_e300_d54beb"
LATTICEGPT_SEED = "42"

RENYI_ALPHAS = [0.5, 1.0, 2.0, 5.0]  # inf handled separately

from vatd_exact_partition import CRITICAL_TEMPERATURE as Tc


# ──────────────────────────────────────────────────────────────
# MCMC Sample Generation
# ──────────────────────────────────────────────────────────────

def generate_mcmc_samples(T_val, batch_size, device, L=16, n_sweeps=10):
    """Generate equilibrium Ising samples via Swendsen-Wang MCMC."""
    # Initialize random spins in {-1, +1}
    samples = 2 * torch.randint(0, 2, (batch_size, 1, L, L), device=device).float() - 1
    # Fix first spin for symmetry breaking
    samples[:, :, 0, 0] = 1.0
    T_tensor = torch.full((batch_size,), T_val, device=device)
    samples = swendsen_wang_update(samples, T=T_tensor, n_sweeps=n_sweeps, fix_first=True)
    return samples


# ──────────────────────────────────────────────────────────────
# Compute Rank Metrics for a Single Activation
# ──────────────────────────────────────────────────────────────

def compute_rank_metrics(act, prefix="channel"):
    """
    Compute all rank metrics for a given activation tensor.

    Args:
        act: [N, C, H, W] activation tensor
        prefix: "channel" or "spatial"

    Returns:
        dict with metric values
    """
    N, C, H, W = act.shape
    metrics = {}

    if prefix == "channel":
        # Average over spatial dims → [N, C]
        x = act.mean(dim=(-2, -1))
        x = x - x.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(x, full_matrices=False)
        dim2 = C
    else:
        # Average over channels → [N, H*W]
        x = act.mean(dim=1).reshape(N, H * W)
        x = x - x.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(x, full_matrices=False)
        dim2 = H * W

    metrics[f"{prefix}_erank"] = effective_rank(S)
    metrics[f"{prefix}_vn_erank"] = von_neumann_effective_rank(S)
    metrics[f"{prefix}_stable_rank"] = stable_rank(S)
    metrics[f"{prefix}_pr"] = participation_ratio(S)
    metrics[f"{prefix}_numerical_rank_99"] = numerical_rank(S, 0.99)
    metrics[f"{prefix}_numerical_rank_95"] = numerical_rank(S, 0.95)
    metrics[f"{prefix}_renyi2"] = renyi2_rank(S)
    metrics[f"{prefix}_nuclear_rank"] = nuclear_rank(S)
    metrics[f"{prefix}_elbow_rank"] = elbow_rank(S)
    metrics[f"{prefix}_opt_threshold"] = optimal_hard_threshold(S, N, dim2)
    metrics[f"{prefix}_H_shannon"] = shannon_entropy(S, norm="L1")
    metrics[f"{prefix}_S_vN"] = von_neumann_entropy(S)
    metrics[f"{prefix}_sgr_max"] = spectral_gap_ratio(S, k=None)
    metrics[f"{prefix}_sgr_k3"] = spectral_gap_ratio(S, k=3)
    metrics[f"{prefix}_mp_outliers"] = marchenko_pastur_outlier_count(S, N, dim2)

    # Renyi spectrum: R(alpha) for multiple alphas
    for alpha in RENYI_ALPHAS:
        metrics[f"{prefix}_renyi_{alpha}"] = renyi_effective_rank(S, alpha=alpha, norm="L1")
    # alpha=inf (nuclear rank, already computed but add under renyi_ namespace)
    metrics[f"{prefix}_renyi_inf"] = nuclear_rank(S)

    return metrics


# ──────────────────────────────────────────────────────────────
# Run Analysis for a Single Model
# ──────────────────────────────────────────────────────────────

def run_null_analysis(model, temperatures, device, batch_size=200,
                      n_batches=3, console=None, model_label="model",
                      use_mcmc_samples=True):
    """
    Run rank analysis using MCMC-generated samples (not model-generated).

    For null models (random weights), we must use externally generated samples
    since the model cannot produce meaningful samples.

    For trained models, we also use MCMC samples so the input distribution
    is identical across random and trained models — this is the key control.
    """
    if console is None:
        console = Console()

    if hasattr(model, 'backbone'):
        num_layers = len(model.backbone.blocks)
    else:
        num_layers = len(model.masked_conv.hidden_convs)
    layer_keys = list(range(num_layers)) + ["final"]

    records = []
    total_samples = batch_size * n_batches

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"{model_label}", total=len(temperatures))

        for T_val in temperatures:
            beta_val = 1.0 / T_val
            progress.update(task, description=f"{model_label} T={T_val:.3f}")

            # Generate MCMC equilibrium samples
            all_samples = []
            for _ in range(n_batches):
                s = generate_mcmc_samples(T_val, batch_size, device)
                all_samples.append(s)
            all_samples = torch.cat(all_samples, dim=0)

            # Collect activations via forward pass
            activations = collect_activations(model, all_samples, T_val, device)

            # Compute rank metrics per layer
            for lk in layer_keys:
                if lk not in activations:
                    continue

                act = activations[lk]
                layer_name = f"layer_{lk}" if isinstance(lk, int) else lk

                row = {
                    "T": T_val,
                    "beta": beta_val,
                    "T_over_Tc": T_val / Tc,
                    "layer": layer_name,
                    "model": model_label,
                }
                row.update(compute_rank_metrics(act, "channel"))
                row.update(compute_rank_metrics(act, "spatial"))
                records.append(row)

            progress.advance(task)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def plot_null_comparison(df, figs_dir, layer_focus=None):
    """
    Plot eRank(T) comparison: trained vs random for both architectures.

    Main figure: 2×2 grid
      [0,0] PixelCNN channel eRank: trained vs random (per layer or avg)
      [0,1] LatticeGPT channel eRank: trained vs random
      [1,0] PixelCNN Renyi spectrum at Tc: trained vs random
      [1,1] LatticeGPT Renyi spectrum at Tc: trained vs random
    """
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    models = df["model"].unique()
    architectures = {}
    for m in models:
        if "PixelCNN" in m:
            arch = "PixelCNN"
        elif "LatticeGPT" in m or "Transformer" in m:
            arch = "LatticeGPT"
        else:
            arch = m
        if arch not in architectures:
            architectures[arch] = []
        architectures[arch].append(m)

    arch_list = sorted(architectures.keys())
    n_arch = len(arch_list)

    # ── Figure 1: Channel eRank vs Temperature ──
    fig, axes = plt.subplots(2, n_arch, figsize=(7 * n_arch, 12))
    if n_arch == 1:
        axes = axes.reshape(2, 1)

    colors = {"trained": "#2196F3", "random": "#FF5722"}
    styles = {"trained": "-", "random": "--"}

    for col_idx, arch in enumerate(arch_list):
        model_names = architectures[arch]
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        for m_name in model_names:
            m_df = df[df["model"] == m_name]
            is_random = "random" in m_name.lower()
            ckey = "random" if is_random else "trained"

            # Top: average channel eRank across layers
            layers = sorted(m_df["layer"].unique())
            avg_erank = m_df.groupby("T")["channel_erank"].mean().reset_index()
            avg_erank = avg_erank.sort_values("T")

            ax_top.plot(
                avg_erank["T"], avg_erank["channel_erank"],
                styles[ckey], color=colors[ckey], linewidth=2,
                label=m_name, alpha=0.9,
            )

            # Bottom: Renyi spectrum at closest T to Tc
            T_near_Tc = m_df["T"].unique()
            T_at_Tc = T_near_Tc[np.argmin(np.abs(T_near_Tc - Tc))]
            tc_df = m_df[(m_df["T"] == T_at_Tc) & (m_df["layer"] == layers[0])]

            if len(tc_df) > 0:
                row = tc_df.iloc[0]
                alphas = RENYI_ALPHAS + [float('inf')]
                alpha_labels = [str(a) for a in RENYI_ALPHAS] + ["∞"]
                renyi_vals = []
                for a in RENYI_ALPHAS:
                    renyi_vals.append(row.get(f"channel_renyi_{a}", np.nan))
                renyi_vals.append(row.get("channel_renyi_inf", np.nan))

                x_pos = list(range(len(alphas)))
                ax_bot.plot(
                    x_pos, renyi_vals,
                    "o-", color=colors[ckey], linewidth=2, markersize=6,
                    label=m_name, alpha=0.9,
                )
                ax_bot.set_xticks(x_pos)
                ax_bot.set_xticklabels(alpha_labels)

        ax_top.axvline(Tc, color="red", ls="--", alpha=0.5, lw=1, label=f"$T_c$={Tc:.3f}")
        ax_top.set_xlabel("Temperature $T$")
        ax_top.set_ylabel("Channel eRank (avg over layers)")
        ax_top.set_title(f"{arch}: eRank vs Temperature")
        ax_top.legend(fontsize=8)
        ax_top.grid(True, alpha=0.3)

        ax_bot.set_xlabel(r"Rényi order $\alpha$")
        ax_bot.set_ylabel(r"$R(\alpha, L1)$")
        ax_bot.set_title(f"{arch}: Rényi Spectrum at $T \\approx T_c$")
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3)

    fig.suptitle("Phase 0: Null Model eRank Comparison (MCMC samples)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figs_dir / "null_model_erank_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Per-layer eRank at Tc ──
    fig2, axes2 = plt.subplots(1, n_arch, figsize=(7 * n_arch, 5))
    if n_arch == 1:
        axes2 = [axes2]

    for col_idx, arch in enumerate(arch_list):
        ax = axes2[col_idx]
        model_names = architectures[arch]

        for m_name in model_names:
            m_df = df[df["model"] == m_name]
            is_random = "random" in m_name.lower()
            ckey = "random" if is_random else "trained"

            T_near_Tc = m_df["T"].unique()
            T_at_Tc = T_near_Tc[np.argmin(np.abs(T_near_Tc - Tc))]
            tc_df = m_df[m_df["T"] == T_at_Tc].sort_values("layer")

            layers = tc_df["layer"].values
            eranks = tc_df["channel_erank"].values

            x_pos = list(range(len(layers)))
            ax.bar(
                [x + (0.2 if is_random else -0.2) for x in x_pos],
                eranks, width=0.35,
                color=colors[ckey], alpha=0.7, label=m_name,
            )

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([l.replace("layer_", "L").replace("final", "F") for l in layers],
                           fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Channel eRank")
        ax.set_title(f"{arch}: Per-Layer eRank at $T \\approx T_c$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig2.tight_layout()
    fig2.savefig(figs_dir / "null_model_per_layer_at_Tc.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Figure 3: Trained eRank ratio (trained/random) ──
    fig3, axes3 = plt.subplots(1, n_arch, figsize=(7 * n_arch, 5))
    if n_arch == 1:
        axes3 = [axes3]

    for col_idx, arch in enumerate(arch_list):
        ax = axes3[col_idx]
        model_names = architectures[arch]

        trained_names = [m for m in model_names if "random" not in m.lower()]
        random_names = [m for m in model_names if "random" in m.lower()]

        if not trained_names or not random_names:
            continue

        t_df = df[df["model"] == trained_names[0]].groupby("T")["channel_erank"].mean().reset_index()
        r_df = df[df["model"] == random_names[0]].groupby("T")["channel_erank"].mean().reset_index()

        merged = t_df.merge(r_df, on="T", suffixes=("_trained", "_random"))
        merged["ratio"] = merged["channel_erank_trained"] / merged["channel_erank_random"]

        ax.plot(merged["T"], merged["ratio"], "o-", color="#4CAF50", linewidth=2, markersize=4)
        ax.axvline(Tc, color="red", ls="--", alpha=0.5, lw=1)
        ax.axhline(1.0, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("Temperature $T$")
        ax.set_ylabel("eRank ratio (trained / random)")
        ax.set_title(f"{arch}: eRank Ratio")
        ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(figs_dir / "null_model_erank_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    return [
        figs_dir / "null_model_erank_comparison.png",
        figs_dir / "null_model_per_layer_at_Tc.png",
        figs_dir / "null_model_erank_ratio.png",
    ]


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 0: Null Model eRank Analysis")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--n_batches", type=int, default=3)
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="Number of random initializations for null models")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer temperatures and samples")
    parser.add_argument("--replot", type=str, default=None,
                        help="Replot from existing CSV")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    console = Console()

    # Replot mode
    if args.replot:
        df = pd.read_csv(args.replot)
        figs_dir = Path(args.replot).parent / "figs"
        console.print(f"[green]Replotting from {args.replot}[/green]")
        plot_null_comparison(df, figs_dir)
        console.print(f"[green]Figures saved to {figs_dir}[/green]")
        return

    device = args.device
    if args.quick:
        batch_size = 100
        n_batches = 2
        n_temps_coarse = 15
        n_temps_critical = 8
    else:
        batch_size = args.batch_size
        n_batches = args.n_batches
        n_temps_coarse = 25
        n_temps_critical = 15

    # Temperature grid (use wider range covering both model configs)
    temperatures = temperature_grid(
        T_min=1.0, T_max=5.0,
        n_coarse=n_temps_coarse,
        n_critical=n_temps_critical,
        Tc=Tc,
    )

    console.print(f"\n[bold]Phase 0: Null Model eRank Analysis[/bold]")
    console.print(f"  Temperatures: {len(temperatures)} points in [{temperatures[0]:.2f}, {temperatures[-1]:.2f}]")
    console.print(f"  Samples per T: {batch_size * n_batches}")
    console.print(f"  Random seeds: {args.n_seeds}")
    console.print(f"  Device: {device}")
    console.print()

    all_dfs = []

    # ── 1. Load trained models ──
    console.print("[bold cyan]Loading trained PixelCNN...[/bold cyan]")
    pixelcnn_trained, pixelcnn_config = load_model(
        PIXELCNN_PROJECT, PIXELCNN_GROUP, PIXELCNN_SEED
    )
    pixelcnn_trained = pixelcnn_trained.to(device).eval()

    console.print("[bold cyan]Loading trained LatticeGPT...[/bold cyan]")
    latticegpt_trained, latticegpt_config = load_model(
        LATTICEGPT_PROJECT, LATTICEGPT_GROUP, LATTICEGPT_SEED
    )
    latticegpt_trained = latticegpt_trained.to(device).eval()

    # ── 2. Run analysis on trained models (with MCMC samples) ──
    console.print("\n[bold green]Analyzing trained PixelCNN...[/bold green]")
    df_pcnn_trained = run_null_analysis(
        pixelcnn_trained, temperatures, device,
        batch_size=batch_size, n_batches=n_batches,
        console=console, model_label="PixelCNN (trained)",
    )
    all_dfs.append(df_pcnn_trained)

    console.print("\n[bold green]Analyzing trained LatticeGPT...[/bold green]")
    df_lgpt_trained = run_null_analysis(
        latticegpt_trained, temperatures, device,
        batch_size=batch_size, n_batches=n_batches,
        console=console, model_label="LatticeGPT (trained)",
    )
    all_dfs.append(df_lgpt_trained)

    # ── 3. Create and analyze random-weight models ──
    for seed_idx in range(args.n_seeds):
        random_seed = 1000 + seed_idx
        torch.manual_seed(random_seed)

        console.print(f"\n[bold yellow]Random PixelCNN (seed {random_seed})...[/bold yellow]")
        random_pixelcnn = pixelcnn_config.create_model().to(device).eval()
        df_pcnn_random = run_null_analysis(
            random_pixelcnn, temperatures, device,
            batch_size=batch_size, n_batches=n_batches,
            console=console,
            model_label=f"PixelCNN (random s{random_seed})",
        )
        all_dfs.append(df_pcnn_random)
        del random_pixelcnn
        torch.cuda.empty_cache()

        torch.manual_seed(random_seed)
        console.print(f"\n[bold yellow]Random LatticeGPT (seed {random_seed})...[/bold yellow]")
        random_latticegpt = latticegpt_config.create_model().to(device).eval()
        df_lgpt_random = run_null_analysis(
            random_latticegpt, temperatures, device,
            batch_size=batch_size, n_batches=n_batches,
            console=console,
            model_label=f"LatticeGPT (random s{random_seed})",
        )
        all_dfs.append(df_lgpt_random)
        del random_latticegpt
        torch.cuda.empty_cache()

    # ── 4. Combine and average random seeds ──
    df_all = pd.concat(all_dfs, ignore_index=True)

    # Create averaged random model entries
    for arch in ["PixelCNN", "LatticeGPT"]:
        random_mask = df_all["model"].str.contains(f"{arch}.*random", regex=True)
        if random_mask.any():
            random_df = df_all[random_mask].copy()
            group_keys = ["T", "beta", "T_over_Tc", "layer"]
            numeric_cols = [c for c in random_df.select_dtypes(include=[np.number]).columns
                           if c not in group_keys]
            avg_df = random_df.groupby(group_keys)[numeric_cols].mean().reset_index()
            avg_df["model"] = f"{arch} (random avg)"
            df_all = pd.concat([df_all, avg_df], ignore_index=True)

    # ── 5. Output ──
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs/null_model_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "null_model_erank.csv"
    df_all.to_csv(csv_path, index=False)
    console.print(f"\n[green]Results saved to {csv_path}[/green]")

    # Plot comparison (using averaged random models)
    plot_df = df_all[~df_all["model"].str.contains("random s", regex=False)]
    figs_dir = output_dir / "figs"
    plot_null_comparison(plot_df, figs_dir)
    console.print(f"[green]Figures saved to {figs_dir}/[/green]")

    # ── 6. Summary statistics ──
    console.print("\n[bold]═══ Phase 0 Summary ═══[/bold]")

    for arch in ["PixelCNN", "LatticeGPT"]:
        trained = plot_df[plot_df["model"] == f"{arch} (trained)"]
        random = plot_df[plot_df["model"] == f"{arch} (random avg)"]

        if trained.empty or random.empty:
            continue

        # eRank at closest T to Tc (average over layers)
        T_near = trained["T"].unique()
        T_at_Tc = T_near[np.argmin(np.abs(T_near - Tc))]

        er_trained_Tc = trained[trained["T"] == T_at_Tc]["channel_erank"].mean()
        er_random_Tc = random[random["T"] == T_at_Tc]["channel_erank"].mean()

        # eRank at high T (T > 3)
        er_trained_hi = trained[trained["T"] > 3.0]["channel_erank"].mean()
        er_random_hi = random[random["T"] > 3.0]["channel_erank"].mean()

        # Dip depth
        dip_trained = er_trained_hi - er_trained_Tc if er_trained_hi > er_trained_Tc else 0
        dip_random = er_random_hi - er_random_Tc if er_random_hi > er_random_Tc else 0

        console.print(f"\n  [bold]{arch}[/bold]")
        console.print(f"    eRank at T≈Tc:  trained={er_trained_Tc:.2f}  random={er_random_Tc:.2f}")
        console.print(f"    eRank at T>3:   trained={er_trained_hi:.2f}  random={er_random_hi:.2f}")
        console.print(f"    Dip depth:      trained={dip_trained:.2f}  random={dip_random:.2f}")
        ratio = dip_random / dip_trained if dip_trained > 0 else float('inf')
        console.print(f"    Ratio (random/trained dip): {ratio:.2f}")

        if ratio > 0.5:
            console.print(f"    [red]⚠ Tier 1 WARNING: Random {arch} shows >{ratio*100:.0f}% of trained dip![/red]")
            console.print(f"    [red]  → eRank dip may be a geometric artifact of local connectivity[/red]")
        elif ratio > 0.2:
            console.print(f"    [yellow]  Tier 1 partial: Random shows {ratio*100:.0f}% of trained dip[/yellow]")
            console.print(f"    [yellow]  → Some geometric contribution; learning adds significant structure[/yellow]")
        else:
            console.print(f"    [green]  Tier 1 cleared: Random shows only {ratio*100:.0f}% of trained dip[/green]")
            console.print(f"    [green]  → eRank dip is predominantly a learned phenomenon[/green]")

    console.print("\n[bold green]Phase 0 complete.[/bold green]")


if __name__ == "__main__":
    main()
