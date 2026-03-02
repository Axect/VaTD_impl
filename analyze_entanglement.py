"""
Neural Entanglement Entropy Analysis (Result 2).

Measures the entanglement entropy of neural network activations by computing
the cross-covariance SVD entropy between spatial bipartitions. At the critical
temperature, this should follow the Calabrese-Cardy (2004) formula:

    S_E(l) = (c/3) * ln((L/π) * sin(πl/L)) + const

where c is the CFT central charge and l is the subsystem size.

This allows extraction of the central charge c directly from the trained
autoregressive model's internal representations.

Usage:
    python analyze_entanglement.py --project <proj> --group <group> --seed 42 --device cuda:0
    python analyze_entanglement.py  # Interactive mode
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
from scipy.optimize import curve_fit
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn,
)
import argparse

from util import select_project, select_group, select_seed, select_device, load_model
from analyze_rank import get_critical_temperature, collect_activations


# ──────────────────────────────────────────────────────────────
# Calabrese-Cardy Formula
# ──────────────────────────────────────────────────────────────


def calabrese_cardy(l, L, c, const):
    """
    Calabrese-Cardy formula for entanglement entropy on a periodic chain.

    S(l) = (c/3) * ln((L/π) * sin(πl/L)) + const

    Args:
        l: subsystem size (number of columns)
        L: total system size
        c: central charge
        const: non-universal additive constant
    """
    return (c / 3.0) * np.log((L / np.pi) * np.sin(np.pi * l / L)) + const


# ──────────────────────────────────────────────────────────────
# Cross-Covariance SVD Entropy
# ──────────────────────────────────────────────────────────────


def compute_cross_covariance_entropy(activations, l, L):
    """
    Compute entanglement entropy via cross-covariance SVD.

    1. Split activations spatially: A = first l columns, B = remaining L-l columns
    2. Flatten: A → [N, C·H·l], B → [N, C·H·(L-l)]
    3. Center both
    4. Cross-covariance: C_AB = A^T @ B / N
    5. SVD → singular values → Shannon entropy

    Args:
        activations: [N, C, H, W] tensor
        l: number of columns in subsystem A
        L: total width (W dimension)

    Returns:
        S_E: entanglement entropy (scalar)
        singular_values: raw SVD singular values
    """
    N, C, H, W = activations.shape
    assert W == L, f"Width {W} != lattice size {L}"
    assert 0 < l < L, f"Partition size l={l} must be in (0, {L})"

    # Split into subsystems A and B
    act_A = activations[:, :, :, :l].reshape(N, C * H * l)        # [N, C·H·l]
    act_B = activations[:, :, :, l:].reshape(N, C * H * (L - l))  # [N, C·H·(L-l)]

    # Center
    act_A = act_A - act_A.mean(dim=0, keepdim=True)
    act_B = act_B - act_B.mean(dim=0, keepdim=True)

    # Cross-covariance matrix
    C_AB = act_A.T @ act_B / N  # [C·H·l, C·H·(L-l)]

    # SVD
    _, S, _ = torch.linalg.svd(C_AB, full_matrices=False)

    # Shannon entropy of normalized singular values
    S_pos = S[S > 1e-10]
    if len(S_pos) == 0:
        return 0.0, S

    p = S_pos / S_pos.sum()
    entropy = -(p * torch.log(p)).sum().item()

    return entropy, S


# ──────────────────────────────────────────────────────────────
# Analysis Loop
# ──────────────────────────────────────────────────────────────


def collect_layer_activations(model, samples, T_val, device, layer_idx=None):
    """
    Collect activations from a specific layer (or all layers).

    Uses the same hook infrastructure as analyze_rank.py.

    Args:
        model: DiscretePixelCNN in eval mode
        samples: [B, 1, H, W] in {-1, +1}
        T_val: temperature scalar
        device: torch device
        layer_idx: if int, return only that layer's activations;
                   if None, return all (same as collect_activations)

    Returns:
        Tensor [N, C, H, W] for the requested layer
    """
    acts = collect_activations(model, samples, T_val, device)

    if layer_idx is not None:
        if layer_idx in acts:
            return acts[layer_idx]
        elif f"layer_{layer_idx}" in acts:
            return acts[f"layer_{layer_idx}"]
        else:
            raise KeyError(f"Layer {layer_idx} not found. Available: {list(acts.keys())}")

    return acts


def run_entanglement_analysis(
    model, Tc, L, device, temperatures=None, batch_size=600,
    layer_indices=None, console=None,
):
    """
    Main entanglement entropy analysis.

    For each temperature and layer:
    1. Generate samples
    2. Collect activations
    3. For each bipartition l = 1, ..., L/2:
       compute cross-covariance entropy S_E(l)
    4. Fit Calabrese-Cardy formula → extract c

    Args:
        model: trained DiscretePixelCNN
        Tc: critical temperature
        L: lattice size
        device: torch device
        temperatures: list of temperatures (default: [Tc, 2*Tc, 5*Tc])
        batch_size: number of samples
        layer_indices: which layers to analyze (default: all)
        console: Rich console

    Returns:
        DataFrame with columns: T, layer, l, S_E, c_fit, c_err, R2
    """
    if console is None:
        console = Console()

    if temperatures is None:
        temperatures = [
            0.5 * Tc,   # ordered phase
            Tc,          # critical point
            2.0 * Tc,    # disordered phase
            5.0 * Tc,    # deep disordered
        ]

    # Determine layers to analyze
    num_layers = len(model.masked_conv.hidden_convs)
    if layer_indices is None:
        layer_indices = list(range(num_layers))

    l_values = list(range(1, L // 2 + 1))  # l = 1, 2, ..., L/2

    records = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        total = len(temperatures) * len(layer_indices)
        task = progress.add_task("Entanglement analysis", total=total)

        for T_val in temperatures:
            beta = 1.0 / T_val
            progress.update(task, description=f"T={T_val:.3f} (β={beta:.3f})")

            # Generate samples
            model.eval()
            T_tensor = torch.full((batch_size,), T_val, device=device)
            with torch.no_grad():
                samples = model.sample(batch_size=batch_size, T=T_tensor)

            # Collect activations from all layers at once
            all_acts = collect_activations(model, samples, T_val, device)

            for li in layer_indices:
                if li in all_acts:
                    act = all_acts[li]
                elif "final" in all_acts and li == num_layers:
                    act = all_acts["final"]
                else:
                    progress.advance(task)
                    continue

                layer_name = f"layer_{li}" if isinstance(li, int) else str(li)

                # Compute S_E for each bipartition
                S_E_values = []
                for l in l_values:
                    S_E, _ = compute_cross_covariance_entropy(act, l, L)
                    S_E_values.append(S_E)

                    records.append({
                        "T": T_val,
                        "beta": beta,
                        "T_over_Tc": T_val / Tc,
                        "layer": layer_name,
                        "l": l,
                        "S_E": S_E,
                    })

                # Fit Calabrese-Cardy
                S_E_arr = np.array(S_E_values)
                l_arr = np.array(l_values, dtype=float)

                try:
                    popt, pcov = curve_fit(
                        lambda l_fit, c, const: calabrese_cardy(l_fit, L, c, const),
                        l_arr, S_E_arr,
                        p0=[0.5, S_E_arr.mean()],
                        bounds=([0.0, -100], [10.0, 100]),
                        maxfev=5000,
                    )
                    c_fit, const_fit = popt
                    c_err = np.sqrt(np.diag(pcov))[0]

                    # R² goodness of fit
                    S_pred = calabrese_cardy(l_arr, L, c_fit, const_fit)
                    ss_res = np.sum((S_E_arr - S_pred) ** 2)
                    ss_tot = np.sum((S_E_arr - S_E_arr.mean()) ** 2)
                    R2 = 1 - ss_res / (ss_tot + 1e-15)
                except (RuntimeError, ValueError):
                    c_fit, c_err, R2 = np.nan, np.nan, np.nan

                # Store fit result for this T/layer combination
                for rec in records[-len(l_values):]:
                    rec["c_fit"] = c_fit
                    rec["c_err"] = c_err
                    rec["R2"] = R2

                progress.advance(task)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────


def plot_entanglement_entropy(df, L, figs_dir, q=2, Tc=None):
    """
    Publication figure for entanglement entropy (Result 2).

    (a) S_E(l) data + Calabrese-Cardy fit at Tc for each layer
    (b) Extracted c values across temperatures (all layers)
    """
    from potts_exact_partition import CENTRAL_CHARGES

    c_exact = CENTRAL_CHARGES.get(q, None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel (a): S_E(l) at Tc ──
    if Tc is not None:
        df_Tc = df[df["T"].between(0.95 * Tc, 1.05 * Tc)]
    else:
        df_Tc = df[df["T_over_Tc"].between(0.95, 1.05)]

    if len(df_Tc) == 0:
        # Use nearest to Tc
        df_Tc = df[df["T_over_Tc"] == df["T_over_Tc"].sub(1.0).abs().min() + 1.0]

    layers = sorted(df_Tc["layer"].unique())
    n_layers = len(layers)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_layers, 1)))

    for ci, layer in enumerate(layers):
        ld = df_Tc[df_Tc["layer"] == layer].sort_values("l")
        ax1.plot(
            ld["l"], ld["S_E"],
            "o", color=colors[ci], markersize=5,
            label=layer.replace("layer_", "Block "), alpha=0.8,
        )

        # Overlay CC fit if available
        c_fit = ld["c_fit"].iloc[0]
        R2 = ld["R2"].iloc[0]
        if not np.isnan(c_fit):
            l_smooth = np.linspace(0.5, L / 2, 100)
            S_fit = calabrese_cardy(l_smooth, L, c_fit, ld["S_E"].mean() - (c_fit / 3) * np.log(L / np.pi * np.sin(np.pi * ld["l"].mean() / L)))

            # Refit with exact const
            try:
                from scipy.optimize import curve_fit as cf
                popt, _ = cf(
                    lambda l_f, c, const: calabrese_cardy(l_f, L, c, const),
                    ld["l"].values.astype(float), ld["S_E"].values,
                    p0=[c_fit, ld["S_E"].mean()],
                    bounds=([0.0, -100], [10.0, 100]),
                )
                S_fit = calabrese_cardy(l_smooth, L, popt[0], popt[1])
                ax1.plot(l_smooth, S_fit, "-", color=colors[ci], alpha=0.5, linewidth=1.2)
            except Exception:
                pass

    ax1.set_xlabel("Subsystem size $\\ell$", fontsize=12)
    ax1.set_ylabel("$S_E(\\ell)$", fontsize=12)
    T_label = f"$T \\approx T_c$" if Tc else "$T/T_c \\approx 1$"
    ax1.set_title(f"(a) Entanglement entropy at {T_label}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.15)

    # ── Panel (b): Extracted c across temperatures ──
    # Get one c_fit per (T, layer) combination
    c_summary = df.groupby(["T", "T_over_Tc", "layer"]).first()[["c_fit", "c_err", "R2"]].reset_index()

    for ci, layer in enumerate(sorted(c_summary["layer"].unique())):
        ld = c_summary[c_summary["layer"] == layer].sort_values("T_over_Tc")
        mask = ld["R2"] > 0.5  # Only show reasonable fits
        if mask.sum() == 0:
            continue
        ax2.errorbar(
            ld.loc[mask, "T_over_Tc"], ld.loc[mask, "c_fit"],
            yerr=ld.loc[mask, "c_err"].clip(upper=2.0),
            fmt="o-", markersize=4, linewidth=1.2, capsize=3,
            label=layer.replace("layer_", "Block "), alpha=0.8,
        )

    if c_exact is not None:
        ax2.axhline(c_exact, color="red", ls="--", alpha=0.6, lw=1.5,
                     label=f"CFT exact: $c = {c_exact}$")

    ax2.axvline(1.0, color="gray", ls="--", alpha=0.4, lw=1, label="$T/T_c = 1$")
    ax2.set_xlabel("$T / T_c$", fontsize=12)
    ax2.set_ylabel("Extracted central charge $c$", fontsize=12)
    ax2.set_title("(b) Central charge vs temperature", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.15)
    ax2.set_ylim(-0.5, max(3.0, 2 * c_exact if c_exact else 3.0))

    model_label = "Ising" if q == 2 else f"{q}-Potts"
    fig.suptitle(
        f"Neural Entanglement Entropy: {model_label} ($L={L}$)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    path = Path(figs_dir) / "entanglement_entropy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Neural Entanglement Entropy Analysis"
    )
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=600,
                        help="Number of samples per temperature (default: 600)")
    parser.add_argument("--extra_temps", action="store_true",
                        help="Add more temperatures around Tc for detailed c(T) profile")
    args = parser.parse_args()

    console = Console()
    console.print("[bold green]Neural Entanglement Entropy Analysis (Result 2)[/bold green]")

    # Model selection
    project = args.project if args.project else select_project()
    group_name = args.group if args.group else select_group(project)
    seed = args.seed if args.seed else select_seed(project, group_name)
    device = args.device if args.device else select_device()

    console.print(f"\n[bold]Loading:[/bold] {project}/{group_name}/{seed}")
    model, config = load_model(project, group_name, seed)
    model = model.to(device)
    model.eval()

    Tc, q = get_critical_temperature(config)
    model_label = "2D Ising" if q == 2 else f"{q}-state Potts"
    console.print(f"Model type: {model_label}, Tc = {Tc:.4f}")

    if hasattr(model, 'use_pytorch_mhc'):
        model.use_pytorch_mhc()

    L = model.size[0]
    console.print(f"Lattice: {L}x{L}")

    # Temperature grid
    if args.extra_temps:
        temperatures = [
            0.5 * Tc, 0.7 * Tc, 0.9 * Tc,
            0.95 * Tc, Tc, 1.05 * Tc, 1.1 * Tc,
            1.3 * Tc, 2.0 * Tc, 5.0 * Tc,
        ]
    else:
        temperatures = [0.5 * Tc, Tc, 2.0 * Tc, 5.0 * Tc]

    console.print(f"Temperatures: {', '.join(f'{T:.3f}' for T in temperatures)}")
    console.print(f"Samples per temperature: {args.batch_size}")

    # Run analysis
    console.print("\n[bold cyan]Running entanglement entropy analysis[/bold cyan]")
    df = run_entanglement_analysis(
        model, Tc, L, device,
        temperatures=temperatures,
        batch_size=args.batch_size,
        console=console,
    )

    # Save data
    output_dir = Path(f"runs/{project}/{group_name}")
    csv_path = output_dir / f"entanglement_analysis_{seed}.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"[green]Data saved:[/green] {csv_path}")

    # Generate plots
    figs_dir = Path(f"figs/{group_name}")
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig_path = plot_entanglement_entropy(df, L, figs_dir, q=q, Tc=Tc)
    console.print(f"[green]Entanglement plot:[/green] {fig_path}")

    # Summary
    console.print("\n[bold cyan]Central Charge Extraction Summary[/bold cyan]")
    c_summary = df.groupby(["T", "layer"]).first()[["c_fit", "c_err", "R2"]].reset_index()

    from potts_exact_partition import CENTRAL_CHARGES
    c_exact = CENTRAL_CHARGES.get(q, None)

    if c_exact:
        console.print(f"CFT exact value: c = {c_exact}")

    console.print(f"\n{'T':>8s}  {'Layer':>10s}  {'c_fit':>8s}  {'c_err':>8s}  {'R²':>6s}")
    console.print("-" * 48)
    for _, row in c_summary.sort_values(["T", "layer"]).iterrows():
        console.print(
            f"{row['T']:>8.3f}  {row['layer']:>10s}  {row['c_fit']:>8.3f}  "
            f"{row['c_err']:>8.3f}  {row['R2']:>6.3f}"
        )

    # Best estimate: c from Tc with highest R²
    tc_fits = c_summary[
        (c_summary["T"].between(0.95 * Tc, 1.05 * Tc)) &
        (c_summary["R2"] > 0.5)
    ]
    if len(tc_fits) > 0:
        best = tc_fits.loc[tc_fits["R2"].idxmax()]
        console.print(
            f"\n[bold]Best estimate at Tc:[/bold] c = {best['c_fit']:.3f} ± {best['c_err']:.3f} "
            f"(R² = {best['R2']:.3f}, layer: {best['layer']})"
        )
        if c_exact:
            deviation = abs(best['c_fit'] - c_exact) / c_exact * 100
            console.print(f"Deviation from exact: {deviation:.1f}%")

    console.print(f"\n[bold green]Analysis complete.[/bold green]")


if __name__ == "__main__":
    main()
