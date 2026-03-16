import os
os.environ['VATD_NO_MHC'] = '1'  # Prevent mHC.cu CUDA extension from loading

import math
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import beaupy
from rich.console import Console
from pathlib import Path

from util import (
    select_project,
    select_group,
    select_seed,
    select_device,
    load_model,
    _plot_clock_sample,
)
from clock import (
    create_clock_energy_fn,
    CLOCK_TC,
    CLOCK_CENTRAL_CHARGE,
    compute_helicity_modulus,
    compute_vortex_density,
)


def compute_clock_thermodynamics(
    model,
    energy_fn,
    q,
    L,
    device="cpu",
    num_temps=20,
    batch_size=1000,
    T_min=None,
    T_max=None,
    verbose=True,
):
    """
    Compute thermodynamic quantities for the clock model using sampling only.

    No exact partition function is available for the clock model, so all quantities
    are computed via Monte Carlo sampling from the trained model.

    Observables computed per temperature:
      - Mean energy: <E> = mean(energy(samples))
      - Heat capacity: Cv = beta^2 * Var(E)
      - Vector magnetization: M = (1/N)|sum exp(i*2*pi*s/q)|, averaged over batch
      - Susceptibility: chi = beta * N * (<M^2> - <M>^2)
      - Helicity modulus: Upsilon (order parameter for BKT transition)
      - Vortex density: fraction of plaquettes with non-zero winding number

    Args:
        model: Trained model (DiscretePixelCNN)
        energy_fn: Clock energy function
        q: Number of clock states
        L: Lattice size
        device: Device to run on
        num_temps: Number of temperature points
        batch_size: Samples per temperature (larger for better statistics)
        T_min: Minimum temperature (default: 0.5 * T_BKT)
        T_max: Maximum temperature (default: 1.5 * T_BKT)
        verbose: Print progress information

    Returns:
        pandas.DataFrame with columns:
            - T, beta, T/T_BKT: Temperature parameters
            - E_mean: Mean energy per site
            - Cv: Heat capacity per site
            - M_mean: Mean vector magnetization
            - chi: Magnetic susceptibility
            - helicity: Helicity modulus (Upsilon)
            - vortex_density: Fraction of plaquettes with vortices
    """
    T_BKT = CLOCK_TC.get(q, 0.89)

    if T_min is None:
        T_min = 0.5 * T_BKT
    if T_max is None:
        T_max = 1.5 * T_BKT

    T_values = np.linspace(T_min, T_max, num_temps)
    N = L * L
    angle_step = 2.0 * math.pi / q

    results = []

    model.eval()
    with torch.no_grad():
        for idx, T in enumerate(T_values):
            T = float(T)
            if verbose and idx % 5 == 0:
                print(f"    Processing temperature {idx+1}/{num_temps} (T={T:.4f})...")

            beta = 1.0 / T
            T_tensor = torch.full((batch_size,), T, device=device)

            # Sample from model
            samples = model.sample(batch_size=batch_size, T=T_tensor)
            energies = energy_fn(samples).squeeze()  # (batch_size,)

            # Mean energy and heat capacity
            E_mean = energies.mean().item()
            E2_mean = (energies ** 2).mean().item()
            Cv = (beta ** 2) * (E2_mean - E_mean ** 2)

            # Per-sample vector magnetization for susceptibility
            # M_i = (1/N) * |sum_j exp(i * 2*pi*s_j/q)|
            angles = samples[:, 0].float() * angle_step  # (B, H, W)
            cos_sum = torch.cos(angles).sum(dim=[-1, -2])  # (B,)
            sin_sum = torch.sin(angles).sum(dim=[-1, -2])  # (B,)
            M_per_sample = torch.sqrt(cos_sum ** 2 + sin_sum ** 2) / N  # (B,)

            M_mean = M_per_sample.mean().item()
            M2_mean = (M_per_sample ** 2).mean().item()

            # Susceptibility: chi = beta * N * (<M^2> - <M>^2)
            chi = beta * N * (M2_mean - M_mean ** 2)

            # Helicity modulus (needs T tensor for beta calculation)
            helicity = compute_helicity_modulus(samples, T_tensor, q)

            # Vortex density
            vortex_density = compute_vortex_density(samples, q)

            results.append(
                {
                    "T": T,
                    "beta": beta,
                    "T/T_BKT": T / T_BKT,
                    "E_mean": E_mean,
                    "Cv": Cv,
                    "M_mean": M_mean,
                    "chi": chi,
                    "helicity": helicity,
                    "vortex_density": vortex_density,
                }
            )

    return pd.DataFrame(results)


def visualize_clock_samples(
    model,
    q,
    L,
    temperatures,
    device="cpu",
    num_samples=4,
    output_dir="figs",
    filename="clock_samples.png",
):
    """
    Visualize clock model samples at multiple temperatures using quiver arrows.

    Creates a grid of subplots with n_temps rows x num_samples columns.
    Each cell shows a single lattice sample with HSV background colors and
    arrow overlays indicating the spin direction at each site.

    Row labels indicate T and T/T_BKT for easy reference.

    Args:
        model: Trained clock model
        q: Number of clock states
        L: Lattice size
        temperatures: List of temperatures to visualize
        device: Device to run on
        num_samples: Number of samples per temperature (columns)
        output_dir: Directory to save the plot
        filename: Output filename

    Returns:
        str: Path to the saved figure
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    T_BKT = CLOCK_TC.get(q, 0.89)
    n_temps = len(temperatures)

    # Clock samples need larger cells to display arrows clearly
    cell_size = 2.5
    fig, axes = plt.subplots(
        n_temps, num_samples,
        figsize=(num_samples * cell_size + 1.2, n_temps * cell_size),
    )

    # Normalize axes shape to always be 2D
    if n_temps == 1 and num_samples == 1:
        axes = np.array([[axes]])
    elif n_temps == 1:
        axes = axes.reshape(1, -1)
    elif num_samples == 1:
        axes = axes.reshape(-1, 1)

    model.eval()
    with torch.no_grad():
        for i, T in enumerate(temperatures):
            T_tensor = torch.full((num_samples,), float(T), device=device)
            samples = model.sample(batch_size=num_samples, T=T_tensor)

            for j in range(num_samples):
                ax = axes[i, j]
                sample_np = samples[j, 0].cpu().numpy()  # (H, W)
                _plot_clock_sample(ax, sample_np, q)

                if j == 0:
                    ax.set_ylabel(
                        f"T={T:.3f}\n(T/T_BKT={T/T_BKT:.2f})",
                        fontsize=9,
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=55,
                    )

                if i == 0:
                    ax.set_title(f"Sample {j+1}", fontsize=9)

    plt.suptitle(
        f"{q}-state Clock Model Samples (arrows = spin direction, color = angle)",
        fontsize=11,
    )
    plt.tight_layout()

    output_path = f"{output_dir}/{filename}"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_clock_thermodynamics(
    df,
    q,
    output_dir="figs",
    filename="clock_thermodynamics.png",
    title_suffix="",
):
    """
    Generate comprehensive thermodynamic plots for the clock model.

    Creates a 2x3 figure:
      - (0,0): Mean energy vs T
      - (0,1): Heat capacity vs T
      - (0,2): Vector magnetization vs T
      - (1,0): Susceptibility vs T
      - (1,1): Helicity modulus vs T with (2/pi)*T_BKT universal jump line
      - (1,2): Vortex density vs T

    All panels include a vertical dashed red line at T_BKT.

    Args:
        df: DataFrame from compute_clock_thermodynamics()
        q: Number of clock states
        output_dir: Directory to save the plot
        filename: Output filename
        title_suffix: Suffix appended to all subplot titles

    Returns:
        str: Path to the saved figure
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    T_BKT = CLOCK_TC.get(q, 0.89)
    # Universal helicity modulus jump at BKT transition
    helicity_jump = 2.0 / math.pi * T_BKT

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def add_tbkt_line(ax, label=True):
        ax.axvline(
            T_BKT,
            color="r",
            linestyle="--",
            alpha=0.6,
            label=f"T_BKT = {T_BKT:.3f}" if label else None,
        )

    # ============================================
    # (0,0): Mean Energy
    # ============================================
    ax = axes[0, 0]
    ax.plot(df["T"], df["E_mean"], "o-", color="steelblue", markersize=4)
    add_tbkt_line(ax)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Mean Energy <E>")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Mean Energy{title_suffix}")

    # ============================================
    # (0,1): Heat Capacity
    # ============================================
    ax = axes[0, 1]
    ax.plot(df["T"], df["Cv"], "o-", color="darkorange", markersize=4)
    add_tbkt_line(ax)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Heat Capacity Cv")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Heat Capacity{title_suffix}")

    # ============================================
    # (0,2): Magnetization
    # ============================================
    ax = axes[0, 2]
    ax.plot(df["T"], df["M_mean"], "o-", color="mediumseagreen", markersize=4)
    ax.axhline(0, color="k", linestyle="-", alpha=0.2)
    add_tbkt_line(ax)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Vector Magnetization <M>")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Magnetization{title_suffix}")

    # ============================================
    # (1,0): Susceptibility
    # ============================================
    ax = axes[1, 0]
    ax.plot(df["T"], df["chi"], "o-", color="mediumpurple", markersize=4)
    add_tbkt_line(ax)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Susceptibility chi")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Susceptibility{title_suffix}")

    # ============================================
    # (1,1): Helicity Modulus with universal jump reference
    # ============================================
    ax = axes[1, 1]
    ax.plot(df["T"], df["helicity"], "o-", color="firebrick", markersize=4, label="Model")
    ax.axhline(
        helicity_jump,
        color="navy",
        linestyle=":",
        alpha=0.7,
        label=f"(2/pi)*T_BKT = {helicity_jump:.3f}",
    )
    add_tbkt_line(ax)
    ax.axhline(0, color="k", linestyle="-", alpha=0.2)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Helicity Modulus Upsilon")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Helicity Modulus{title_suffix}")

    # ============================================
    # (1,2): Vortex Density
    # ============================================
    ax = axes[1, 2]
    ax.plot(df["T"], df["vortex_density"], "o-", color="saddlebrown", markersize=4)
    add_tbkt_line(ax)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Vortex Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Vortex Density{title_suffix}")

    plt.suptitle(
        f"{q}-state Clock Model Thermodynamics  "
        f"(c = {CLOCK_CENTRAL_CHARGE}, T_BKT = {T_BKT:.3f})",
        fontsize=13,
    )
    plt.tight_layout()

    output_path = f"{output_dir}/{filename}"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_mcmc_comparison(
    df,
    mcmc_ref,
    q,
    output_dir="figs",
    filename="clock_mcmc_comparison.png",
):
    """
    Overlay model thermodynamic quantities against MCMC reference data.

    Creates a 2x1 figure comparing energy and magnetization between the
    trained model and a precomputed MCMC reference dataset.

    Args:
        df: DataFrame from compute_clock_thermodynamics()
        mcmc_ref: dict with keys "T", "E_mean", "M_mean" (MCMC reference data)
        q: Number of clock states
        output_dir: Directory to save the plot
        filename: Output filename

    Returns:
        str: Path to the saved figure
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    T_BKT = CLOCK_TC.get(q, 0.89)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def add_tbkt_line(ax):
        ax.axvline(
            T_BKT,
            color="r",
            linestyle="--",
            alpha=0.6,
            label=f"T_BKT = {T_BKT:.3f}",
        )

    # ============================================
    # Left: Energy comparison
    # ============================================
    ax = axes[0]
    ax.plot(df["T"], df["E_mean"], "o-", color="steelblue", markersize=4, label="Model")
    if "E_mean" in mcmc_ref:
        ax.plot(
            mcmc_ref["T"],
            mcmc_ref["E_mean"],
            "s--",
            color="darkorange",
            markersize=4,
            alpha=0.8,
            label="MCMC ref",
        )
    add_tbkt_line(ax)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Mean Energy <E>")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{q}-Clock: Energy — Model vs MCMC")

    # ============================================
    # Right: Magnetization comparison
    # ============================================
    ax = axes[1]
    ax.plot(df["T"], df["M_mean"], "o-", color="mediumseagreen", markersize=4, label="Model")
    if "M_mean" in mcmc_ref:
        ax.plot(
            mcmc_ref["T"],
            mcmc_ref["M_mean"],
            "s--",
            color="darkorange",
            markersize=4,
            alpha=0.8,
            label="MCMC ref",
        )
    ax.axhline(0, color="k", linestyle="-", alpha=0.2)
    add_tbkt_line(ax)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Vector Magnetization <M>")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{q}-Clock: Magnetization — Model vs MCMC")

    plt.tight_layout()

    output_path = f"{output_dir}/{filename}"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def select_tests(console, has_mcmc_ref=False) -> list:
    """
    Interactive multi-select for clock model analysis tests.

    Available tests:
      test1: Thermodynamic Quantities (around T_BKT)
      test2: Thermodynamic Quantities (Full Range)
      test3: Lattice Visualization (around T_BKT)
      test4: Lattice Visualization (Full Range)
      test5: MCMC Reference Comparison (only if ref data exists)

    Args:
        console: Rich Console instance
        has_mcmc_ref: Whether MCMC reference data is available

    Returns:
        list of selected test key strings
    """
    test_options = [
        ("test1", "Test 1: Thermodynamic Quantities (around T_BKT)"),
        ("test2", "Test 2: Thermodynamic Quantities (Full Range)"),
        ("test3", "Test 3: Lattice Visualization (around T_BKT)"),
        ("test4", "Test 4: Lattice Visualization (Full Range)"),
    ]

    if has_mcmc_ref:
        test_options.append(("test5", "Test 5: MCMC Reference Comparison"))

    console.print("\n[bold cyan]Select tests to run:[/bold cyan]")
    console.print("[dim](Use Space to select/deselect, Enter to confirm)[/dim]\n")

    display_options = [opt[1] for opt in test_options]

    selected_indices = beaupy.select_multiple(
        display_options,
        tick_character="✓",
        ticked_indices=list(range(len(display_options))),  # All selected by default
        minimal_count=1,
        return_indices=True,
    )

    if selected_indices is None:
        console.print("[yellow]No tests selected. Exiting.[/yellow]")
        return []

    selected_keys = [test_options[i][0] for i in selected_indices]

    console.print("\n[bold]Selected tests:[/bold]")
    for i in selected_indices:
        console.print(f"  [green]✓[/green] {test_options[i][1]}")

    return selected_keys


def main():
    console = Console()
    console.print("[bold green]Clock Model Analysis — Variational Thermodynamic Divergence[/bold green]")

    # ========================================
    # Interactive model selection
    # ========================================
    console.print("\nSelect a project to analyze:")
    project = select_project()

    console.print("Select a group to analyze:")
    group_name = select_group(project)

    console.print("Select a seed to analyze:")
    seed = select_seed(project, group_name)

    console.print("Select a device:")
    device = select_device()

    # ========================================
    # Load model and detect clock parameters
    # ========================================
    console.print(f"\n[bold]Loading model:[/bold] {project}/{group_name}/{seed}")
    model, config = load_model(project, group_name, seed)
    model = model.to(device)
    model.eval()

    # Use pure PyTorch fallback for mHC if available
    if hasattr(model, 'use_pytorch_mhc'):
        model.use_pytorch_mhc()

    # Detect clock model parameters from model
    q = getattr(model, 'category', None)
    if q is None:
        console.print("[red]Error: model.category not found. Cannot determine q.[/red]")
        return

    T_BKT = CLOCK_TC.get(q, 0.89)
    L = model.size[0]
    num_params = sum(p.numel() for p in model.parameters())

    console.print(f"[bold cyan]Clock model:[/bold cyan] q = {q}")
    console.print(f"Lattice size: {L}x{L}")
    console.print(f"Parameters: {num_params:,}")
    console.print(f"T_BKT = {T_BKT:.4f}")
    console.print(f"Central charge: c = {CLOCK_CENTRAL_CHARGE}")

    # ========================================
    # Create clock energy function
    # ========================================
    energy_fn = create_clock_energy_fn(L=L, q=q, d=2, device=device)
    console.print(f"[dim]Energy function created: {q}-state clock on {L}x{L} lattice[/dim]")

    # ========================================
    # Try loading MCMC reference data
    # ========================================
    ref_path = f"refs/clock{q}_L{L}.pt"
    mcmc_ref = None
    has_mcmc_ref = False

    if os.path.exists(ref_path):
        try:
            mcmc_ref = torch.load(ref_path, map_location="cpu", weights_only=True)
            # Convert tensors to numpy for plotting
            for key in mcmc_ref:
                if isinstance(mcmc_ref[key], torch.Tensor):
                    mcmc_ref[key] = mcmc_ref[key].numpy()
            has_mcmc_ref = True
            console.print(f"[green]✓[/green] Loaded MCMC reference from {ref_path}")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load MCMC reference from {ref_path}: {e}[/yellow]")
    else:
        console.print(f"[dim]No MCMC reference found at {ref_path} (skipping comparison)[/dim]")

    # ========================================
    # Get temperature range from config
    # ========================================
    config_dict = config.gen_config()
    net_config = config_dict.get("net_config", {})
    train_beta_min = net_config.get("beta_min", 1.0 / (1.5 * T_BKT))
    train_beta_max = net_config.get("beta_max", 1.0 / (0.5 * T_BKT))
    T_train_min = 1.0 / train_beta_max
    T_train_max = 1.0 / train_beta_min

    console.print(
        f"Training temperature range: T = [{T_train_min:.4f}, {T_train_max:.4f}]"
    )

    # Figures and results output directories
    figs_dir = f"figs/{group_name}"
    output_dir = f"runs/{project}/{group_name}"
    Path(figs_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Figures will be saved to: {figs_dir}[/dim]")
    console.print(f"[dim]Results will be saved to: {output_dir}[/dim]")

    # ========================================
    # Interactive test selection
    # ========================================
    selected_tests = select_tests(console, has_mcmc_ref=has_mcmc_ref)
    if not selected_tests:
        return

    batch_size = 1000

    # ========================================
    # Test 1: Thermodynamic Quantities (around T_BKT)
    # ========================================
    if "test1" in selected_tests:
        console.print(
            f"\n[bold cyan]Test 1: Thermodynamic Quantities (around T_BKT = {T_BKT:.4f})[/bold cyan]"
        )
        T_min_critical = 0.7 * T_BKT
        T_max_critical = 1.3 * T_BKT
        console.print(
            f"Temperature range: T = [{T_min_critical:.4f}, {T_max_critical:.4f}]"
        )
        console.print("Computing thermodynamic quantities...")

        df_critical = compute_clock_thermodynamics(
            model,
            energy_fn,
            q,
            L,
            device=device,
            num_temps=20,
            batch_size=batch_size,
            T_min=T_min_critical,
            T_max=T_max_critical,
            verbose=True,
        )

        # Display summary table
        console.print("\n" + "=" * 80)
        console.print("THERMODYNAMIC QUANTITIES (around T_BKT)")
        console.print("=" * 80)
        display_cols = ["T", "T/T_BKT", "E_mean", "Cv", "M_mean", "chi", "helicity", "vortex_density"]
        console.print(df_critical[display_cols].to_string(index=False))
        console.print("=" * 80)

        # Summary near T_BKT
        idx_tbkt = (df_critical["T"] - T_BKT).abs().idxmin()
        row_tbkt = df_critical.loc[idx_tbkt]
        console.print(f"\n[bold yellow]At T_BKT = {T_BKT:.4f}:[/bold yellow]")
        console.print(f"  T = {row_tbkt['T']:.4f}  (T/T_BKT = {row_tbkt['T/T_BKT']:.3f})")
        console.print(f"  Energy:          {row_tbkt['E_mean']:.6f}")
        console.print(f"  Heat capacity:   {row_tbkt['Cv']:.6f}")
        console.print(f"  Magnetization:   {row_tbkt['M_mean']:.6f}")
        console.print(f"  Susceptibility:  {row_tbkt['chi']:.6f}")
        console.print(f"  Helicity:        {row_tbkt['helicity']:.6f}  (jump = {2/math.pi*T_BKT:.4f})")
        console.print(f"  Vortex density:  {row_tbkt['vortex_density']:.6f}")

        # Peak Cv location
        idx_cv_peak = df_critical["Cv"].idxmax()
        row_cv = df_critical.loc[idx_cv_peak]
        console.print(f"\n[bold]Peak heat capacity:[/bold]")
        console.print(
            f"  T = {row_cv['T']:.4f}  (T/T_BKT = {row_cv['T/T_BKT']:.3f})  Cv = {row_cv['Cv']:.4f}"
        )

        # Generate plots
        console.print("\n[bold]Generating thermodynamic plots (critical range)...[/bold]")
        thermo_plot_path = plot_clock_thermodynamics(
            df_critical,
            q,
            output_dir=figs_dir,
            filename=f"clock_thermodynamics_critical_{seed}.png",
            title_suffix=" (around T_BKT)",
        )
        console.print(f"[green]✓[/green] Saved plot to {thermo_plot_path}")

        # Save CSV
        csv_path = f"{output_dir}/clock_thermo_critical_{seed}.csv"
        df_critical.to_csv(csv_path, index=False)
        console.print(f"[green]✓[/green] Saved results to {csv_path}")

    # ========================================
    # Test 2: Thermodynamic Quantities (Full Range)
    # ========================================
    if "test2" in selected_tests:
        console.print("\n[bold cyan]Test 2: Thermodynamic Quantities (Full Range)[/bold cyan]")
        T_min_full = 0.3 * T_BKT
        T_max_full = 2.0 * T_BKT
        console.print(
            f"Temperature range: T = [{T_min_full:.4f}, {T_max_full:.4f}]"
        )
        console.print("Computing thermodynamic quantities...")

        df_full = compute_clock_thermodynamics(
            model,
            energy_fn,
            q,
            L,
            device=device,
            num_temps=30,
            batch_size=batch_size,
            T_min=T_min_full,
            T_max=T_max_full,
            verbose=True,
        )

        # Display summary table
        console.print("\n" + "=" * 80)
        console.print("THERMODYNAMIC QUANTITIES (Full Range)")
        console.print("=" * 80)
        display_cols = ["T", "T/T_BKT", "E_mean", "Cv", "M_mean", "chi", "helicity", "vortex_density"]
        console.print(df_full[display_cols].to_string(index=False))
        console.print("=" * 80)

        # Summary near T_BKT
        idx_tbkt = (df_full["T"] - T_BKT).abs().idxmin()
        row_tbkt = df_full.loc[idx_tbkt]
        console.print(f"\n[bold yellow]At T_BKT = {T_BKT:.4f}:[/bold yellow]")
        console.print(f"  T = {row_tbkt['T']:.4f}  (T/T_BKT = {row_tbkt['T/T_BKT']:.3f})")
        console.print(f"  Helicity:        {row_tbkt['helicity']:.6f}  "
                      f"(universal jump = {2/math.pi*T_BKT:.4f})")
        console.print(f"  Vortex density:  {row_tbkt['vortex_density']:.6f}")

        # Helicity jump estimate
        low_T_mask = df_full["T/T_BKT"] < 0.9
        high_T_mask = df_full["T/T_BKT"] > 1.1
        if low_T_mask.any() and high_T_mask.any():
            helicity_low = df_full.loc[low_T_mask, "helicity"].iloc[-1]
            helicity_high = df_full.loc[high_T_mask, "helicity"].iloc[0]
            console.print(f"\n[bold]Helicity modulus jump estimate:[/bold]")
            console.print(f"  Below T_BKT (T/T_BKT ~ 0.9): Upsilon = {helicity_low:.4f}")
            console.print(f"  Above T_BKT (T/T_BKT ~ 1.1): Upsilon = {helicity_high:.4f}")
            console.print(f"  Jump = {helicity_low - helicity_high:.4f}")
            console.print(f"  Theory (2/pi * T_BKT) = {2/math.pi*T_BKT:.4f}")

        # Generate plots
        console.print("\n[bold]Generating thermodynamic plots (full range)...[/bold]")
        thermo_full_path = plot_clock_thermodynamics(
            df_full,
            q,
            output_dir=figs_dir,
            filename=f"clock_thermodynamics_full_{seed}.png",
            title_suffix=" (Full Range)",
        )
        console.print(f"[green]✓[/green] Saved plot to {thermo_full_path}")

        # Save CSV
        csv_path_full = f"{output_dir}/clock_thermo_full_{seed}.csv"
        df_full.to_csv(csv_path_full, index=False)
        console.print(f"[green]✓[/green] Saved results to {csv_path_full}")

    # ========================================
    # Test 3: Lattice Visualization (around T_BKT)
    # ========================================
    if "test3" in selected_tests:
        console.print(
            f"\n[bold cyan]Test 3: Lattice Visualization (around T_BKT = {T_BKT:.4f})[/bold cyan]"
        )

        temperatures_critical = [
            0.7 * T_BKT,
            0.85 * T_BKT,
            T_BKT,
            1.15 * T_BKT,
            1.3 * T_BKT,
        ]

        console.print(f"Generating clock samples at {len(temperatures_critical)} temperatures:")
        for T in temperatures_critical:
            console.print(f"  T = {T:.4f}  (T/T_BKT = {T/T_BKT:.2f})")

        console.print("\n[bold]Generating lattice visualizations...[/bold]")
        vis_path_critical = visualize_clock_samples(
            model,
            q,
            L,
            temperatures_critical,
            device=device,
            num_samples=4,
            output_dir=figs_dir,
            filename=f"clock_samples_critical_{seed}.png",
        )
        console.print(f"[green]✓[/green] Saved lattice visualization to {vis_path_critical}")

    # ========================================
    # Test 4: Lattice Visualization (Full Range)
    # ========================================
    if "test4" in selected_tests:
        console.print("\n[bold cyan]Test 4: Lattice Visualization (Full Range)[/bold cyan]")

        temperatures_full = np.linspace(0.3 * T_BKT, 2.0 * T_BKT, 6).tolist()

        console.print(f"Generating clock samples at {len(temperatures_full)} temperatures (full range):")
        for T in temperatures_full:
            console.print(f"  T = {T:.4f}  (T/T_BKT = {T/T_BKT:.2f})")

        console.print("\n[bold]Generating lattice visualizations (full range)...[/bold]")
        vis_path_full = visualize_clock_samples(
            model,
            q,
            L,
            temperatures_full,
            device=device,
            num_samples=4,
            output_dir=figs_dir,
            filename=f"clock_samples_full_{seed}.png",
        )
        console.print(f"[green]✓[/green] Saved lattice visualization to {vis_path_full}")

    # ========================================
    # Test 5: MCMC Reference Comparison
    # ========================================
    if "test5" in selected_tests and has_mcmc_ref and mcmc_ref is not None:
        console.print("\n[bold cyan]Test 5: MCMC Reference Comparison[/bold cyan]")

        # Use matching temperature range from MCMC reference if available
        if "T" in mcmc_ref:
            T_mcmc_min = float(mcmc_ref["T"].min())
            T_mcmc_max = float(mcmc_ref["T"].max())
            console.print(
                f"MCMC reference range: T = [{T_mcmc_min:.4f}, {T_mcmc_max:.4f}]"
            )
            T_min_comp = T_mcmc_min
            T_max_comp = T_mcmc_max
        else:
            T_min_comp = 0.5 * T_BKT
            T_max_comp = 1.5 * T_BKT

        console.print("Computing model thermodynamics for comparison...")
        df_comp = compute_clock_thermodynamics(
            model,
            energy_fn,
            q,
            L,
            device=device,
            num_temps=20,
            batch_size=batch_size,
            T_min=T_min_comp,
            T_max=T_max_comp,
            verbose=True,
        )

        console.print("\n[bold]Generating MCMC comparison plots...[/bold]")
        mcmc_plot_path = plot_mcmc_comparison(
            df_comp,
            mcmc_ref,
            q,
            output_dir=figs_dir,
            filename=f"clock_mcmc_comparison_{seed}.png",
        )
        console.print(f"[green]✓[/green] Saved comparison plot to {mcmc_plot_path}")

        # Quick quantitative comparison
        if "T" in mcmc_ref and "E_mean" in mcmc_ref:
            # Interpolate model predictions at MCMC reference temperatures
            model_E_interp = np.interp(
                mcmc_ref["T"], df_comp["T"].values, df_comp["E_mean"].values
            )
            E_mae = np.abs(model_E_interp - mcmc_ref["E_mean"]).mean()
            console.print(f"\n[bold]Energy comparison:[/bold]")
            console.print(f"  Mean absolute error vs MCMC: {E_mae:.6f}")

        if "T" in mcmc_ref and "M_mean" in mcmc_ref:
            model_M_interp = np.interp(
                mcmc_ref["T"], df_comp["T"].values, df_comp["M_mean"].values
            )
            M_mae = np.abs(model_M_interp - mcmc_ref["M_mean"]).mean()
            console.print(f"\n[bold]Magnetization comparison:[/bold]")
            console.print(f"  Mean absolute error vs MCMC: {M_mae:.6f}")

        # Save comparison CSV
        csv_comp_path = f"{output_dir}/clock_mcmc_comparison_{seed}.csv"
        df_comp.to_csv(csv_comp_path, index=False)
        console.print(f"[green]✓[/green] Saved comparison results to {csv_comp_path}")

    console.print(
        "\n[bold green]Clock model analysis complete! All results saved.[/bold green]"
    )


if __name__ == "__main__":
    main()
