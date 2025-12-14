import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import beaupy
from rich.console import Console
from pathlib import Path

from util import (
    select_project,
    select_group,
    select_seed,
    select_device,
    load_model,
)
from main import create_ising_energy_fn
from vatd_exact_partition import logZ as exact_logZ, CRITICAL_TEMPERATURE


def test_model_ising(
    model,
    energy_fn,
    L,
    device="cpu",
    num_temps=20,
    batch_size=500,
    T_min=None,
    T_max=None,
):
    """
    Evaluate trained model at multiple temperatures.

    Args:
        model: Trained DiscretePixelCNN model
        energy_fn: Ising energy function
        L: Lattice size
        device: Device to run on
        num_temps: Number of temperature points to test
        batch_size: Samples per temperature
        T_min: Minimum temperature (default: 0.7*Tc)
        T_max: Maximum temperature (default: 1.3*Tc)

    Returns:
        pandas.DataFrame with columns:
            - T: Temperature
            - beta: Inverse temperature (1/T)
            - T/Tc: Reduced temperature
            - model_loss: Model's computed loss (mean of log_prob + beta*energy)
            - exact_loss: Exact loss (-exact_logz)
            - loss_error: model_loss - exact_loss
            - model_logz: Model's log partition function estimate (-model_loss)
            - exact_logz: Exact log partition function
            - logz_error: model_logz - exact_logz
            - abs_error: Absolute log Z error
    """
    # Temperature range
    if T_min is None:
        T_min = 0.7 * CRITICAL_TEMPERATURE
    if T_max is None:
        T_max = 1.3 * CRITICAL_TEMPERATURE

    T_values = np.linspace(T_min, T_max, num_temps)

    results = []

    model.eval()
    with torch.no_grad():
        for T in T_values:
            beta = 1.0 / T
            # Create temperature tensor with same batch size as samples
            T_tensor = torch.full((batch_size,), T, device=device)

            # Sample from model
            samples = model.sample(batch_size=batch_size, T=T_tensor)
            log_prob = model.log_prob(samples, T=T_tensor)
            energy = energy_fn(samples)

            # Compute loss
            beta_tensor = torch.tensor([beta], device=device).unsqueeze(-1)
            loss_raw = log_prob + beta_tensor * energy
            model_loss = loss_raw.mean().item()

            # Compute exact values
            exact_logz = exact_logZ(n=L, j=1.0, beta=torch.tensor(beta)).item()
            exact_loss = -exact_logz

            # Compute model log Z estimate
            model_logz = -model_loss

            # Error metrics
            logz_error = model_logz - exact_logz
            loss_error = model_loss - exact_loss

            results.append(
                {
                    "T": T,
                    "beta": beta,
                    "T/Tc": T / CRITICAL_TEMPERATURE,
                    "model_loss": model_loss,
                    "exact_loss": exact_loss,
                    "loss_error": loss_error,
                    "model_logz": model_logz,
                    "exact_logz": exact_logz,
                    "logz_error": logz_error,
                    "abs_error": abs(logz_error),
                }
            )

    return pd.DataFrame(results)


def compute_thermodynamic_quantities(
    model,
    energy_fn,
    L,
    device="cpu",
    num_temps=20,
    batch_size=1000,
    T_min=None,
    T_max=None,
):
    """
    Compute thermodynamic quantities using sampling and automatic differentiation.

    Two methods are used:
    1. Sampling (Monte Carlo): Uses samples from the model
       - Mean energy: <E> = mean(energy(samples))
       - Heat capacity: C_v = beta^2 * (<E^2> - <E>^2)
       - Magnetization: M = <|m|> where m = (1/N) * sum(s_i) per sample
       - Susceptibility: chi = beta * N * (<m^2> - <|m|>^2)

    2. Automatic Differentiation:
       - From exact log Z: d(log Z)/d(beta) and d^2(log Z)/d(beta)^2
       - From model log_prob: differentiating through log_prob(sample, T) w.r.t. beta

    Args:
        model: Trained DiscretePixelCNN model
        energy_fn: Ising energy function
        L: Lattice size
        device: Device to run on
        num_temps: Number of temperature points
        batch_size: Samples per temperature (larger for better statistics)
        T_min: Minimum temperature (default: 0.7*Tc)
        T_max: Maximum temperature (default: 1.3*Tc)

    Returns:
        pandas.DataFrame with columns:
            - T, beta, T/Tc: Temperature parameters
            - E_sampling, Cv_sampling, M_sampling, chi_sampling: Sampling-based
            - E_exact_AD, Cv_exact_AD: Exact log Z via AD
            - E_model_AD, Cv_model_AD: Model log_prob via AD
    """
    # Temperature range
    if T_min is None:
        T_min = 0.7 * CRITICAL_TEMPERATURE
    if T_max is None:
        T_max = 1.3 * CRITICAL_TEMPERATURE

    T_values = np.linspace(T_min, T_max, num_temps)
    N = L * L  # Total number of spins

    results = []

    model.eval()

    for T in T_values:
        beta = 1.0 / T
        T_tensor = torch.full((batch_size,), T, device=device)

        # ============================================
        # METHOD 1: Sampling-based quantities
        # ============================================
        with torch.no_grad():
            samples = model.sample(batch_size=batch_size, T=T_tensor)
            energies = energy_fn(samples).squeeze()  # (batch_size,)

            # Mean energy
            E_mean = energies.mean().item()
            E2_mean = (energies**2).mean().item()

            # Heat capacity via fluctuation-dissipation: C_v = beta^2 * Var(E)
            Cv_sampling = (beta**2) * (E2_mean - E_mean**2)

            # Magnetization per spin: m = (1/N) * sum(s_i)
            magnetization_per_spin = samples.sum(dim=(1, 2, 3)) / N  # (batch_size,)
            abs_m = magnetization_per_spin.abs()
            M_mean = abs_m.mean().item()
            M2_mean = (magnetization_per_spin**2).mean().item()

            # Susceptibility: chi = beta * N * (<m^2> - <|m|>^2)
            chi_sampling = beta * N * (M2_mean - M_mean**2)

        # ============================================
        # METHOD 2a: AD through exact log Z (Onsager)
        # ============================================
        # E_exact = -d(log Z)/d(beta)
        # Cv_exact = beta^2 * d^2(log Z)/d(beta)^2

        beta_tensor = torch.tensor(beta, dtype=torch.float64, requires_grad=True)
        log_Z = exact_logZ(n=L, j=1.0, beta=beta_tensor)

        # First derivative
        grad_logZ = torch.autograd.grad(log_Z, beta_tensor, create_graph=True)[0]
        E_exact_AD = -grad_logZ.item()

        # Second derivative
        grad2_logZ = torch.autograd.grad(grad_logZ, beta_tensor)[0]
        Cv_exact_AD = (beta**2) * grad2_logZ.item()

        # ============================================
        # METHOD 2b: AD through model log_prob
        # ============================================
        # Use fixed samples (detached) but differentiate log_prob w.r.t. beta
        # loss(beta) = <log q(x|T) + beta * E(x)> approx -log Z
        # d(loss)/d(beta) includes both explicit beta term and implicit T=1/beta

        with torch.no_grad():
            samples_for_ad = model.sample(batch_size=batch_size, T=T_tensor)
            energies_for_ad = energy_fn(samples_for_ad).squeeze()

        # Compute log_prob with gradient tracking on beta
        beta_model = torch.tensor(
            beta, dtype=torch.float32, device=device, requires_grad=True
        )
        T_model = 1.0 / beta_model
        T_expanded = T_model.expand(batch_size)

        # samples_for_ad is fixed, but T_expanded carries gradient
        log_prob_ad = model.log_prob(samples_for_ad, T=T_expanded)
        loss_model = (log_prob_ad + beta_model * energies_for_ad.unsqueeze(-1)).mean()

        grad_loss = torch.autograd.grad(loss_model, beta_model, create_graph=True)[0]
        E_model_AD = grad_loss.item()

        # Second derivative for heat capacity
        # Note: loss ≈ -log Z, so d²(log Z)/dβ² ≈ -d²(loss)/dβ²
        # Therefore Cv = β² × d²(log Z)/dβ² = -β² × d²(loss)/dβ²
        grad2_loss = torch.autograd.grad(grad_loss, beta_model)[0]
        Cv_model_AD = -(beta**2) * grad2_loss.item()

        # Collect results
        results.append(
            {
                "T": T,
                "beta": beta,
                "T/Tc": T / CRITICAL_TEMPERATURE,
                # Sampling
                "E_sampling": E_mean,
                "Cv_sampling": Cv_sampling,
                "M_sampling": M_mean,
                "chi_sampling": chi_sampling,
                # Exact AD
                "E_exact_AD": E_exact_AD,
                "Cv_exact_AD": Cv_exact_AD,
                # Model AD
                "E_model_AD": E_model_AD,
                "Cv_model_AD": Cv_model_AD,
            }
        )

    return pd.DataFrame(results)


def visualize_lattice_samples(
    model,
    L,
    temperatures,
    device="cpu",
    num_samples=4,
    output_dir="figs",
    filename="lattice_samples.png",
):
    """
    Visualize sample lattices from the model at different temperatures.

    Args:
        model: Trained DiscretePixelCNN model
        L: Lattice size
        temperatures: List of temperatures to sample at
        device: Device to run on
        num_samples: Number of samples to show per temperature
        output_dir: Directory to save plots
        filename: Output filename
    """
    Path(output_dir).mkdir(exist_ok=True)

    model.eval()
    num_temps = len(temperatures)

    fig, axes = plt.subplots(
        num_temps, num_samples, figsize=(3 * num_samples, 3 * num_temps)
    )

    # Handle single row case
    if num_temps == 1:
        axes = axes.reshape(1, -1)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    with torch.no_grad():
        for i, T in enumerate(temperatures):
            T_tensor = torch.full((num_samples,), T, device=device)

            # Sample from model
            samples = model.sample(batch_size=num_samples, T=T_tensor)

            # Convert to numpy and reshape
            samples_np = samples.cpu().numpy()

            for j in range(num_samples):
                ax = axes[i, j]
                lattice = samples_np[j].reshape(L, L)

                # Plot lattice (-1: white, +1: black)
                im = ax.imshow(
                    lattice, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest"
                )

                if j == 0:
                    ax.set_ylabel(
                        f"T={T:.3f}\n(T/Tc={T/CRITICAL_TEMPERATURE:.2f})", fontsize=10
                    )
                else:
                    ax.set_ylabel("")

                ax.set_xticks([])
                ax.set_yticks([])

                if i == 0:
                    ax.set_title(f"Sample {j+1}")

    plt.tight_layout()
    output_path = f"{output_dir}/{filename}"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_model_vs_exact(
    df, output_dir="figs", title_suffix="", filename="model_test_analysis.png"
):
    """
    Generate comprehensive plots comparing model vs exact values.

    Creates 6 subplots (3 rows × 2 columns):
    - Row 1: Log Z comparison
    - Row 2: Loss comparison
    - Row 3: Error metrics

    Args:
        df: DataFrame with test results
        output_dir: Directory to save plots (default: 'figs')
        title_suffix: Suffix to add to plot titles (e.g., '(Critical Range)')
        filename: Output filename (default: 'model_test_analysis.png')
    """
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # Row 1: Log Z Comparison
    # Plot 1: Model log Z vs Exact log Z
    ax = axes[0, 0]
    ax.plot(df["T"], df["model_logz"], "o-", label="Model log Z", markersize=4)
    ax.plot(df["T"], df["exact_logz"], "s-", label="Exact log Z", markersize=4)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("log Z")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Log Partition Function: Model vs Exact{title_suffix}")

    # Plot 2: Log Z Error vs Temperature
    ax = axes[0, 1]
    ax.plot(df["T"], df["logz_error"], "o-", markersize=4)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Error (Model - Exact)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Log Z Error vs Temperature{title_suffix}")

    # Row 2: Loss Comparison
    # Plot 3: Model Loss vs Exact Loss
    ax = axes[1, 0]
    ax.plot(df["T"], df["model_loss"], "o-", label="Model Loss", markersize=4)
    ax.plot(df["T"], df["exact_loss"], "s-", label="Exact Loss", markersize=4)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Loss: Model vs Exact{title_suffix}")

    # Plot 4: Loss Error vs Temperature
    ax = axes[1, 1]
    ax.plot(df["T"], df["loss_error"], "o-", markersize=4)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Error (Model - Exact)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Loss Error vs Temperature{title_suffix}")

    # Row 3: Additional Metrics
    # Plot 5: Absolute Error (log scale)
    ax = axes[2, 0]
    ax.semilogy(df["T"], df["abs_error"], "o-", markersize=4)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("|Log Z Error| (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Absolute Error{title_suffix}")

    # Plot 6: Error vs Reduced Temperature (T/Tc)
    ax = axes[2, 1]
    ax.plot(df["T/Tc"], df["logz_error"], "o-", markersize=4)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(1.0, color="r", linestyle="--", alpha=0.5, label="T/Tc = 1")
    ax.set_xlabel("T / Tc")
    ax.set_ylabel("Log Z Error (Model - Exact)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Error vs Reduced Temperature{title_suffix}")

    plt.tight_layout()
    output_path = f"{output_dir}/{filename}"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_thermodynamic_quantities(
    df,
    output_dir="figs",
    title_suffix="",
    filename="thermodynamic_analysis.png",
):
    """
    Generate comprehensive plots for thermodynamic quantities.

    Creates 2x2 subplot figure:
    - (0,0): Mean Energy - Sampling vs Exact AD vs Model AD
    - (0,1): Heat Capacity - Sampling vs Exact AD vs Model AD
    - (1,0): Magnetization - Sampling only
    - (1,1): Susceptibility - Sampling only

    Args:
        df: DataFrame from compute_thermodynamic_quantities()
        output_dir: Directory to save plots
        title_suffix: Suffix for plot titles
        filename: Output filename

    Returns:
        str: Path to saved plot
    """
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ============================================
    # Plot (0,0): Mean Energy Comparison
    # ============================================
    ax = axes[0, 0]
    ax.plot(df["T"], df["E_sampling"], "o-", label="Sampling", markersize=4)
    ax.plot(df["T"], df["E_exact_AD"], "s--", label="Exact (AD)", markersize=4)
    ax.plot(df["T"], df["E_model_AD"], "^:", label="Model (AD)", markersize=4)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Mean Energy ⟨E⟩")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Mean Energy{title_suffix}")

    # ============================================
    # Plot (0,1): Heat Capacity Comparison
    # ============================================
    ax = axes[0, 1]
    ax.plot(df["T"], df["Cv_sampling"], "o-", label="Sampling", markersize=4)
    ax.plot(df["T"], df["Cv_exact_AD"], "s--", label="Exact (AD)", markersize=4)
    ax.plot(df["T"], df["Cv_model_AD"], "^:", label="Model (AD)", markersize=4)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Heat Capacity Cv")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Heat Capacity{title_suffix}")

    # ============================================
    # Plot (1,0): Magnetization (Sampling only)
    # ============================================
    ax = axes[1, 0]
    ax.plot(df["T"], df["M_sampling"], "o-", label="Sampling ⟨|m|⟩", markersize=4)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.axhline(0, color="k", linestyle="-", alpha=0.2)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Magnetization ⟨|m|⟩")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Magnetization{title_suffix}")

    # ============================================
    # Plot (1,1): Susceptibility (Sampling only)
    # ============================================
    ax = axes[1, 1]
    ax.plot(df["T"], df["chi_sampling"], "o-", label="Sampling", markersize=4)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Susceptibility χ")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Susceptibility{title_suffix}")

    plt.tight_layout()
    output_path = f"{output_dir}/{filename}"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_thermodynamic_errors(
    df,
    output_dir="figs",
    title_suffix="",
    filename="thermodynamic_errors.png",
):
    """
    Generate error comparison plots for thermodynamic quantities.

    Creates 2x2 subplot figure comparing:
    - Energy: Sampling vs Exact AD, Model AD vs Exact AD
    - Heat Capacity: Sampling vs Exact AD, Model AD vs Exact AD

    Args:
        df: DataFrame from compute_thermodynamic_quantities()
        output_dir: Directory to save plots
        title_suffix: Suffix for plot titles
        filename: Output filename

    Returns:
        str: Path to saved plot
    """
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute errors
    E_sampling_error = df["E_sampling"] - df["E_exact_AD"]
    E_model_error = df["E_model_AD"] - df["E_exact_AD"]
    Cv_sampling_error = df["Cv_sampling"] - df["Cv_exact_AD"]
    Cv_model_error = df["Cv_model_AD"] - df["Cv_exact_AD"]

    # ============================================
    # Plot (0,0): Energy - Sampling Error
    # ============================================
    ax = axes[0, 0]
    ax.plot(df["T"], E_sampling_error, "o-", markersize=4)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Error (Sampling - Exact)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Energy: Sampling Error{title_suffix}")

    # ============================================
    # Plot (0,1): Energy - Model AD Error
    # ============================================
    ax = axes[0, 1]
    ax.plot(df["T"], E_model_error, "o-", markersize=4, color="orange")
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Error (Model AD - Exact)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Energy: Model AD Error{title_suffix}")

    # ============================================
    # Plot (1,0): Heat Capacity - Sampling Error
    # ============================================
    ax = axes[1, 0]
    ax.plot(df["T"], Cv_sampling_error, "o-", markersize=4)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Error (Sampling - Exact)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Heat Capacity: Sampling Error{title_suffix}")

    # ============================================
    # Plot (1,1): Heat Capacity - Model AD Error
    # ============================================
    ax = axes[1, 1]
    ax.plot(df["T"], Cv_model_error, "o-", markersize=4, color="orange")
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(
        CRITICAL_TEMPERATURE,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Tc = {CRITICAL_TEMPERATURE:.3f}",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Error (Model AD - Exact)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Heat Capacity: Model AD Error{title_suffix}")

    plt.tight_layout()
    output_path = f"{output_dir}/{filename}"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def main():
    console = Console()
    console.print("[bold green]Testing Trained Model on Ising System[/bold green]")

    # Interactive selection
    console.print("\nSelect a project to analyze:")
    project = select_project()

    console.print("Select a group to analyze:")
    group_name = select_group(project)

    console.print("Select a seed to analyze:")
    seed = select_seed(project, group_name)

    console.print("Select a device:")
    device = select_device()

    # Load model
    console.print(f"\n[bold]Loading model:[/bold] {project}/{group_name}/{seed}")
    model, config = load_model(project, group_name, seed)
    model = model.to(device)
    model.eval()

    # Get lattice size from model
    L = model.size[0]
    console.print(f"Lattice size: {L}×{L}")

    # Create energy function
    energy_fn = create_ising_energy_fn(L=L, d=2, device=device)
    console.print(f"Critical temperature: {CRITICAL_TEMPERATURE:.4f}")

    # Get validation temperature range
    # Validation uses fixed range (0.1-2.0) for extrapolation testing
    # Training range is in net_config, but validation is always wider
    beta_min = 0.1
    beta_max = 2.0
    T_val_max = 1.0 / beta_min  # Higher temperature
    T_val_min = 1.0 / beta_max  # Lower temperature
    console.print(
        f"Validation range: T = [{T_val_min:.4f}, {T_val_max:.4f}] (β = [{beta_min:.4f}, {beta_max:.4f}])"
    )

    # Get training range from config for reference
    config_dict = config.gen_config()
    net_config = config_dict.get("net_config", {})
    train_beta_min = net_config.get("beta_min", 0.2)
    train_beta_max = net_config.get("beta_max", 1.0)
    T_train_max = 1.0 / train_beta_min
    T_train_min = 1.0 / train_beta_max
    console.print(
        f"Training range: T = [{T_train_min:.4f}, {T_train_max:.4f}] (β = [{train_beta_min:.4f}, {train_beta_max:.4f}])"
    )

    # ========================================
    # Test 1: Critical Temperature Range
    # ========================================
    console.print(
        "\n[bold cyan]Test 1: Critical Temperature Range (0.7*Tc to 1.3*Tc)[/bold cyan]"
    )
    df_critical = test_model_ising(
        model, energy_fn, L, device=device, num_temps=20, batch_size=500
    )

    # Display results for critical range
    console.print("\n" + "=" * 70)
    console.print("CRITICAL RANGE RESULTS")
    console.print("=" * 70)
    console.print(df_critical.to_string(index=False))
    console.print("=" * 70)

    # Summary statistics for critical range
    idx_min_error = df_critical["abs_error"].idxmin()
    idx_max_error = df_critical["abs_error"].idxmax()

    console.print(f"\n[bold green]Best accuracy:[/bold green]")
    console.print(
        f"  T = {df_critical.loc[idx_min_error, 'T']:.4f} (T/Tc = {df_critical.loc[idx_min_error, 'T/Tc']:.3f})"
    )
    console.print(f"  |Error| = {df_critical.loc[idx_min_error, 'abs_error']:.6f}")

    console.print(f"\n[bold red]Worst accuracy:[/bold red]")
    console.print(
        f"  T = {df_critical.loc[idx_max_error, 'T']:.4f} (T/Tc = {df_critical.loc[idx_max_error, 'T/Tc']:.3f})"
    )
    console.print(f"  |Error| = {df_critical.loc[idx_max_error, 'abs_error']:.6f}")

    # Error at critical temperature
    idx_critical_temp = (df_critical["T"] - CRITICAL_TEMPERATURE).abs().idxmin()
    console.print(
        f"\n[bold yellow]At critical temperature Tc = {CRITICAL_TEMPERATURE:.4f}:[/bold yellow]"
    )
    console.print(f"  T = {df_critical.loc[idx_critical_temp, 'T']:.4f}")
    console.print(
        f"  Log Z Error = {df_critical.loc[idx_critical_temp, 'logz_error']:.6f}"
    )
    console.print(f"  |Error| = {df_critical.loc[idx_critical_temp, 'abs_error']:.6f}")

    # Generate plots for critical range
    console.print("\n[bold]Generating plots for critical range...[/bold]")
    plot_path_critical = plot_model_vs_exact(
        df_critical,
        title_suffix=" (Critical Range)",
        filename="model_test_critical.png",
    )
    console.print(f"[green]✓[/green] Saved plot to {plot_path_critical}")

    # Save results for critical range
    output_dir = f"runs/{project}/{group_name}"
    output_file_critical = f"{output_dir}/test_results_critical_{seed}.csv"
    df_critical.to_csv(output_file_critical, index=False)
    console.print(f"[green]✓[/green] Saved results to {output_file_critical}")

    # ========================================
    # Test 2: Validation Range
    # ========================================
    console.print("\n[bold cyan]Test 2: Validation Range[/bold cyan]")
    df_validation = test_model_ising(
        model,
        energy_fn,
        L,
        device=device,
        num_temps=20,
        batch_size=500,
        T_min=T_val_min,
        T_max=T_val_max,
    )

    # Display results for validation range
    console.print("\n" + "=" * 70)
    console.print("VALIDATION RANGE RESULTS")
    console.print("=" * 70)
    console.print(df_validation.to_string(index=False))
    console.print("=" * 70)

    # Summary statistics for validation range
    idx_min_error_val = df_validation["abs_error"].idxmin()
    idx_max_error_val = df_validation["abs_error"].idxmax()

    console.print(f"\n[bold green]Best accuracy:[/bold green]")
    console.print(
        f"  T = {df_validation.loc[idx_min_error_val, 'T']:.4f} (T/Tc = {df_validation.loc[idx_min_error_val, 'T/Tc']:.3f})"
    )
    console.print(
        f"  |Error| = {df_validation.loc[idx_min_error_val, 'abs_error']:.6f}"
    )

    console.print(f"\n[bold red]Worst accuracy:[/bold red]")
    console.print(
        f"  T = {df_validation.loc[idx_max_error_val, 'T']:.4f} (T/Tc = {df_validation.loc[idx_max_error_val, 'T/Tc']:.3f})"
    )
    console.print(
        f"  |Error| = {df_validation.loc[idx_max_error_val, 'abs_error']:.6f}"
    )

    # Error at critical temperature (if in range)
    if T_val_min <= CRITICAL_TEMPERATURE <= T_val_max:
        idx_critical_temp_val = (
            (df_validation["T"] - CRITICAL_TEMPERATURE).abs().idxmin()
        )
        console.print(
            f"\n[bold yellow]At critical temperature Tc = {CRITICAL_TEMPERATURE:.4f}:[/bold yellow]"
        )
        console.print(f"  T = {df_validation.loc[idx_critical_temp_val, 'T']:.4f}")
        console.print(
            f"  Log Z Error = {df_validation.loc[idx_critical_temp_val, 'logz_error']:.6f}"
        )
        console.print(
            f"  |Error| = {df_validation.loc[idx_critical_temp_val, 'abs_error']:.6f}"
        )

    # Generate plots for validation range
    console.print("\n[bold]Generating plots for validation range...[/bold]")
    plot_path_validation = plot_model_vs_exact(
        df_validation,
        title_suffix=" (Validation Range)",
        filename="model_test_validation.png",
    )
    console.print(f"[green]✓[/green] Saved plot to {plot_path_validation}")

    # Save results for validation range
    output_file_validation = f"{output_dir}/test_results_validation_{seed}.csv"
    df_validation.to_csv(output_file_validation, index=False)
    console.print(f"[green]✓[/green] Saved results to {output_file_validation}")

    # ========================================
    # Test 3: Visualize Sample Lattices (Validation Range)
    # ========================================
    console.print(
        "\n[bold cyan]Test 3: Visualizing Sample Lattices (Validation Range)[/bold cyan]"
    )

    # Select temperatures across validation range
    num_temp_samples = 6
    temperatures_val = np.linspace(T_val_min, T_val_max, num_temp_samples)

    console.print(
        f"Generating lattice samples at {num_temp_samples} temperatures across validation range:"
    )
    for T in temperatures_val:
        console.print(f"  T = {T:.4f} (T/Tc = {T/CRITICAL_TEMPERATURE:.2f})")

    console.print("\n[bold]Generating lattice visualizations...[/bold]")
    lattice_plot_path_val = visualize_lattice_samples(
        model,
        L,
        temperatures_val,
        device=device,
        num_samples=4,
        output_dir="figs",
        filename=f"lattice_samples_validation_{seed}.png",
    )
    console.print(
        f"[green]✓[/green] Saved validation range lattice samples to {lattice_plot_path_val}"
    )

    # ========================================
    # Test 4: Visualize Sample Lattices (Critical Range)
    # ========================================
    console.print(
        "\n[bold cyan]Test 4: Visualizing Sample Lattices (Critical Range)[/bold cyan]"
    )

    # Select temperatures near critical temperature
    T_low = 0.8 * CRITICAL_TEMPERATURE
    T_critical = CRITICAL_TEMPERATURE
    T_high = 1.2 * CRITICAL_TEMPERATURE
    temperatures_critical = [T_low, T_critical, T_high]

    console.print(f"Generating lattice samples at temperatures near Tc:")
    console.print(
        f"  Low:      T = {T_low:.4f} (T/Tc = {T_low/CRITICAL_TEMPERATURE:.2f})"
    )
    console.print(
        f"  Critical: T = {T_critical:.4f} (T/Tc = {T_critical/CRITICAL_TEMPERATURE:.2f})"
    )
    console.print(
        f"  High:     T = {T_high:.4f} (T/Tc = {T_high/CRITICAL_TEMPERATURE:.2f})"
    )

    console.print("\n[bold]Generating lattice visualizations...[/bold]")
    lattice_plot_path_critical = visualize_lattice_samples(
        model,
        L,
        temperatures_critical,
        device=device,
        num_samples=4,
        output_dir="figs",
        filename=f"lattice_samples_critical_{seed}.png",
    )
    console.print(
        f"[green]✓[/green] Saved critical range lattice samples to {lattice_plot_path_critical}"
    )

    # ========================================
    # Test 5: Thermodynamic Quantities Analysis
    # ========================================
    console.print(
        "\n[bold cyan]Test 5: Thermodynamic Quantities Analysis[/bold cyan]"
    )
    console.print(
        "Computing thermodynamic quantities (sampling + automatic differentiation)..."
    )

    df_thermo = compute_thermodynamic_quantities(
        model,
        energy_fn,
        L,
        device=device,
        num_temps=20,
        batch_size=1000,
        T_min=0.7 * CRITICAL_TEMPERATURE,
        T_max=1.3 * CRITICAL_TEMPERATURE,
    )

    # Display results
    console.print("\n" + "=" * 80)
    console.print("THERMODYNAMIC QUANTITIES")
    console.print("=" * 80)
    # Show selected columns for readability
    display_cols = [
        "T",
        "T/Tc",
        "E_sampling",
        "E_exact_AD",
        "Cv_sampling",
        "Cv_exact_AD",
        "M_sampling",
        "chi_sampling",
    ]
    console.print(df_thermo[display_cols].to_string(index=False))
    console.print("=" * 80)

    # Generate plots
    console.print("\n[bold]Generating thermodynamic quantity plots...[/bold]")
    thermo_plot_path = plot_thermodynamic_quantities(
        df_thermo,
        title_suffix=" (Critical Range)",
        filename=f"thermodynamic_analysis_{seed}.png",
    )
    console.print(f"[green]✓[/green] Saved plot to {thermo_plot_path}")

    thermo_error_path = plot_thermodynamic_errors(
        df_thermo,
        title_suffix=" (Critical Range)",
        filename=f"thermodynamic_errors_{seed}.png",
    )
    console.print(f"[green]✓[/green] Saved error plot to {thermo_error_path}")

    # Save results
    thermo_output_file = f"{output_dir}/thermodynamic_results_{seed}.csv"
    df_thermo.to_csv(thermo_output_file, index=False)
    console.print(f"[green]✓[/green] Saved results to {thermo_output_file}")

    # Summary statistics at critical temperature
    idx_tc = (df_thermo["T"] - CRITICAL_TEMPERATURE).abs().idxmin()
    row_tc = df_thermo.loc[idx_tc]
    console.print(f"\n[bold yellow]Summary at Critical Temperature:[/bold yellow]")
    console.print(f"  T = {row_tc['T']:.4f} (T/Tc = {row_tc['T/Tc']:.3f})")
    console.print(
        f"  Energy: Sampling={row_tc['E_sampling']:.4f}, "
        f"Exact AD={row_tc['E_exact_AD']:.4f}, "
        f"Model AD={row_tc['E_model_AD']:.4f}"
    )
    console.print(
        f"  Heat Capacity: Sampling={row_tc['Cv_sampling']:.4f}, "
        f"Exact AD={row_tc['Cv_exact_AD']:.4f}, "
        f"Model AD={row_tc['Cv_model_AD']:.4f}"
    )
    console.print(f"  Magnetization: {row_tc['M_sampling']:.4f}")
    console.print(f"  Susceptibility: {row_tc['chi_sampling']:.4f}")

    console.print(
        "\n[bold green]Analysis complete! All results saved to output directory.[/bold green]"
    )


if __name__ == "__main__":
    main()
