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


def test_model_ising(model, energy_fn, L, device='cpu', num_temps=20, batch_size=500):
    """
    Evaluate trained model at multiple temperatures.

    Args:
        model: Trained DiscretePixelCNN model
        energy_fn: Ising energy function
        L: Lattice size
        device: Device to run on
        num_temps: Number of temperature points to test
        batch_size: Samples per temperature

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
    # Temperature range: 0.7*Tc to 1.3*Tc
    T_min = 0.7 * CRITICAL_TEMPERATURE
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

            results.append({
                'T': T,
                'beta': beta,
                'T/Tc': T / CRITICAL_TEMPERATURE,
                'model_loss': model_loss,
                'exact_loss': exact_loss,
                'loss_error': loss_error,
                'model_logz': model_logz,
                'exact_logz': exact_logz,
                'logz_error': logz_error,
                'abs_error': abs(logz_error),
            })

    return pd.DataFrame(results)


def plot_model_vs_exact(df, output_dir='figs'):
    """
    Generate comprehensive plots comparing model vs exact values.

    Creates 6 subplots (3 rows × 2 columns):
    - Row 1: Log Z comparison
    - Row 2: Loss comparison
    - Row 3: Error metrics

    Args:
        df: DataFrame with test results
        output_dir: Directory to save plots (default: 'figs')
    """
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # Row 1: Log Z Comparison
    # Plot 1: Model log Z vs Exact log Z
    ax = axes[0, 0]
    ax.plot(df['T'], df['model_logz'], 'o-', label='Model log Z', markersize=4)
    ax.plot(df['T'], df['exact_logz'], 's-', label='Exact log Z', markersize=4)
    ax.axvline(CRITICAL_TEMPERATURE, color='r', linestyle='--', alpha=0.5,
               label=f'Tc = {CRITICAL_TEMPERATURE:.3f}')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('log Z')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Log Partition Function: Model vs Exact')

    # Plot 2: Log Z Error vs Temperature
    ax = axes[0, 1]
    ax.plot(df['T'], df['logz_error'], 'o-', markersize=4)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(CRITICAL_TEMPERATURE, color='r', linestyle='--', alpha=0.5,
               label=f'Tc = {CRITICAL_TEMPERATURE:.3f}')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Error (Model - Exact)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Log Z Error vs Temperature')

    # Row 2: Loss Comparison
    # Plot 3: Model Loss vs Exact Loss
    ax = axes[1, 0]
    ax.plot(df['T'], df['model_loss'], 'o-', label='Model Loss', markersize=4)
    ax.plot(df['T'], df['exact_loss'], 's-', label='Exact Loss', markersize=4)
    ax.axvline(CRITICAL_TEMPERATURE, color='r', linestyle='--', alpha=0.5,
               label=f'Tc = {CRITICAL_TEMPERATURE:.3f}')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Loss: Model vs Exact')

    # Plot 4: Loss Error vs Temperature
    ax = axes[1, 1]
    ax.plot(df['T'], df['loss_error'], 'o-', markersize=4)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(CRITICAL_TEMPERATURE, color='r', linestyle='--', alpha=0.5,
               label=f'Tc = {CRITICAL_TEMPERATURE:.3f}')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Error (Model - Exact)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Loss Error vs Temperature')

    # Row 3: Additional Metrics
    # Plot 5: Absolute Error (log scale)
    ax = axes[2, 0]
    ax.semilogy(df['T'], df['abs_error'], 'o-', markersize=4)
    ax.axvline(CRITICAL_TEMPERATURE, color='r', linestyle='--', alpha=0.5,
               label=f'Tc = {CRITICAL_TEMPERATURE:.3f}')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('|Log Z Error| (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Absolute Error (log scale)')

    # Plot 6: Error vs Reduced Temperature (T/Tc)
    ax = axes[2, 1]
    ax.plot(df['T/Tc'], df['logz_error'], 'o-', markersize=4)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='T/Tc = 1')
    ax.set_xlabel('T / Tc')
    ax.set_ylabel('Log Z Error (Model - Exact)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Error vs Reduced Temperature')

    plt.tight_layout()
    output_path = f'{output_dir}/model_test_analysis.png'
    plt.savefig(output_path, dpi=150)
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

    # Test model
    console.print("\n[bold]Testing model at multiple temperatures...[/bold]")
    df = test_model_ising(
        model, energy_fn, L,
        device=device,
        num_temps=20,
        batch_size=500
    )

    # Display results
    console.print("\n" + "="*70)
    console.print("MODEL TESTING RESULTS")
    console.print("="*70)
    console.print(df.to_string(index=False))
    console.print("="*70)

    # Summary statistics
    idx_min_error = df['abs_error'].idxmin()
    idx_max_error = df['abs_error'].idxmax()

    console.print(f"\n[bold green]Best accuracy:[/bold green]")
    console.print(f"  T = {df.loc[idx_min_error, 'T']:.4f} (T/Tc = {df.loc[idx_min_error, 'T/Tc']:.3f})")
    console.print(f"  |Error| = {df.loc[idx_min_error, 'abs_error']:.6f}")

    console.print(f"\n[bold red]Worst accuracy:[/bold red]")
    console.print(f"  T = {df.loc[idx_max_error, 'T']:.4f} (T/Tc = {df.loc[idx_max_error, 'T/Tc']:.3f})")
    console.print(f"  |Error| = {df.loc[idx_max_error, 'abs_error']:.6f}")

    # Error at critical temperature
    idx_critical = (df['T'] - CRITICAL_TEMPERATURE).abs().idxmin()
    console.print(f"\n[bold yellow]At critical temperature Tc = {CRITICAL_TEMPERATURE:.4f}:[/bold yellow]")
    console.print(f"  T = {df.loc[idx_critical, 'T']:.4f}")
    console.print(f"  Log Z Error = {df.loc[idx_critical, 'logz_error']:.6f}")
    console.print(f"  |Error| = {df.loc[idx_critical, 'abs_error']:.6f}")

    # Generate plots
    console.print("\n[bold]Generating plots...[/bold]")
    plot_path = plot_model_vs_exact(df)
    console.print(f"[green]✓[/green] Saved plot to {plot_path}")

    # Save results
    output_dir = f'runs/{project}/{group_name}'
    output_file = f'{output_dir}/test_results_{seed}.csv'
    df.to_csv(output_file, index=False)
    console.print(f"[green]✓[/green] Saved results to {output_file}")


if __name__ == "__main__":
    main()
