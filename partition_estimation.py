"""
Partition Function Estimation for Discrete Flow Matching.

This module provides methods to estimate the partition function Z of the
2D Ising model using a trained generative model:

1. Thermodynamic Integration (TI):
   log Z(β) = log Z(0) + ∫₀^β ⟨E⟩_β' dβ'

2. ELBO-based estimation:
   log Z ≥ -F[q] = -E_q[log q + βE]

Both methods are compared against the exact Onsager solution for validation.
"""

import torch
import numpy as np
from typing import Callable, Optional, List, Tuple
import math

# Import exact partition function
from vatd_exact_partition import logZ as exact_logZ, CRITICAL_TEMPERATURE


def thermodynamic_integration(
    model,
    energy_fn: Callable,
    beta_target: float,
    n_points: int = 50,
    n_samples: int = 1000,
    fix_first: Optional[int] = 1,
    device: str = "cuda"
) -> Tuple[torch.Tensor, dict]:
    """
    Compute log partition function via thermodynamic integration.

    The partition function is computed using the thermodynamic identity:

        ∂/∂β log Z(β) = -⟨E⟩_β

    Integrating from β=0 (where Z = 2^N) to β_target:

        log Z(β) = log Z(0) - ∫₀^β ⟨E⟩_β' dβ'

    Note: The negative sign comes from the fact that ⟨E⟩ is typically negative
    for the Ising model at low temperatures.

    Args:
        model: Trained generative model with sample(batch_size, T) method
        energy_fn: Energy function E(x) returning (B, 1) tensor
        beta_target: Target inverse temperature
        n_points: Number of integration points
        n_samples: Number of samples per temperature for Monte Carlo average
        fix_first: If not None, first spin is fixed (reduces Z by factor of 2)
        device: Device to run computations on

    Returns:
        Tuple of (log_Z estimate, metrics dict with integration details)
    """
    model.eval()

    # Lattice size
    if hasattr(model, 'size'):
        H, W = model.size
    else:
        H, W = 16, 16
    N = H * W

    # log Z(0) = log(2^N) for free spins
    # If first spin is fixed, log Z(0) = log(2^(N-1))
    if fix_first is not None:
        log_Z0 = (N - 1) * math.log(2)
    else:
        log_Z0 = N * math.log(2)

    # Beta grid (avoid exactly 0 for numerical stability)
    beta_min = 0.001
    betas = torch.linspace(beta_min, beta_target, n_points)

    # Compute mean energy at each beta
    mean_energies = []
    std_energies = []

    with torch.no_grad():
        for beta in betas:
            T = 1.0 / beta.item()
            T_tensor = torch.full((n_samples,), T, device=device)

            # Sample from model
            samples = model.sample(batch_size=n_samples, T=T_tensor)

            # Compute energies
            energies = energy_fn(samples).squeeze(-1)  # (n_samples,)

            mean_E = energies.mean().item()
            std_E = energies.std().item()

            mean_energies.append(mean_E)
            std_energies.append(std_E)

    # Convert to tensors
    mean_energies = torch.tensor(mean_energies)
    std_energies = torch.tensor(std_energies)
    betas_np = betas.numpy()

    # Trapezoidal integration: ∫⟨E⟩dβ
    # Note: We integrate the mean energy, not -⟨E⟩
    # The formula is: log Z(β) = log Z(0) - ∫⟨E⟩dβ
    # But since ⟨E⟩ < 0 at low T, this becomes: log Z(β) = log Z(0) + |∫⟨E⟩dβ|
    integral = np.trapz(mean_energies.numpy(), betas_np)

    # log Z(β) = log Z(0) - integral of ⟨E⟩
    log_Z = log_Z0 - integral

    # Compute uncertainty estimate (propagation of std through trapezoid rule)
    # Very rough estimate
    d_beta = beta_target / (n_points - 1)
    integral_std = d_beta * np.sqrt(np.sum(np.array(std_energies) ** 2 / n_samples))

    metrics = {
        "log_Z": log_Z,
        "log_Z0": log_Z0,
        "integral": integral,
        "integral_std": integral_std,
        "mean_energies": mean_energies.tolist(),
        "betas": betas.tolist(),
        "n_points": n_points,
        "n_samples": n_samples,
    }

    return torch.tensor(log_Z), metrics


def elbo_partition(
    model,
    energy_fn: Callable,
    T: float,
    n_samples: int = 1000,
    device: str = "cuda"
) -> Tuple[torch.Tensor, dict]:
    """
    Estimate log partition function using ELBO.

    The variational free energy provides a bound on log Z:

        F[q] = E_q[log q(x) + β·E(x)] ≥ -log Z

    Therefore:
        log Z ≥ -F[q]

    This is the same objective used in VaTD training.

    Args:
        model: Trained generative model with sample() and log_prob() methods
        energy_fn: Energy function E(x) returning (B, 1) tensor
        T: Temperature
        n_samples: Number of samples for Monte Carlo estimate
        device: Device to run computations on

    Returns:
        Tuple of (log_Z lower bound, metrics dict)
    """
    model.eval()

    beta = 1.0 / T
    T_tensor = torch.full((n_samples,), T, device=device)

    with torch.no_grad():
        # Sample from model
        samples = model.sample(batch_size=n_samples, T=T_tensor)

        # Compute log probability
        log_q = model.log_prob(samples, T_tensor)  # (n_samples, 1)

        # Compute energy
        energy = energy_fn(samples)  # (n_samples, 1)

        # Variational free energy: F = E[log q + β·E]
        F_samples = log_q + beta * energy  # (n_samples, 1)
        F_mean = F_samples.mean().item()
        F_std = F_samples.std().item()

    # log Z ≥ -F
    log_Z_lower = -F_mean

    metrics = {
        "log_Z_lower": log_Z_lower,
        "F_mean": F_mean,
        "F_std": F_std,
        "log_q_mean": log_q.mean().item(),
        "energy_mean": energy.mean().item(),
        "temperature": T,
        "beta": beta,
        "n_samples": n_samples,
    }

    return torch.tensor(log_Z_lower), metrics


def validate_partition_function(
    model,
    energy_fn: Callable,
    val_betas: List[float],
    lattice_size: int = 16,
    n_samples_ti: int = 1000,
    n_points_ti: int = 50,
    n_samples_elbo: int = 1000,
    fix_first: Optional[int] = 1,
    device: str = "cuda",
    verbose: bool = True
) -> dict:
    """
    Validate partition function estimates against exact Onsager solution.

    Computes both TI and ELBO estimates at multiple temperatures and
    compares with the exact analytical result.

    Args:
        model: Trained generative model
        energy_fn: Energy function
        val_betas: List of beta values for validation
        lattice_size: Lattice size L (L×L lattice)
        n_samples_ti: Samples per temperature for TI
        n_points_ti: Integration points for TI
        n_samples_elbo: Samples for ELBO estimation
        fix_first: If not None, first spin is fixed
        device: Device for computations
        verbose: Print results

    Returns:
        Dictionary with validation results
    """
    results = {
        "betas": [],
        "temperatures": [],
        "log_Z_exact": [],
        "log_Z_ti": [],
        "log_Z_elbo": [],
        "error_ti": [],
        "error_elbo": [],
        "relative_error_ti": [],
        "relative_error_elbo": [],
    }

    if verbose:
        print("=" * 80)
        print("Partition Function Validation")
        print("=" * 80)
        print(f"Lattice size: {lattice_size}×{lattice_size}")
        print(f"Critical temperature: Tc = {CRITICAL_TEMPERATURE:.4f}")
        print(f"First spin fixed: {fix_first is not None}")
        print("-" * 80)
        header = f"{'β':>8} {'T':>8} {'T/Tc':>8} {'log Z exact':>14} {'log Z TI':>14} {'log Z ELBO':>14} {'Err TI':>10} {'Err ELBO':>10}"
        print(header)
        print("-" * 80)

    for beta in val_betas:
        T = 1.0 / beta

        # Exact partition function
        log_Z_exact_raw = exact_logZ(n=lattice_size, j=1.0, beta=torch.tensor(beta))

        # If first spin is fixed, adjust by log(2)
        if fix_first is not None:
            log_Z_exact = log_Z_exact_raw.item() - math.log(2)
        else:
            log_Z_exact = log_Z_exact_raw.item()

        # Thermodynamic Integration
        log_Z_ti, ti_metrics = thermodynamic_integration(
            model, energy_fn, beta,
            n_points=n_points_ti,
            n_samples=n_samples_ti,
            fix_first=fix_first,
            device=device
        )
        log_Z_ti = log_Z_ti.item()

        # ELBO
        log_Z_elbo, elbo_metrics = elbo_partition(
            model, energy_fn, T,
            n_samples=n_samples_elbo,
            device=device
        )
        log_Z_elbo = log_Z_elbo.item()

        # Errors
        error_ti = abs(log_Z_ti - log_Z_exact)
        error_elbo = abs(log_Z_elbo - log_Z_exact)
        rel_error_ti = error_ti / abs(log_Z_exact) * 100
        rel_error_elbo = error_elbo / abs(log_Z_exact) * 100

        # Store results
        results["betas"].append(beta)
        results["temperatures"].append(T)
        results["log_Z_exact"].append(log_Z_exact)
        results["log_Z_ti"].append(log_Z_ti)
        results["log_Z_elbo"].append(log_Z_elbo)
        results["error_ti"].append(error_ti)
        results["error_elbo"].append(error_elbo)
        results["relative_error_ti"].append(rel_error_ti)
        results["relative_error_elbo"].append(rel_error_elbo)

        if verbose:
            T_ratio = T / CRITICAL_TEMPERATURE
            print(f"{beta:>8.4f} {T:>8.4f} {T_ratio:>8.4f} {log_Z_exact:>14.4f} {log_Z_ti:>14.4f} {log_Z_elbo:>14.4f} {rel_error_ti:>9.2f}% {rel_error_elbo:>9.2f}%")

    if verbose:
        print("=" * 80)
        print(f"Mean relative error TI: {np.mean(results['relative_error_ti']):.2f}%")
        print(f"Mean relative error ELBO: {np.mean(results['relative_error_elbo']):.2f}%")
        print("=" * 80)

    return results


def compute_free_energy_curve(
    model,
    energy_fn: Callable,
    beta_range: Tuple[float, float] = (0.1, 1.5),
    n_betas: int = 30,
    n_samples: int = 500,
    lattice_size: int = 16,
    fix_first: Optional[int] = 1,
    device: str = "cuda"
) -> dict:
    """
    Compute the free energy curve F(β) = -log Z(β) / (β·N).

    This is useful for visualizing phase transitions and comparing
    with the exact solution.

    Args:
        model: Trained generative model
        energy_fn: Energy function
        beta_range: (beta_min, beta_max) range
        n_betas: Number of beta points
        n_samples: Samples per temperature
        lattice_size: Lattice size L
        fix_first: If not None, first spin is fixed
        device: Device for computations

    Returns:
        Dictionary with free energy curves
    """
    betas = np.linspace(beta_range[0], beta_range[1], n_betas)
    N = lattice_size * lattice_size

    free_energies_model = []
    free_energies_exact = []
    energies_model = []
    magnetizations = []

    model.eval()

    with torch.no_grad():
        for beta in betas:
            T = 1.0 / beta
            T_tensor = torch.full((n_samples,), T, device=device)

            # Sample from model
            samples = model.sample(batch_size=n_samples, T=T_tensor)

            # Compute observables
            energy = energy_fn(samples).squeeze(-1).mean().item()
            log_q = model.log_prob(samples, T_tensor).squeeze(-1).mean().item()

            # Free energy from ELBO: F ≈ (log q + β·E) / β
            F_model = (log_q + beta * energy) / beta / N

            # Exact free energy
            log_Z_exact = exact_logZ(n=lattice_size, j=1.0, beta=torch.tensor(beta))
            if fix_first is not None:
                log_Z_exact = log_Z_exact - math.log(2)
            F_exact = -log_Z_exact.item() / beta / N

            # Magnetization
            m = samples.mean().item()

            free_energies_model.append(F_model)
            free_energies_exact.append(F_exact)
            energies_model.append(energy / N)
            magnetizations.append(abs(m))

    return {
        "betas": betas.tolist(),
        "temperatures": (1.0 / betas).tolist(),
        "F_model": free_energies_model,
        "F_exact": free_energies_exact,
        "E_per_site": energies_model,
        "magnetization": magnetizations,
        "T_c": CRITICAL_TEMPERATURE,
        "beta_c": 1.0 / CRITICAL_TEMPERATURE,
    }


if __name__ == "__main__":
    """Test partition function estimation with a simple model."""
    print("Testing Partition Function Estimation")
    print("=" * 70)

    # Import model
    from model_dfm import DiscreteFlowMatcher

    # Test configuration
    hparams = {
        "size": 8,
        "fix_first": 1,
        "batch_size": 64,
        "num_beta": 4,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "num_flow_steps": 30,
        "t_max": 5.0,
        "hidden_channels": 32,
        "hidden_conv_layers": 2,
        "hidden_width": 64,
        "hidden_fc_layers": 1,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model (untrained, just for testing interface)
    model = DiscreteFlowMatcher(hparams, device=device).to(device)

    # Simple energy function
    def energy_fn(x):
        right = torch.roll(x, -1, dims=-1)
        down = torch.roll(x, -1, dims=-2)
        energy = -(x * right + x * down).sum(dim=[1, 2, 3])
        return energy.unsqueeze(-1)

    # Test Thermodynamic Integration
    print("\n--- Testing Thermodynamic Integration ---")
    log_Z_ti, ti_metrics = thermodynamic_integration(
        model, energy_fn, beta_target=0.5,
        n_points=10, n_samples=100,
        fix_first=1, device=device
    )
    print(f"log Z (TI): {log_Z_ti.item():.4f}")
    print(f"log Z(0): {ti_metrics['log_Z0']:.4f}")
    print(f"Integral: {ti_metrics['integral']:.4f}")

    # Test ELBO
    print("\n--- Testing ELBO Estimation ---")
    log_Z_elbo, elbo_metrics = elbo_partition(
        model, energy_fn, T=2.0,
        n_samples=100, device=device
    )
    print(f"log Z (ELBO lower bound): {log_Z_elbo.item():.4f}")
    print(f"F mean: {elbo_metrics['F_mean']:.4f}")

    # Test validation (with untrained model, errors will be large)
    print("\n--- Testing Validation ---")
    val_betas = [0.3, 0.5, 0.7]
    results = validate_partition_function(
        model, energy_fn, val_betas,
        lattice_size=8,
        n_samples_ti=50, n_points_ti=10,
        n_samples_elbo=50,
        fix_first=1, device=device,
        verbose=True
    )

    print("\nNote: Large errors expected with untrained model!")
    print("=" * 70)
