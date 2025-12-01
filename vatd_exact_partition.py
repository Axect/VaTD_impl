"""
Exact partition function for 2D Ising model.

Ported from ~/zbin/vatd/utils/ising/exactZ_torch.py

This module provides exact calculations of the partition function for the 2D Ising model
using the transfer matrix method. The exact solution is crucial for validating generative
models that learn to sample from the Boltzmann distribution.

References:
- Onsager, L. (1944). Crystal statistics. I. A two-dimensional model with an order-disorder transition.
  Physical Review, 65(3-4), 117.
"""

import torch
import numpy as np


def h(j, beta):
    """
    Helper function: h = beta * j

    Args:
        j: Interaction strength
        beta: Inverse temperature (1/T)

    Returns:
        beta * j
    """
    return beta * j


def h_star(j, beta):
    """
    Helper function: h* = arctanh(exp(-2*beta*j))

    This is the dual coupling constant in the transfer matrix formulation.

    Args:
        j: Interaction strength
        beta: Inverse temperature (1/T)

    Returns:
        Dual coupling constant
    """
    return torch.arctanh(torch.exp(-2 * beta * j))


def gamma(n, j, beta, r):
    """
    Gamma function for exact partition calculation.

    This function computes the eigenvalues of the transfer matrix.

    Args:
        n: Lattice size (n×n)
        j: Interaction strength
        beta: Inverse temperature (1/T)
        r: Index for summation

    Returns:
        Gamma value for given parameters
    """
    # Compute argument for arccosh
    arg = (torch.cosh(2*h_star(j, beta)) * torch.cosh(2*h(j, beta)) -
           torch.sinh(2*h_star(j, beta)) * torch.sinh(2*h(j, beta)) * np.cos(r*np.pi/n))

    # Add small epsilon to avoid numerical issues when arg is exactly 1
    # arccosh is only defined for x >= 1, and arccosh(1) = 0
    eps = 1e-10
    arg = torch.clamp(arg, min=1.0 + eps)

    return torch.arccosh(arg)


def logZ(n, j, beta):
    """
    Compute exact log partition function for 2D Ising model.

    Uses the exact Onsager solution via transfer matrix method. The partition function
    is computed as:

        Z = 2 * (2 sinh(2βJ))^(N/2) * Σ [products of eigenvalues]

    where the sum is over four terms corresponding to different boundary conditions.

    Args:
        n: Lattice size (n×n total sites)
        j: Interaction strength (typically 1.0 for standard Ising model)
        beta: Inverse temperature (1/T)

    Returns:
        Log partition function log(Z) as a scalar tensor

    Example:
        >>> # Compute at critical temperature
        >>> L = 16
        >>> T_c = 2.269  # Critical temperature
        >>> beta_c = 1.0 / T_c
        >>> logz = logZ(n=L, j=1.0, beta=torch.tensor(beta_c))
        >>> print(f"log Z = {logz.item():.6f}")
    """
    # Ensure beta is a tensor with float64 for numerical stability
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=torch.float64)
    else:
        beta = beta.to(torch.float64)

    # Four terms in the exact formula
    # Use small epsilon to avoid log(0) when sinh or cosh arguments are very small
    eps = 1e-10
    terms = []

    # Term 1: sum over even r, cosh
    term1 = torch.cat([
        torch.log(torch.clamp(2*torch.cosh(n/2*gamma(n, j, beta, 2*r)), min=eps)).reshape(1)
        for r in range(n)
    ], dim=0).sum(0, keepdim=True)
    terms.append(term1)

    # Term 2: sum over even r, sinh
    term2 = torch.cat([
        torch.log(torch.clamp(2*torch.sinh(n/2*gamma(n, j, beta, 2*r)), min=eps)).reshape(1)
        for r in range(n)
    ], dim=0).sum(0, keepdim=True)
    terms.append(term2)

    # Term 3: sum over odd r, cosh
    term3 = torch.cat([
        torch.log(torch.clamp(2*torch.cosh(n/2*gamma(n, j, beta, 2*r+1)), min=eps)).reshape(1)
        for r in range(n)
    ], dim=0).sum(0, keepdim=True)
    terms.append(term3)

    # Term 4: sum over odd r, sinh
    term4 = torch.cat([
        torch.log(torch.clamp(2*torch.sinh(n/2*gamma(n, j, beta, 2*r+1)), min=eps)).reshape(1)
        for r in range(n)
    ], dim=0).sum(0, keepdim=True)
    terms.append(term4)

    # Combine all terms using logsumexp for numerical stability
    result = (
        torch.logsumexp(torch.cat(terms, dim=0), dim=0)
        - torch.log(torch.tensor(2.))
        + 1/2*n**2*torch.log(2*torch.sinh(2*h(j, beta)))
    )

    return result


def freeEnergy(n, j, beta):
    """
    Compute free energy per site: F = -1/(n^2 * beta) * log Z

    Args:
        n: Lattice size (n×n)
        j: Interaction strength
        beta: Inverse temperature (1/T)

    Returns:
        Free energy per site
    """
    return -1/n**2/beta * logZ(n, j, beta)


# Critical temperature for 2D Ising model (exact value from Onsager)
# Tc = 2J / log(1 + sqrt(2))  where J=1
CRITICAL_TEMPERATURE = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.269185


if __name__ == '__main__':
    """
    Test the exact partition function at various temperatures.
    """
    print("="*70)
    print("Exact Partition Function for 2D Ising Model")
    print("="*70)

    L = 16
    print(f"\nLattice size: {L}×{L}")
    print(f"Critical temperature: Tc = {CRITICAL_TEMPERATURE:.6f}")
    print(f"                      βc = {1.0/CRITICAL_TEMPERATURE:.6f}")
    print("\n" + "-"*70)

    # Test at various temperatures
    test_temps = [
        ("High temp (2×Tc)", 2.0 * CRITICAL_TEMPERATURE),
        ("Above Tc (1.2×Tc)", 1.2 * CRITICAL_TEMPERATURE),
        ("Critical (Tc)", CRITICAL_TEMPERATURE),
        ("Below Tc (0.8×Tc)", 0.8 * CRITICAL_TEMPERATURE),
        ("Low temp (0.5×Tc)", 0.5 * CRITICAL_TEMPERATURE),
    ]

    print(f"{'Temperature':<20} {'T':<10} {'β':<10} {'log Z':<15} {'F/site':<12}")
    print("-"*70)

    for desc, T in test_temps:
        beta = 1.0 / T
        logz = logZ(n=L, j=1.0, beta=torch.tensor(beta))
        free_energy = freeEnergy(L, 1.0, beta)

        print(f"{desc:<20} {T:>9.4f} {beta:>9.4f} {logz.item():>14.6f} {free_energy.item():>11.6f}")

    print("="*70)
