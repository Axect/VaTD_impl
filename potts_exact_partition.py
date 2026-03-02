"""
Exact thermodynamic quantities for the q-state Potts model on 2D square lattice.

No exact finite-lattice partition function exists for q > 2 (unlike Onsager's
solution for the Ising model). Instead, this module provides:
- Exact critical temperatures: Tc = 1 / ln(1 + √q)
- CFT central charges at criticality
- Numbers of relevant scaling operators from conformal field theory
- Baxter's bulk free energy (thermodynamic limit)

These serve as theoretical predictions for neural network verification.

References:
- Wu, F.Y. (1982). Rev. Mod. Phys. 54, 235.
- Baxter, R.J. (1982). "Exactly Solved Models in Statistical Mechanics."
- Di Francesco, P. et al. (1997). "Conformal Field Theory." Chapter 10.
- Dotsenko, Vl. S. (1984). "Critical behaviour and associated conformal
  algebra of the Z3 Potts model." Nucl. Phys. B 235, 54.
"""

import numpy as np


def critical_temperature(q: int) -> float:
    """Exact Tc = 1 / ln(1 + √q) for q-state Potts on 2D square lattice."""
    return 1.0 / np.log(1.0 + np.sqrt(q))


def critical_beta(q: int) -> float:
    """Exact βc = ln(1 + √q) for q-state Potts on 2D square lattice."""
    return np.log(1.0 + np.sqrt(q))


# ──────────────────────────────────────────────────────────────
# CFT Data: Central Charges and Relevant Operators
# ──────────────────────────────────────────────────────────────

# Critical temperatures Tc = 1 / ln(1 + √q)
POTTS_TC = {
    2: 2.0 / np.log(1.0 + np.sqrt(2.0)),  # ≈ 2.269 (Ising)
    3: critical_temperature(3),             # ≈ 0.995
    4: critical_temperature(4),             # ≈ 0.910
}

# Central charges from minimal model CFT: c = 1 - 6/[m(m+1)]
# q=2 (Ising): m=3 → c = 1/2
# q=3 (3-Potts): c = 4/5
# q=4 (4-Potts): c = 1
CENTRAL_CHARGES = {
    2: 0.5,
    3: 0.8,
    4: 1.0,
}

# Number of relevant scaling operators (Δ < 2) from CFT primary fields.
#
# q=2 (Ising, c=1/2):
#   - σ (spin, Δ=1/8), ε (energy, Δ=1), identity (Δ=0) → 3 relevant operators
#
# q=3 (3-Potts, c=4/5):
#   - Identity (Δ=0)
#   - σ, σ* (spin/anti-spin, Δ=2/15) — Z₃ pair
#   - ε (energy, Δ=4/5)
#   - X, X* (subleading spin, Δ=4/3) — Z₃ pair (marginally relevant check: 4/3<2 ✓)
#   → 6 relevant operators
#
# q=4 (4-Potts, c=1):
#   - Identity (Δ=0)
#   - σ₁, σ₂, σ₃ (Z₄ spin operators, Δ=1/8 each)
#   - ε (energy, Δ=1/2)
#   - Twist operators (Δ=1)
#   - Subleading spin operators
#   → 8 relevant operators
#
# Note: These counts include both holomorphic and anti-holomorphic sectors
# and represent the number of independent scaling fields relevant to the
# phase transition. The exact count can vary by convention (whether marginal
# operators Δ=2 are included, treatment of descendant fields, etc.).
RELEVANT_OPERATORS = {
    2: 3,
    3: 6,
    4: 8,
}

# Scaling dimensions of primary fields
SCALING_DIMENSIONS = {
    2: {
        "identity": 0.0,
        "spin": 1 / 8,       # σ
        "energy": 1.0,       # ε
    },
    3: {
        "identity": 0.0,
        "spin": 2 / 15,      # σ
        "spin_conj": 2 / 15,  # σ*
        "energy": 4 / 5,     # ε
        "subleading_spin": 4 / 3,  # X
        "subleading_spin_conj": 4 / 3,  # X*
    },
    4: {
        "identity": 0.0,
        "spin_1": 1 / 8,     # σ₁
        "spin_2": 1 / 8,     # σ₂
        "spin_3": 1 / 8,     # σ₃
        "energy": 1 / 2,     # ε
        "twist_1": 1.0,
        "twist_2": 1.0,
        "subleading_energy": 3 / 2,
    },
}


def baxter_bulk_free_energy(q: int, beta: float) -> float:
    """
    Baxter's bulk free energy per site in the thermodynamic limit (L → ∞).

    f(β) = -T ln(q) for T → ∞ (paramagnetic limit)
    f(βc) has known exact value from Baxter (1973).

    For finite systems, this serves as an approximate reference.
    Returns -β·f (dimensionless).

    Note: This is an approximation; the full Baxter solution involves
    elliptic functions. Here we provide the high-T and low-T limiting forms.
    """
    T = 1.0 / beta
    Tc = critical_temperature(q)

    if T > 2 * Tc:
        # High-T expansion: f ≈ -T·ln(q) - (β/2)·2·(q-1)/q² + ...
        return beta * T * np.log(q) + beta**2 * (q - 1) / q**2
    elif T < 0.5 * Tc:
        # Low-T: ground state energy -2J per site (2D square lattice, 2 bonds/site)
        return 2.0 * beta
    else:
        # Near Tc: interpolate (not exact, placeholder for validation)
        return beta * T * np.log(q) * 0.5 + beta


def get_model_info(q: int) -> dict:
    """Get all known theoretical quantities for a q-state Potts model."""
    return {
        "q": q,
        "Tc": critical_temperature(q),
        "beta_c": critical_beta(q),
        "central_charge": CENTRAL_CHARGES.get(q),
        "n_relevant_operators": RELEVANT_OPERATORS.get(q),
        "scaling_dimensions": SCALING_DIMENSIONS.get(q),
    }


if __name__ == "__main__":
    print("q-state Potts Model: Exact Thermodynamic Data")
    print("=" * 55)
    for q in [2, 3, 4]:
        info = get_model_info(q)
        print(f"\nq = {q}:")
        print(f"  Tc = {info['Tc']:.6f}")
        print(f"  βc = {info['beta_c']:.6f}")
        print(f"  c  = {info['central_charge']}")
        print(f"  # relevant operators = {info['n_relevant_operators']}")
        if info["scaling_dimensions"]:
            print(f"  Primary fields:")
            for name, dim in info["scaling_dimensions"].items():
                print(f"    {name}: Δ = {dim:.4f}")
