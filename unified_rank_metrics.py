"""
Unified Rank Metrics Framework for Low-Rank Analysis.

Implements the two-parameter Rényi-normalization family:

    R(α, norm) = exp(H_α(p^(norm)))

where α ∈ [0, ∞] is the Rényi order and norm ∈ {L1, L2sq} is the
normalization convention.

Grid positions:

    α \\ norm     │  L1: p_i = σ_i/Σσ_j   │  L2sq: p̃_i = σ_i²/Σσ_j²
    ─────────────┼────────────────────────┼──────────────────────────
    0            │  algebraic rank        │  algebraic rank
    1 (Shannon)  │  eRank (primary)       │  von Neumann eRank (entanglement)
    2 (collision)│  Rényi-2 rank          │  participation ratio
    ∞ (min)      │  nuclear rank          │  stable rank

Additionally provides standalone metrics not in the grid:
  - numerical_rank (integer, threshold-based)
  - elbow_rank (integer, geometric)
  - optimal_hard_threshold (integer, RMT-based Gavish-Donoho)
  - spectral_gap_ratio (continuous, gap-based)

References:
  [1] Roy & Vetterli, "The effective rank", EUSIPCO (2007)
  [2] Rényi, "On measures of entropy and information", Berkeley Symp. (1961)
  [3] Hill, "Diversity and evenness", Ecology (1973)
  [4] Gavish & Donoho, IEEE Trans. Inf. Theory (2014)
  [5] Thibeault et al., Nature Physics (2024)
"""

import torch
import numpy as np
from typing import Optional, Literal

# Threshold for treating singular values as zero
_EPS = 1e-10


# ──────────────────────────────────────────────────────────────
# Core: Unified Two-Parameter Family
# ──────────────────────────────────────────────────────────────


def _normalize(
    sv: torch.Tensor,
    norm: Literal["L1", "L2sq"] = "L1",
) -> torch.Tensor:
    """
    Normalize singular values into a probability distribution.

    Args:
        sv: non-negative singular values (already filtered for > 0).
        norm: 'L1'   → p_i = σ_i / Σσ_j       (amplitude weighting)
              'L2sq' → p̃_i = σ_i² / Σσ_j²     (energy/variance weighting)
    """
    if norm == "L1":
        return sv / sv.sum()
    elif norm == "L2sq":
        sv2 = sv ** 2
        return sv2 / sv2.sum()
    else:
        raise ValueError(f"Unknown norm: {norm!r}. Use 'L1' or 'L2sq'.")


def _renyi_entropy(
    p: torch.Tensor,
    alpha: float,
) -> float:
    """
    Rényi entropy H_α(p).

    H_α(p) = 1/(1-α) · ln(Σ p_i^α)    for α ≠ 1
    H_1(p) = -Σ p_i ln p_i              (Shannon, α → 1 limit)
    H_∞(p) = -ln(max p_i)               (min-entropy, α → ∞ limit)
    """
    if alpha == float("inf") or alpha > 1e6:
        # Min-entropy: H_∞ = -ln(max p_i)
        return -torch.log(p.max()).item()
    elif abs(alpha - 1.0) < 1e-8:
        # Shannon entropy: H_1 = -Σ p_i ln p_i
        return -(p * torch.log(p)).sum().item()
    else:
        # General Rényi: H_α = 1/(1-α) · ln(Σ p_i^α)
        return (1.0 / (1.0 - alpha)) * torch.log((p ** alpha).sum()).item()


def renyi_effective_rank(
    singular_values: torch.Tensor,
    alpha: float = 1.0,
    norm: Literal["L1", "L2sq"] = "L1",
) -> float:
    """
    Unified rank metric: exp(H_α(p^(norm))).

    This is the core function of the two-parameter framework.

    Args:
        singular_values: 1D tensor of singular values (sorted descending).
        alpha: Rényi order.
            0.5  → tail-sensitive
            1.0  → Shannon (eRank or vN-eRank depending on norm)
            2.0  → collision entropy
            ∞    → min-entropy (most conservative)
        norm: normalization convention.
            'L1'   → p_i = σ_i / Σσ_j  (amplitude weighting)
            'L2sq' → p̃_i = σ_i² / Σσ_j² (energy/variance weighting)

    Returns:
        Effective rank as a float in [1, min(m, n)].

    Grid mapping:
        (α=1,  L1)   = eRank (Roy & Vetterli, 2007)
        (α=1,  L2sq) = von Neumann effective rank = exp(S_vN)
        (α=2,  L1)   = Rényi-2 rank
        (α=2,  L2sq) = participation ratio
        (α=∞,  L1)   = nuclear rank (Thibeault et al., 2024)
        (α=∞,  L2sq) = stable rank
    """
    sv = singular_values[singular_values > _EPS]
    if len(sv) == 0:
        return 1.0

    p = _normalize(sv, norm)
    H = _renyi_entropy(p, alpha)
    return np.exp(H)


# ──────────────────────────────────────────────────────────────
# Convenience Wrappers (Named Metrics)
# ──────────────────────────────────────────────────────────────


def effective_rank(singular_values: torch.Tensor) -> float:
    """eRank = exp(H(p)), p_i = σ_i/Σσ_j. Grid position: (α=1, L1)."""
    return renyi_effective_rank(singular_values, alpha=1.0, norm="L1")


def von_neumann_effective_rank(singular_values: torch.Tensor) -> float:
    """
    Von Neumann effective rank: exp(S_vN).

    S_vN = -Σ p̃_i ln p̃_i, where p̃_i = σ_i² / Σσ_j².

    This is the Shannon entropy of the *eigenvalue* distribution of the
    covariance matrix A^T A / Tr(A^T A), exponentiated to give a count.

    Grid position: (α=1, L2sq).

    Physical interpretation:
      - Connects to entanglement entropy via the reduced density matrix
      - The correct metric for Calabrese-Cardy analysis
      - Always ≤ eRank (L2sq normalization concentrates the distribution)
    """
    return renyi_effective_rank(singular_values, alpha=1.0, norm="L2sq")


def stable_rank(singular_values: torch.Tensor) -> float:
    """
    Stable rank = Σσ² / σ_max² = 1/max(p̃_i).

    Grid position: (α=∞, L2sq).
    Most conservative continuous rank measure.
    """
    return renyi_effective_rank(singular_values, alpha=float("inf"), norm="L2sq")


def participation_ratio(singular_values: torch.Tensor) -> float:
    """
    Participation ratio = (Σσ²)² / Σσ⁴ = 1/Σp̃_i².

    Grid position: (α=2, L2sq).
    Standard IPR from condensed matter physics.
    """
    return renyi_effective_rank(singular_values, alpha=2.0, norm="L2sq")


def nuclear_rank(singular_values: torch.Tensor) -> float:
    """
    Nuclear rank = Σσ / σ_max = 1/max(p_i).

    Grid position: (α=∞, L1).
    From Thibeault et al., Nature Physics (2024).
    """
    return renyi_effective_rank(singular_values, alpha=float("inf"), norm="L1")


def renyi2_rank(singular_values: torch.Tensor) -> float:
    """
    Rényi-2 rank (collision rank) = (Σσ)² / Σσ² = 1/Σp_i².

    Grid position: (α=2, L1).
    """
    return renyi_effective_rank(singular_values, alpha=2.0, norm="L1")


# ──────────────────────────────────────────────────────────────
# Raw Entropy (H without exponentiation)
# ──────────────────────────────────────────────────────────────


def shannon_entropy(
    singular_values: torch.Tensor,
    norm: Literal["L1", "L2sq"] = "L1",
) -> float:
    """
    Raw Shannon entropy H(p) without exponentiation.

    Returns entropy in nats. Related to eRank by: eRank = exp(H).

    Use this for:
      - Information-theoretic decompositions (additive quantities)
      - Rate-distortion analysis
      - When the question is "how many bits?" not "how many dimensions?"
    """
    sv = singular_values[singular_values > _EPS]
    if len(sv) == 0:
        return 0.0
    p = _normalize(sv, norm)
    return _renyi_entropy(p, alpha=1.0)


def von_neumann_entropy(singular_values: torch.Tensor) -> float:
    """
    Von Neumann entropy S_vN = -Σ p̃_i ln p̃_i, p̃_i = σ_i²/Σσ_j².

    The entropy of the eigenvalue distribution of the normalized
    covariance matrix. This is what appears in the Calabrese-Cardy formula.
    """
    return shannon_entropy(singular_values, norm="L2sq")


# ──────────────────────────────────────────────────────────────
# Full Spectrum Computation
# ──────────────────────────────────────────────────────────────

# Standard α values for the Rényi spectrum
STANDARD_ALPHAS = [0.5, 1.0, 2.0, 5.0, float("inf")]
STANDARD_NORMS: list[Literal["L1", "L2sq"]] = ["L1", "L2sq"]

# Named grid positions
GRID_NAMES = {
    (1.0, "L1"): "eRank",
    (1.0, "L2sq"): "vN_eRank",
    (2.0, "L1"): "renyi2_rank",
    (2.0, "L2sq"): "participation_ratio",
    (float("inf"), "L1"): "nuclear_rank",
    (float("inf"), "L2sq"): "stable_rank",
}


def compute_full_grid(
    singular_values: torch.Tensor,
    alphas: Optional[list[float]] = None,
    norms: Optional[list[Literal["L1", "L2sq"]]] = None,
) -> dict[str, float]:
    """
    Compute the full Rényi-normalization grid of rank metrics.

    Returns a flat dict with keys like 'alpha_1.0_L1' (= eRank),
    'alpha_1.0_L2sq' (= vN-eRank), etc. Named metrics are also
    included as aliases.

    Args:
        singular_values: 1D tensor of singular values.
        alphas: list of Rényi orders (default: [0.5, 1, 2, 5, ∞]).
        norms: list of normalizations (default: ['L1', 'L2sq']).

    Returns:
        Dict mapping metric keys to values. Example keys:
          'alpha_0.5_L1', 'alpha_1.0_L1' (= 'eRank'), 'alpha_inf_L1' (= 'nuclear_rank'),
          'alpha_1.0_L2sq' (= 'vN_eRank'), 'alpha_2.0_L2sq' (= 'participation_ratio'), ...
    """
    if alphas is None:
        alphas = STANDARD_ALPHAS
    if norms is None:
        norms = STANDARD_NORMS

    result = {}
    for alpha in alphas:
        for norm in norms:
            value = renyi_effective_rank(singular_values, alpha=alpha, norm=norm)

            # Canonical key
            alpha_str = "inf" if alpha == float("inf") else f"{alpha}"
            key = f"alpha_{alpha_str}_{norm}"
            result[key] = value

            # Named alias if it exists
            name = GRID_NAMES.get((alpha, norm))
            if name:
                result[name] = value

    return result


def compute_entropy_grid(
    singular_values: torch.Tensor,
    alphas: Optional[list[float]] = None,
    norms: Optional[list[Literal["L1", "L2sq"]]] = None,
) -> dict[str, float]:
    """
    Same as compute_full_grid but returns raw entropies H_α(p)
    instead of exp(H_α(p)).

    Useful for additive decompositions and information-theoretic analysis.
    """
    if alphas is None:
        alphas = STANDARD_ALPHAS
    if norms is None:
        norms = STANDARD_NORMS

    sv = singular_values[singular_values > _EPS]
    if len(sv) == 0:
        return {f"H_alpha_{a}_{n}": 0.0 for a in alphas for n in norms}

    result = {}
    for alpha in alphas:
        for norm in norms:
            p = _normalize(sv, norm)
            H = _renyi_entropy(p, alpha)
            alpha_str = "inf" if alpha == float("inf") else f"{alpha}"
            result[f"H_alpha_{alpha_str}_{norm}"] = H

    return result


# ──────────────────────────────────────────────────────────────
# Standalone Metrics (Not in the Grid)
# ──────────────────────────────────────────────────────────────


def numerical_rank(singular_values: torch.Tensor, threshold: float = 0.99) -> int:
    """
    Numerical rank: min k s.t. Σ_{i≤k} σ_i² / Σσ_j² ≥ threshold.

    Integer-valued. Not part of the Rényi grid (threshold-based, not entropic).
    """
    sv = singular_values[singular_values > _EPS]
    if len(sv) == 0:
        return 1
    energy = sv ** 2
    total = energy.sum()
    cumsum = torch.cumsum(energy, dim=0)
    mask = cumsum >= threshold * total
    if mask.any():
        return int((mask.float().argmax() + 1).item())
    return len(sv)


def elbow_rank(singular_values: torch.Tensor) -> int:
    """
    Elbow rank: argmax of |y_k - x_k| where y = cumulative energy
    fraction, x = k/n (uniform baseline).

    Integer-valued. Geometric method from Thibeault et al. (2024).
    """
    sv = singular_values[singular_values > _EPS]
    if len(sv) <= 1:
        return 1
    n = len(sv)
    x = torch.arange(1, n + 1, dtype=torch.float64) / n
    energy = sv.double() ** 2
    y = torch.cumsum(energy, dim=0) / energy.sum()
    dist = (y - x).abs()
    return int((dist.argmax() + 1).item())


def optimal_hard_threshold(
    singular_values: torch.Tensor,
    N: int,
    M: int,
) -> int:
    """
    Gavish-Donoho (2014) optimal hard threshold.

    Returns count of singular values above the RMT-optimal threshold.
    Minimax-optimal under the spiked covariance model.

    Args:
        N: number of rows (samples).
        M: number of columns (features).
    """
    sv = singular_values[singular_values > _EPS]
    if len(sv) <= 1:
        return 1

    beta = min(N, M) / max(N, M)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43

    sv_median = sv.median().item()
    mu_beta = np.sqrt(2 * beta + beta ** 2 / 3)
    if mu_beta < 1e-12:
        return len(sv)

    sigma_noise = sv_median / (mu_beta * np.sqrt(max(N, M)))
    threshold = omega * sigma_noise * np.sqrt(max(N, M))

    count = int((sv > threshold).sum().item())
    return max(count, 1)


def spectral_gap_ratio(
    singular_values: torch.Tensor,
    k: Optional[int] = None,
) -> float:
    """
    Spectral gap ratio: (σ_k - σ_{k+1}) / σ_k.

    Args:
        k: index at which to measure the gap (1-indexed).
           If None, returns the maximum relative gap across all k.
           Use k=3 for Ising (CFT prediction), k=6 for 3-Potts, etc.
    """
    sv = singular_values[singular_values > _EPS]
    if len(sv) <= 1:
        return 0.0

    gaps = (sv[:-1] - sv[1:]) / (sv[:-1] + _EPS)

    if k is not None:
        idx = k - 1  # 0-indexed
        if idx < 0 or idx >= len(gaps):
            return 0.0
        return gaps[idx].item()
    else:
        return gaps.max().item()


def bbp_threshold(gamma: float, sigma: float) -> float:
    """
    Baik-Ben Arous-Péché (BBP) phase transition threshold.

    Returns the upper edge of the Marchenko-Pastur distribution:
        σ · (1 + √γ)

    Singular values above this threshold are statistical outliers,
    corresponding to genuine signal components (spiked eigenvalues).

    Args:
        gamma: aspect ratio min(N, M) / max(N, M), in (0, 1].
        sigma: estimated noise standard deviation.

    Returns:
        BBP threshold (float). SVs above this are signal outliers.

    Reference:
        Baik, Ben Arous & Péché, "Phase transition of the largest eigenvalue
        for nontrivial mean in spiked random matrix models", Ann. Probab. (2005)
    """
    return sigma * (1.0 + np.sqrt(gamma))


def marchenko_pastur_outlier_count(
    singular_values: torch.Tensor,
    N: int,
    M: int,
) -> int:
    """
    Count singular values exceeding the BBP threshold (MP outliers).

    Estimates noise σ from the median SV using Gavish-Donoho's
    asymptotic formula (same as optimal_hard_threshold), then
    counts SVs above σ·(1 + √γ).

    This gives an integer count of statistically significant
    spectral components — at T_c, this should match the number
    of relevant CFT operators (3 for Ising, 6 for 3-Potts, 8 for 4-Potts).

    Args:
        singular_values: 1D tensor of singular values (sorted descending).
        N: number of rows (samples).
        M: number of columns (features).

    Returns:
        Integer count of outlier singular values (≥ 1).

    Reference:
        Marchenko & Pastur, "Distribution of eigenvalues for some sets of
        random matrices", Mat. Sb. (1967)
    """
    sv = singular_values[singular_values > _EPS]
    if len(sv) <= 1:
        return 1

    gamma = min(N, M) / max(N, M)

    # Estimate noise σ from median SV (Gavish-Donoho style)
    sv_median = sv.median().item()
    mu_beta = np.sqrt(2 * gamma + gamma ** 2 / 3)
    if mu_beta < 1e-12:
        return len(sv)

    sigma_noise = sv_median / (mu_beta * np.sqrt(max(N, M)))

    # BBP threshold
    threshold = bbp_threshold(gamma, sigma_noise) * np.sqrt(max(N, M))

    count = int((sv > threshold).sum().item())
    return max(count, 1)


# ──────────────────────────────────────────────────────────────
# Ordering Theorem Verification
# ──────────────────────────────────────────────────────────────


def verify_ordering(singular_values: torch.Tensor) -> dict[str, float]:
    """
    Compute all grid metrics and verify the ordering theorem.

    The correct ordering consists of three chains:

    1. Within L1 norm (fixed norm, increasing α):
       algebraic_rank ≥ eRank ≥ Rényi-2(L1) ≥ nuclear_rank ≥ 1

    2. Within L2sq norm (fixed norm, increasing α):
       algebraic_rank ≥ vN_eRank ≥ PR ≥ stable_rank ≥ 1

    3. Cross-norm at same α (L1 ≥ L2sq for same α):
       eRank ≥ vN_eRank
       Rényi-2(L1) ≥ PR
       nuclear_rank ≥ stable_rank

    NOTE: The interleaved ordering eRank ≥ vN ≥ R2_L1 ≥ ...
    does NOT hold in general because L1 and L2sq produce different
    probability distributions, and Rényi ordering only applies
    within a fixed distribution.

    Returns dict with metric values and an 'ordering_holds' boolean.
    """
    sv = singular_values[singular_values > _EPS]
    alg_rank = float(len(sv))

    metrics = {
        "algebraic_rank": alg_rank,
        "eRank": effective_rank(singular_values),
        "vN_eRank": von_neumann_effective_rank(singular_values),
        "renyi2_L1": renyi2_rank(singular_values),
        "participation_ratio": participation_ratio(singular_values),
        "nuclear_rank": nuclear_rank(singular_values),
        "stable_rank": stable_rank(singular_values),
    }

    tol = 1e-6

    # Chain 1: L1 norm (α increasing: 1 → 2 → ∞)
    l1_chain = [alg_rank, metrics["eRank"], metrics["renyi2_L1"], metrics["nuclear_rank"], 1.0]
    l1_ok = all(l1_chain[i] >= l1_chain[i + 1] - tol for i in range(len(l1_chain) - 1))

    # Chain 2: L2sq norm (α increasing: 1 → 2 → ∞)
    l2_chain = [alg_rank, metrics["vN_eRank"], metrics["participation_ratio"], metrics["stable_rank"], 1.0]
    l2_ok = all(l2_chain[i] >= l2_chain[i + 1] - tol for i in range(len(l2_chain) - 1))

    # Chain 3: Cross-norm (L1 ≥ L2sq at same α)
    cross_ok = (
        metrics["eRank"] >= metrics["vN_eRank"] - tol
        and metrics["renyi2_L1"] >= metrics["participation_ratio"] - tol
        and metrics["nuclear_rank"] >= metrics["stable_rank"] - tol
    )

    metrics["ordering_holds"] = l1_ok and l2_ok and cross_ok
    metrics["l1_chain_holds"] = l1_ok
    metrics["l2_chain_holds"] = l2_ok
    metrics["cross_norm_holds"] = cross_ok

    return metrics


# ──────────────────────────────────────────────────────────────
# Batch-Wise Statistics (Error Bars)
# ──────────────────────────────────────────────────────────────


def compute_metrics_with_uncertainty(
    singular_values_batches: list[torch.Tensor],
    N: Optional[int] = None,
    M: Optional[int] = None,
    alphas: Optional[list[float]] = None,
) -> dict[str, dict[str, float]]:
    """
    Compute all metrics on multiple batches and return mean ± std.

    Args:
        singular_values_batches: list of 1D tensors, one per batch.
        N, M: matrix dimensions (needed for GD threshold).
        alphas: Rényi orders for the grid.

    Returns:
        Dict mapping metric names to {'mean': ..., 'std': ..., 'values': [...]}.
    """
    if alphas is None:
        alphas = STANDARD_ALPHAS

    all_results = []
    for sv in singular_values_batches:
        grid = compute_full_grid(sv, alphas=alphas)
        grid["numerical_rank_99"] = float(numerical_rank(sv, 0.99))
        grid["numerical_rank_95"] = float(numerical_rank(sv, 0.95))
        grid["elbow_rank"] = float(elbow_rank(sv))
        if N is not None and M is not None:
            grid["opt_threshold"] = float(optimal_hard_threshold(sv, N, M))
        grid["sgr_max"] = spectral_gap_ratio(sv, k=None)
        grid["sgr_k3"] = spectral_gap_ratio(sv, k=3)
        # Raw entropies
        grid["H_shannon_L1"] = shannon_entropy(sv, norm="L1")
        grid["S_vN"] = von_neumann_entropy(sv)
        all_results.append(grid)

    # Aggregate
    all_keys = all_results[0].keys()
    stats = {}
    for key in all_keys:
        values = [r[key] for r in all_results]
        arr = np.array(values, dtype=np.float64)
        stats[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "values": values,
        }

    return stats
