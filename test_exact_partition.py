#!/usr/bin/env python3
"""Test exact partition function against reference implementation"""

import torch
import sys
import os

# Add reference implementation path
sys.path.append("/home/xteca/zbin/vatd")

from vatd_exact_partition import logZ as our_logZ, CRITICAL_TEMPERATURE
from utils import isingLogzTr as ref_logZ


def test_exact_partition():
    """Compare our implementation against reference"""

    print("Testing exact partition function implementation")
    print("=" * 80)
    print(f"Critical temperature (Onsager): Tc = {CRITICAL_TEMPERATURE:.6f}")
    print("=" * 80)

    test_cases = [
        (4, 1.0, 0.5),  # Small lattice, high temp
        (8, 1.0, 0.5),  # Medium lattice
        (16, 1.0, 0.5),  # Standard lattice, high temp
        (16, 1.0, 0.3),  # High temp
        (16, 1.0, 1.0),  # Medium temp
        (16, 1.0, 1.5),  # Low temp
        (16, 1.0, 2.0),  # Very low temperature
    ]

    print(
        f"\n{'L':>3} {'beta':>6} {'Our logZ':>14} {'Ref logZ':>14} {'Abs Diff':>12} {'Rel Diff':>12} {'Status':>8}"
    )
    print("-" * 80)

    all_pass = True
    for L, j, beta in test_cases:
        our_result = our_logZ(
            n=L, j=j, beta=torch.tensor(beta, dtype=torch.float64)
        ).item()
        ref_result = ref_logZ(
            n=L, j=j, beta=torch.tensor(beta, dtype=torch.float64)
        ).item()

        diff = abs(our_result - ref_result)
        rel_diff = diff / abs(ref_result) if ref_result != 0 else 0

        status = "✓ PASS" if diff < 1e-5 else "✗ FAIL"
        if diff >= 1e-5:
            all_pass = False

        print(
            f"{L:>3} {beta:>6.3f} {our_result:>14.6f} {ref_result:>14.6f} "
            f"{diff:>12.2e} {rel_diff:>12.2e} {status:>8}"
        )

    print("=" * 80)
    if all_pass:
        print("✓ All tests passed!")
        print("The exact partition function implementation matches the reference.")
    else:
        print("✗ Some tests failed!")
        print("There may be numerical precision issues. Review the implementation.")

    return all_pass


if __name__ == "__main__":
    success = test_exact_partition()
    sys.exit(0 if success else 1)
