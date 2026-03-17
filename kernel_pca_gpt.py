"""
Kernel PCA eRank analysis for LatticeGPT (Ising v0.21).

Computes SVD eRank and kernel (RBF) eRank from intermediate activations
of a trained LatticeGPT model across 25 temperatures from T=1.0 to T=5.0.

Usage:
    VATD_NO_MHC=1 .venv/bin/python kernel_pca_gpt.py
"""

import os
os.environ['VATD_NO_MHC'] = '1'

import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import KernelPCA

from util import load_model
from vatd_exact_partition import CRITICAL_TEMPERATURE

# ── Configuration ──
PROJECT = "Ising_VaTD_v0.21"
GROUP = "LatticeGPT_lr2e-1_e300_6b6541"
SEED = "42"
DEVICE = "cuda:0"
N_SAMPLES = 200
OUTPUT_CSV = "outputs/kernel_pca_gpt_results.csv"
Tc = CRITICAL_TEMPERATURE  # ~2.269


# ── Temperature grid: 25 points, denser near Tc ──
def make_temperature_grid():
    """25 temperatures from 1.0 to 5.0, denser near Tc."""
    # 12 coarse log-spaced covering full range
    T_coarse = np.logspace(np.log10(1.0), np.log10(5.0), 12)
    # 15 dense linear around Tc
    T_dense = np.linspace(0.85 * Tc, 1.15 * Tc, 15)
    T_all = np.unique(np.concatenate([T_coarse, T_dense]))
    T_all.sort()
    # Trim to ~25
    if len(T_all) > 25:
        # Subsample coarse region while keeping dense region intact
        mask_dense = (T_all >= 0.85 * Tc) & (T_all <= 1.15 * Tc)
        T_dense_pts = T_all[mask_dense]
        T_coarse_pts = T_all[~mask_dense]
        n_coarse_keep = 25 - len(T_dense_pts)
        if n_coarse_keep > 0 and len(T_coarse_pts) > n_coarse_keep:
            idx = np.linspace(0, len(T_coarse_pts) - 1, n_coarse_keep, dtype=int)
            T_coarse_pts = T_coarse_pts[idx]
        T_all = np.sort(np.concatenate([T_coarse_pts, T_dense_pts]))
    return T_all


# ── Activation collection (adapted from analyze_rank.py) ──
def collect_activations(model, samples, T_val, device):
    """
    Collect activations from each transformer block and final_norm.

    Returns dict: {0: [N, C, H, W], 1: ..., 'final': [N, C, H, W]}
    """
    captured = {}
    hooks = []
    H, W = model.size

    for i, block in enumerate(model.backbone.blocks):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                act = out[0] if isinstance(out, tuple) else out
                act = act.detach().cpu()
                B, L, D = act.shape
                captured[idx] = act.permute(0, 2, 1).reshape(B, D, H, W)
            return hook_fn
        hooks.append(block.register_forward_hook(make_hook(i)))

    # Final: after backbone.final_norm
    def final_hook(module, inp, out):
        act = out.detach().cpu()
        B, L, D = act.shape
        captured["final"] = act.permute(0, 2, 1).reshape(B, D, H, W)
    hooks.append(model.backbone.final_norm.register_forward_hook(final_hook))

    T_tensor = torch.full((samples.shape[0],), T_val, device=device)
    with torch.no_grad():
        model.log_prob(samples, T=T_tensor)

    for h in hooks:
        h.remove()

    return captured


# ── Rank metrics ──
def svd_effective_rank(singular_values):
    """Effective rank via Shannon entropy of normalized SVs."""
    s = singular_values
    if isinstance(s, torch.Tensor):
        s = s.numpy()
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def median_heuristic(X):
    """Median heuristic for RBF kernel bandwidth."""
    dists = pairwise_distances(X, metric='euclidean')
    triu_idx = np.triu_indices_from(dists, k=1)
    median_dist = np.median(dists[triu_idx])
    if median_dist < 1e-10:
        median_dist = 1.0
    sigma = median_dist
    gamma = 1.0 / (2.0 * sigma ** 2)
    return gamma, sigma


def kernel_effective_rank(eigenvalues):
    """Effective rank from kernel PCA eigenvalues."""
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) == 0:
        return 1.0
    p = ev / ev.sum()
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def compute_kernel_erank_rbf(X, gamma=None):
    """
    Compute kernel (RBF) effective rank.

    Uses KernelPCA with ARPACK solver; falls back to manual eigendecomposition.
    """
    N = X.shape[0]

    if gamma is None:
        gamma, sigma = median_heuristic(X)
    else:
        sigma = np.sqrt(1.0 / (2.0 * gamma))

    n_comp = min(N - 1, X.shape[1], 199)

    try:
        kpca = KernelPCA(
            kernel='rbf',
            n_components=n_comp,
            gamma=gamma,
            fit_inverse_transform=False,
            eigen_solver='arpack',
        )
        kpca.fit(X)
        eigenvalues = kpca.eigenvalues_
    except Exception:
        from sklearn.metrics.pairwise import pairwise_kernels
        K = pairwise_kernels(X, metric='rbf', gamma=gamma)
        N_k = K.shape[0]
        one_n = np.ones((N_k, N_k)) / N_k
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        eigenvalues = np.linalg.eigvalsh(K_centered)[::-1]

    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    return kernel_effective_rank(eigenvalues), sigma


# ── Main ──
def main():
    print(f"Loading model: {PROJECT}/{GROUP}/{SEED}")
    model, config = load_model(PROJECT, GROUP, SEED)
    model = model.to(DEVICE)
    model.eval()

    H, W = model.size
    n_layers = len(model.backbone.blocks)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Lattice: {H}x{W}, Params: {n_params:,}, Layers: {n_layers}")
    print(f"Tc = {Tc:.4f}")

    temperatures = make_temperature_grid()
    print(f"Temperature grid: {len(temperatures)} points, "
          f"T in [{temperatures.min():.3f}, {temperatures.max():.3f}]")

    layer_keys = list(range(n_layers)) + ["final"]
    records = []

    for ti, T_val in enumerate(temperatures):
        beta_val = 1.0 / T_val
        print(f"[{ti+1}/{len(temperatures)}] T={T_val:.4f} (beta={beta_val:.4f})")

        # Generate samples
        T_tensor = torch.full((N_SAMPLES,), T_val, device=DEVICE)
        with torch.no_grad():
            samples = model.sample(batch_size=N_SAMPLES, T=T_tensor)

        # Collect activations
        activations = collect_activations(model, samples, T_val, DEVICE)

        for lk in layer_keys:
            if lk not in activations:
                continue

            act = activations[lk]  # [N, C, H, W]
            N, C, Hact, Wact = act.shape

            # Channel representation: [N, C] via spatial averaging
            act_ch = act.mean(dim=(-2, -1))  # [N, C]
            act_ch = act_ch - act_ch.mean(dim=0, keepdim=True)  # center

            # SVD eRank
            _, S_ch, _ = torch.linalg.svd(act_ch, full_matrices=False)
            svd_er = svd_effective_rank(S_ch.numpy())

            # Kernel (RBF) eRank
            X_ch = act_ch.numpy().astype(np.float64)
            kernel_er, rbf_sigma = compute_kernel_erank_rbf(X_ch)

            layer_name = f"layer_{lk}" if isinstance(lk, int) else lk

            records.append({
                "T": T_val,
                "beta": beta_val,
                "T_over_Tc": T_val / Tc,
                "layer": layer_name,
                "svd_erank": svd_er,
                "kernel_erank": kernel_er,
                "rbf_sigma": rbf_sigma,
            })

            print(f"  {layer_name}: svd_erank={svd_er:.2f}, kernel_erank={kernel_er:.2f}")

    # Save results
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")
    print(f"Total rows: {len(df)}")

    # Summary at Tc
    T_nearest = df.iloc[(df["T"] - Tc).abs().argsort().iloc[0]]["T"]
    print(f"\nSummary at T={T_nearest:.4f} (nearest Tc):")
    near_df = df[df["T"] == T_nearest]
    for _, row in near_df.iterrows():
        print(f"  {row['layer']}: svd={row['svd_erank']:.2f}, kernel={row['kernel_erank']:.2f}")


if __name__ == "__main__":
    main()
