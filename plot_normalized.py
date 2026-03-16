"""
Normalized eRank and Renyi plots: trained / random baseline.

Removes geometric bias from PixelCNN by dividing by untrained eRank.
"""
import os
os.environ['VATD_NO_MHC'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from vatd_exact_partition import CRITICAL_TEMPERATURE as Tc

# ──────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────

phase0 = pd.read_csv("outputs/null_model_analysis/null_model_erank.csv")
phase2_renyi = pd.read_csv("outputs/phase2b_analysis/renyi_spectrum.csv")
phase2_probe = pd.read_csv("outputs/phase2b_analysis/operator_probes.csv")

figs_dir = Path("outputs/phase2b_analysis/figs")
figs_dir.mkdir(parents=True, exist_ok=True)

RENYI_ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

# ──────────────────────────────────────────────────────────────
# Figure 1: Normalized eRank = trained / random_avg
# ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for arch in ["PixelCNN", "LatticeGPT"]:
    trained = phase0[phase0["model"] == f"{arch} (trained)"]
    random_avg = phase0[phase0["model"] == f"{arch} (random avg)"]

    # Merge on T + layer
    merged = trained.merge(random_avg, on=["T", "layer"], suffixes=("_trained", "_random"))

    # Compute normalized eRank per layer
    merged["norm_erank"] = merged["channel_erank_trained"] / merged["channel_erank_random"]

    layers = sorted(merged["layer"].unique(),
                    key=lambda x: (0, int(x.split("_")[1])) if x.startswith("layer_") else (1, 0))
    n_layers = len(layers)
    layer_colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_layers))

    # Panel 0: All layers per-arch
    ax_idx = 0 if arch == "PixelCNN" else 1
    ax = axes[ax_idx]
    for ci, layer in enumerate(layers):
        ld = merged[merged["layer"] == layer].sort_values("T")
        label = layer.replace("layer_", "Block ").replace("final", "Final")
        ax.plot(ld["T"], ld["norm_erank"],
                "o-", color=layer_colors[ci], markersize=3, linewidth=1.2,
                label=label, alpha=0.85)

    ax.axvline(Tc, color="red", ls="--", alpha=0.5, lw=1, label=f"$T_c$")
    ax.axhline(1, color="gray", ls=":", alpha=0.5, lw=1)
    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel("eRank$_{\\mathrm{trained}}$ / eRank$_{\\mathrm{random}}$", fontsize=12)
    ax.set_title(f"{arch}: Normalized eRank (per layer)", fontsize=13)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.15)

# Panel 2: Block 0 comparison
ax = axes[2]
for arch, color, marker in [("PixelCNN", "#2196F3", "o"), ("LatticeGPT", "#FF9800", "s")]:
    trained = phase0[phase0["model"] == f"{arch} (trained)"]
    random_avg = phase0[phase0["model"] == f"{arch} (random avg)"]
    merged = trained.merge(random_avg, on=["T", "layer"], suffixes=("_trained", "_random"))
    merged["norm_erank"] = merged["channel_erank_trained"] / merged["channel_erank_random"]

    b0 = merged[merged["layer"] == "layer_0"].sort_values("T")
    ax.plot(b0["T"], b0["norm_erank"],
            f"{marker}-", color=color, markersize=4, linewidth=1.8,
            label=f"{arch} Block 0", alpha=0.9)

ax.axvline(Tc, color="red", ls="--", alpha=0.5, lw=1, label=f"$T_c$={Tc:.3f}")
ax.axhline(1, color="gray", ls=":", alpha=0.5, lw=1, label="random baseline")
ax.set_xlabel("Temperature $T$", fontsize=12)
ax.set_ylabel("eRank$_{\\mathrm{trained}}$ / eRank$_{\\mathrm{random}}$", fontsize=12)
ax.set_title("Block 0: Normalized eRank Comparison", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig(figs_dir / "normalized_erank.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {figs_dir / 'normalized_erank.png'}")


# ──────────────────────────────────────────────────────────────
# Figure 2: Normalized Renyi spectrum at Tc
#   R_trained(α) / R_random(α) for each α
# ──────────────────────────────────────────────────────────────

# Phase 0 has renyi columns: channel_renyi_0.5, channel_renyi_1.0, ...
phase0_renyi_alphas = [0.5, 1.0, 2.0, 5.0]  # alphas available in phase0

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for arch_idx, arch in enumerate(["PixelCNN", "LatticeGPT"]):
    ax = axes[arch_idx]
    trained = phase0[(phase0["model"] == f"{arch} (trained)") & (phase0["layer"] == "layer_0")]
    random_avg = phase0[(phase0["model"] == f"{arch} (random avg)") & (phase0["layer"] == "layer_0")]

    # For each T, compute R_trained(α) / R_random(α)
    merged = trained.merge(random_avg, on=["T"], suffixes=("_trained", "_random"))
    merged = merged.sort_values("T")

    for a in phase0_renyi_alphas:
        col_t = f"channel_renyi_{a}_trained"
        col_r = f"channel_renyi_{a}_random"
        if col_t in merged.columns and col_r in merged.columns:
            ratio = merged[col_t] / merged[col_r]
            ax.plot(merged["T"], ratio, "o-", markersize=3, linewidth=1.2,
                    label=f"$\\alpha$={a}", alpha=0.85)

    # Also nuclear rank (renyi_inf)
    col_t = "channel_renyi_inf_trained"
    col_r = "channel_renyi_inf_random"
    if col_t in merged.columns and col_r in merged.columns:
        ratio = merged[col_t] / merged[col_r]
        ax.plot(merged["T"], ratio, "o-", markersize=3, linewidth=1.2,
                label="$\\alpha$=∞", alpha=0.85)

    ax.axvline(Tc, color="red", ls="--", alpha=0.5, lw=1)
    ax.axhline(1, color="gray", ls=":", alpha=0.5, lw=1)
    ax.set_xlabel("Temperature $T$", fontsize=12)
    ax.set_ylabel("$R_{\\mathrm{trained}}(\\alpha) / R_{\\mathrm{random}}(\\alpha)$", fontsize=12)
    ax.set_title(f"{arch}: Normalized Rényi Rank (Block 0)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig(figs_dir / "normalized_renyi_spectrum.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {figs_dir / 'normalized_renyi_spectrum.png'}")


# ──────────────────────────────────────────────────────────────
# Figure 3: At Tc — bar chart of normalized R(α) for both archs
# ──────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5.5))

alpha_list = phase0_renyi_alphas + [float("inf")]
alpha_labels = [str(a) for a in phase0_renyi_alphas] + ["∞"]
x = np.arange(len(alpha_list))
width = 0.35

for arch_idx, (arch, color) in enumerate([("PixelCNN", "#2196F3"), ("LatticeGPT", "#FF9800")]):
    trained = phase0[(phase0["model"] == f"{arch} (trained)") & (phase0["layer"] == "layer_0")]
    random_avg = phase0[(phase0["model"] == f"{arch} (random avg)") & (phase0["layer"] == "layer_0")]

    # Get row closest to Tc
    t_idx = (trained["T"] - Tc).abs().idxmin()
    r_idx = (random_avg["T"] - Tc).abs().idxmin()
    t_row = trained.loc[t_idx]
    r_row = random_avg.loc[r_idx]

    ratios = []
    for a in phase0_renyi_alphas:
        t_val = t_row.get(f"channel_renyi_{a}", np.nan)
        r_val = r_row.get(f"channel_renyi_{a}", np.nan)
        ratios.append(t_val / r_val if r_val > 0 else np.nan)
    # inf
    t_val = t_row.get("channel_renyi_inf", np.nan)
    r_val = r_row.get("channel_renyi_inf", np.nan)
    ratios.append(t_val / r_val if r_val > 0 else np.nan)

    offset = -width/2 if arch_idx == 0 else width/2
    ax.bar(x + offset, ratios, width, label=arch, color=color, alpha=0.8)

ax.axhline(1, color="gray", ls="--", alpha=0.5, lw=1, label="random baseline")
ax.set_xticks(x)
ax.set_xticklabels(alpha_labels)
ax.set_xlabel("Rényi order $\\alpha$", fontsize=12)
ax.set_ylabel("$R_{\\mathrm{trained}}(\\alpha) / R_{\\mathrm{random}}(\\alpha)$ at $T_c$", fontsize=12)
ax.set_title("Normalized Rényi Rank at $T_c$ (Block 0): Trained / Random", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15, axis="y")

plt.tight_layout()
plt.savefig(figs_dir / "normalized_renyi_at_Tc.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {figs_dir / 'normalized_renyi_at_Tc.png'}")


# ──────────────────────────────────────────────────────────────
# Print key numbers at Tc
# ──────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("Key Values at T ≈ Tc (Block 0)")
print("="*60)

for arch in ["PixelCNN", "LatticeGPT"]:
    trained = phase0[(phase0["model"] == f"{arch} (trained)") & (phase0["layer"] == "layer_0")]
    random_avg = phase0[(phase0["model"] == f"{arch} (random avg)") & (phase0["layer"] == "layer_0")]

    t_idx = (trained["T"] - Tc).abs().idxmin()
    r_idx = (random_avg["T"] - Tc).abs().idxmin()
    t_row = trained.loc[t_idx]
    r_row = random_avg.loc[r_idx]

    t_erank = t_row["channel_erank"]
    r_erank = r_row["channel_erank"]
    norm = t_erank / r_erank

    print(f"\n{arch}:")
    print(f"  Trained eRank: {t_erank:.3f}")
    print(f"  Random  eRank: {r_erank:.3f}")
    print(f"  Normalized:    {norm:.3f}")
    print(f"  Renyi spectrum (trained/random):")
    for a in phase0_renyi_alphas:
        tv = t_row.get(f"channel_renyi_{a}", np.nan)
        rv = r_row.get(f"channel_renyi_{a}", np.nan)
        print(f"    R({a}): {tv:.2f}/{rv:.2f} = {tv/rv:.3f}")
    tv = t_row.get("channel_renyi_inf", np.nan)
    rv = r_row.get("channel_renyi_inf", np.nan)
    print(f"    R(∞): {tv:.2f}/{rv:.2f} = {tv/rv:.3f}")
