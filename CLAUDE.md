# VaTD - Variational Thermodynamic Divergence for 2D Ising Model

## Project Overview

Research implementation that learns the Boltzmann distribution of the 2D Ising model via variational free energy minimization. Supports two architectures:
- **DiscretePixelCNN** (autoregressive, main approach) — `model.py`, trained via `util.Trainer`
- **DiscreteFlowMatcher** (parallel sampling) — `model_dfm.py`, trained via `util_dfm.FlowMatchingTrainer`
- **XY model extension** for BKT transition studies

Validates the "Low-rank hypothesis of complex systems" (Nature Physics 2024) using neural networks on the 2D Ising model with emphasis on critical temperature phenomena.

## Tech Stack

- **Language:** Python 3 with PyTorch (CUDA 12.x)
- **Package Manager:** `uv` (use `uv pip install -r requirements.txt`)
- **Experiment Tracking:** Weights & Biases (`wandb`)
- **Config Format:** YAML with typed dataclasses (`config.py`)
- **Hyperparameter Optimization:** Optuna
- **Custom Optimizers:** SPlus (`splus.py`), Muon (`muon.py`)

## Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Training
python main.py --run_config configs/v0.18/<config>.yaml --device cuda:0

# Hyperparameter Optimization
python main.py --run_config <run_config>.yaml --optimize_config <opt_config>.yaml --device cuda:0

# Tests
python test_exact_partition.py

# Analysis (interactive TUI)
python analyze.py
python analyze_rank.py --project <project> --group <group> --seed 42
python analyze_compression.py --project <project> --group <group> --quick
python analyze_critical_temp.py
```

## Project Structure

```
├── main.py              # Entry point: model dispatch, energy function creation
├── model.py             # DiscretePixelCNN: masked convolutions, HC/mHC, scan paths
├── model_dfm.py         # DiscreteFlowMatcher: parallel sampling, Dirichlet paths
├── util.py              # Trainer: REINFORCE, curriculum learning, training modes
├── util_dfm.py          # FlowMatchingTrainer: energy-guided training
├── config.py            # RunConfig, OptimizeConfig dataclasses
├── ising.py             # Energy functions, Swendsen-Wang MCMC, Metropolis-Hastings
├── potts.py             # q-state Potts energy, SW MCMC (q=3,4)
├── vatd_exact_partition.py  # Onsager exact partition function
├── potts_exact_partition.py # Potts Tc, central charges, CFT operator counts
├── partition_estimation.py  # Partition function estimation methods
├── analyze*.py          # Analysis scripts (thermodynamic, low-rank, compression, Tc)
├── analyze_cross_model.py   # Cross-model eRank comparison (PRL Result 1)
├── analyze_entanglement.py  # Neural entanglement entropy (PRL Result 2)
├── analyze_rank_exponent.py # Rank-exponent power law (PRL Result 3)
├── analyze_prl_paper.py     # Unified PRL publication figures
├── splus.py / muon.py   # Custom optimizers
├── hyperbolic_lr.py     # ExpHyperbolicLR scheduler
├── pruner.py            # PFLPruner for Optuna
├── configs/             # 68 YAML configs across v0.0–v0.19
├── docs/                # Research documentation
├── runs/                # Saved models and results (gitignored)
├── figs/                # Generated figures (gitignored)
└── checkpoints/         # Training checkpoints (gitignored)
```

## Architecture & Key Patterns

- **Triple model routing:** `main.py` detects model type via `is_dfm_model()`, `is_xy_model()`, `is_potts_model()` and dispatches to the appropriate trainer/energy function
- **Masked convolutions:** TypeA/TypeB for autoregressive pixel prediction
- **Hyper-Connections (v0.16+):** HCLayer (unconstrained), mHCLayer (doubly-stochastic via Sinkhorn-Knopp), with optional CUDA kernel (`mhc` package) or pure PyTorch fallback
- **Scan path variants:** Row-major, diagonal (8x speedup), Hilbert curve
- **Training modes:** `standard`, `accumulated` (gradient accumulation), `sequential` (per-temperature), `energy_guided` (DFM)
- **Curriculum learning:** Phase 1 high-T → Phase 2 full range
- **Temperature-dependent logit scaling:** `scale = (T_ref / T)^power`
- **REINFORCE with RLOO baseline** for gradient estimation

## Physics Constants

- Lattice: 16×16 with periodic boundary conditions
- Interaction: J = 1
- **Ising (q=2):** Tc = 2/ln(1+√2) ≈ 2.269, c = 1/2, 3 relevant operators
- **3-Potts (q=3):** Tc = 1/ln(1+√3) ≈ 0.995, c = 4/5, 6 relevant operators
- **4-Potts (q=4):** Tc = 1/ln(3) ≈ 0.910, c = 1, 8 relevant operators
- Ising temperature range: T ∈ [1, 5] (β ∈ [0.2, 1.0])
- Potts temperature ranges configured per model in `configs/v0.19/`
- First spin fixed for symmetry breaking (Ising: +1, Potts: state 0)

## Current Research: Low-Rank Hypothesis Verification

Validating the **"Low-rank hypothesis of complex systems"** (Thibeault et al., Nature Physics 2024) through two complementary experiments on trained DiscretePixelCNN models. Full documentation in `docs/LOW_RANK.md` and `docs/LOW_RANK_COMPANION.md` (Korean).

### Core Hypothesis

Complex systems are effectively low-rank — their dynamics operators have rapidly decaying singular value spectra. In the 2D Ising model, this manifests as:
- **Activation rank minimum at Tc**: The neural network's internal representations become low-dimensional at the critical temperature, where the system is dominated by a few scaling fields (magnetization, energy density)
- **Compression sensitivity maximum at Tc**: SVD-truncating model weights causes the most degradation at Tc, where precision is most needed to maintain the critical manifold
- **d(eRank)/dT correlates with Cv**: The temperature derivative of effective rank mirrors the specific heat peak

This creates a **"compression paradox"**: activations are simplest (low-rank) at Tc, yet the model weights must be most precise there — like a pencil balanced on its tip (1D state requiring extreme precision).

### Experiment 1: Activation Rank Analysis (`analyze_rank.py`)

Measures the effective rank of intermediate feature maps across the phase transition:
1. Generate samples at each temperature via autoregressive sampling
2. Collect activations from each residual block via forward pre-hooks
3. Compute SVD on channel-averaged (`[N,C]`) and spatial-averaged (`[N,HW]`) activation matrices
4. Track 9 rank metrics across temperature

**Rank metrics implemented** (all in `analyze_rank.py`):
| Metric | Definition | Character |
|--------|-----------|-----------|
| Effective Rank (eRank) | exp(Shannon entropy of normalized SVs) | Continuous, [1, min(m,n)] |
| Stable Rank | ||A||_F² / ||A||_2² | Noise-robust |
| Participation Ratio | (Σσ²)² / Σσ⁴ | Condensed matter origin |
| Numerical Rank (99%/95%) | Min k for threshold% energy | Discrete (integer) |
| Rényi Rank (α=2) | exp(Rényi entropy) | Sensitive to dominant modes |
| Nuclear Rank | Σσ / σ_max (from Nature Physics 2024) | L1/L∞ ratio |
| Elbow Rank | Max distance from diagonal in scree plot | Geometric method |
| Optimal Hard Threshold | Gavish & Donoho (2014) RMT-based | Statistical signal/noise boundary |

**Key findings** (from v0.16 Compact Dilated ResNet, L=16):
- Channel eRank drops from ~25–30 (high T) to ~3–5 (Tc) — consistent with 2 relevant RG operators + corrections
- Dip deepest in early layers (coarse-graining UV degrees of freedom, mirrors RG flow)
- d(eRank)/dT peak at T≈2.5, shifted above Tc due to finite-size effects (L=16)

**Output**: `rank_vs_temperature.png`, `singular_value_spectra.png`, `derank_dt_vs_Cv.png`, `extended_rank_metrics.png` + CSV

### Experiment 2: Weight Compression Test (`analyze_compression.py`)

Tests whether SVD-truncating model weights causes maximum degradation at Tc:
1. Generate reference samples from full model at each T
2. For each rank fraction k ∈ {90%, 75%, 50%, 25%, 10%}: truncate all Conv2d weights via SVD
3. Measure degradation D(T,k) = KL(q_full || q_k) / N ≥ 0
4. Optional per-block sensitivity: truncate one block at a time (50% rank)

**Key findings**:
- D(T,k) peaks sharply near Tc for aggressive truncation (k=10%: D≈0.07 nats/site at Tc, ≈0 elsewhere)
- k≥50% preserves the distribution everywhere — half the SVs suffice
- D(T,k) ∝ Cv(T) near Tc — compression sensitivity inherits specific heat scaling (both tied to susceptibility divergence)
- Weight SVD spectra are near-full-rank (~88% eRank/full) and temperature-independent — weights encode static physics rules, activations encode dynamic state

**Output**: `compression_test.png`, `weight_svd_spectra.png`, optionally `compression_per_block.png` + CSV

### Experiment 3: Critical Temperature Analysis (`analyze_critical_temp.py`)

Evaluates model accuracy around Tc by comparing model loss against Onsager exact solution:
- Temperature range: [0.7Tc, 1.3Tc] with 20 points
- Measures: loss, exact_logZ, error, normalized error
- Identifies best/worst accuracy temperatures relative to Tc

### Analysis Modes

```bash
# Experiment 1: Activation rank (interactive or CLI)
python analyze_rank.py
python analyze_rank.py --project Ising_VaTD_v0.16 --group <group> --seed 42 --device cuda:0
python analyze_rank.py --critical  # Dense 60-point grid around Tc for smooth d(eRank)/dT
python analyze_rank.py --replot <csv_path>  # Regenerate plots from saved CSV

# Experiment 2: Compression test
python analyze_compression.py --project Ising_VaTD_v0.16 --group <group> --seed 42 --device cuda:0
python analyze_compression.py --per_layer  # Add per-block sensitivity analysis
python analyze_compression.py --replot <csv_path>

# Experiment 3: Critical temperature
python analyze_critical_temp.py --project <proj> --group <group> --seed 42
```

### Current Development Status (v0.18–v0.19)

- **v0.18**: A100 optimization (batch 2048, 262K samples/epoch)
  - `ising_pixelcnn_beta_hc.yaml`: Beta-conditioned HC
  - `ising_pixelcnn_toroidal.yaml`: Circular padding + unconstrained HC
- **v0.19**: q-state Potts model support for PRL paper
  - `potts3_pixelcnn.yaml`: 3-state Potts (q=3, Tc≈0.995)
  - `potts4_pixelcnn.yaml`: 4-state Potts (q=4, Tc≈0.910)
  - `model.py` now accepts `category` param (2=Ising, q=Potts)

The `runs/` and `figs/` directories (gitignored) store trained checkpoints and analysis outputs.

### PRL Paper: "Autoregressive neural networks discover the operator content of CFTs"

Three results combined into one PRL paper:

1. **Result 1 — Multi-model operator counting** (`analyze_cross_model.py`):
   eRank_min at Tc counts CFT relevant operators (3 for Ising, 6 for 3-Potts, 8 for 4-Potts)

2. **Result 2 — Neural entanglement entropy** (`analyze_entanglement.py`):
   Cross-covariance SVD entropy follows Calabrese-Cardy → extracts central charge c

3. **Result 3 — Rank-exponent duality** (`analyze_rank_exponent.py`):
   eRank ~ |T-Tc|^{-φ} universal power law near Tc

Publication figures generated by `analyze_prl_paper.py`.

### References

1. Thibeault, V. et al., "The low-rank hypothesis of complex systems," *Nature Physics* **20**, 294–302 (2024)
2. Roy, O. & Vetterli, M., "The effective rank," *EUSIPCO* (2007)
3. Gavish, M. & Donoho, D., "The optimal hard threshold for singular values is 4/√3," *IEEE Trans. Inf. Theory* (2014)
4. Calabrese, P. & Cardy, J., "Entanglement entropy and quantum field theory," *J. Stat. Mech.* (2004)
5. Di Francesco, P. et al., "Conformal Field Theory," Springer (1997)

## Conventions

- Config versions in `configs/v0.X/` directories
- Snake_case for config fields and variables
- Leading underscore for module-level private imports
- Float32 for computation, float64 for exact partition function
- `VATD_NO_MHC=1` env var disables custom CUDA mHC kernel (A100 compatibility)
- Checkpoint format: `{project}/{group_name}/checkpoints/epoch_{epoch}.pt`
