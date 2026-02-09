# VaTD: Variational Thermodynamic Divergence for 2D Ising Model

Learning the Boltzmann distribution of the 2D Ising model via variational free energy minimization.
Supports autoregressive PixelCNN and Discrete Flow Matching architectures, validated against Onsager's exact partition function solution.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
wandb login  # optional, for W&B experiment tracking
```

## Training

All training runs are launched through `main.py` with a YAML config file.
The model type is auto-detected from the `net` field and routed to the appropriate trainer.

### PixelCNN (v0.16, Compact + Experimental Variants)

```bash
# Compact: 4-layer dilated, multi-scale skip connections, ~46% fewer params than v0.15
python main.py --run_config configs/v0.16/ising_pixelcnn_compact.yaml --device cuda:0

# Hyper-Connection variants (HC / mHC fusion between residual blocks)
python main.py --run_config configs/v0.16/ising_pixelcnn_hc.yaml --device cuda:0
python main.py --run_config configs/v0.16/ising_pixelcnn_mhc.yaml --device cuda:0

# Alternative scan paths (diagonal: ~8x sampling speedup, hilbert: 2D locality)
python main.py --run_config configs/v0.16/ising_pixelcnn_diagonal.yaml --device cuda:0
python main.py --run_config configs/v0.16/ising_pixelcnn_hilbert.yaml --device cuda:0

# Muon optimizer (Newton-Schulz orthogonalization for 2D weights)
python main.py --run_config configs/v0.16/ising_pixelcnn_muon.yaml --device cuda:0
```

### PixelCNN (v0.15, Dilated Convolutions)

```bash
# Dilated only (pure REINFORCE, no MCMC)
python main.py --run_config configs/v0.15/ising_pixelcnn_dilated_only.yaml --device cuda:0

# Dilated + MCMC guidance (Swendsen-Wang)
python main.py --run_config configs/v0.15/ising_pixelcnn_improved.yaml --device cuda:0
```

### PixelCNN (v0.13, Sequential Training)

```bash
python main.py --run_config configs/v0.13/ising_pixelcnn_onecycle.yaml --device cuda:0
```

### Discrete Flow Matching (v0.14)

```bash
python main.py --run_config configs/v0.14/ising_dfm.yaml --device cuda:0
```

### Hyperparameter Optimization

```bash
python main.py --run_config configs/v0.13/ising_pixelcnn_onecycle.yaml \
               --optimize_config configs/optimize_template.yaml --device cuda:0
```

## Analysis

### Thermodynamic Validation

Compare learned free energy against Onsager's exact solution across the temperature range.

```bash
python analyze.py
```

### Critical Temperature Analysis

Estimate Tc from specific heat capacity peaks.

```bash
python analyze_critical_temp.py --project Ising_VaTD_v0.15 \
    --group <group_name> --seed 42
```

---

## Low-Rank Hypothesis Experiments

Experimental verification of the "Low-rank hypothesis of complex systems" (Nature Physics, 2024) using neural networks trained on the 2D Ising model.
See [`docs/LOW_RANK.md`](docs/LOW_RANK.md) for background.

### Experiment 1: Internal Representation Rank (`analyze_rank.py`)

Measures the effective rank of internal feature maps as a function of temperature.
Tests whether activation rank peaks near the critical temperature Tc.

```bash
# Interactive mode
python analyze_rank.py

# CLI mode
python analyze_rank.py \
    --project Ising_VaTD_v0.15 \
    --group DiscretePixelCNN_lr1e-3_e500_f2d43d \
    --seed 42 --device cuda:0

# Quick mode (fewer temperatures, smaller batches)
python analyze_rank.py \
    --project Ising_VaTD_v0.15 \
    --group DiscretePixelCNN_lr1e-3_e500_f2d43d \
    --seed 42 --device cuda:0 --quick

# Replot from existing CSV (no GPU needed)
python analyze_rank.py \
    --replot runs/Ising_VaTD_v0.15/DiscretePixelCNN_lr1e-3_e500_f2d43d/rank_analysis_42.csv
```

**Output:**
- `figs/<group>/rank_vs_temperature.png` — 2x2 figure (channel/spatial eRank, eRank vs Cv, heatmap)
- `figs/<group>/singular_value_spectra.png` — Scree plots at 3 representative temperatures
- `runs/<project>/<group>/rank_analysis_<seed>.csv`

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | 200 | Samples per temperature |
| `--n_batches` | 3 | Batches per temperature |
| `--quick` | off | Reduced grid (20 temps, batch=100) |
| `--replot <csv>` | - | Regenerate plots from existing data |

### Experiment 2: Low-Rank Compression Test (`analyze_compression.py`)

Applies SVD truncation to model weights and measures per-temperature free energy degradation.
Identifies which temperature region is most sensitive to weight compression.

```bash
# Interactive mode
python analyze_compression.py

# CLI mode (full, ~20 min on RTX 3090)
python analyze_compression.py \
    --project Ising_VaTD_v0.15 \
    --group DiscretePixelCNN_lr1e-3_e500_f2d43d \
    --seed 42 --device cuda:0

# Quick mode (~5 min)
python analyze_compression.py \
    --project Ising_VaTD_v0.15 \
    --group DiscretePixelCNN_lr1e-3_e500_f2d43d \
    --seed 42 --device cuda:0 --quick

# With per-block sensitivity analysis
python analyze_compression.py \
    --project Ising_VaTD_v0.15 \
    --group DiscretePixelCNN_lr1e-3_e500_f2d43d \
    --seed 42 --device cuda:0 --per_layer

# Replot from existing CSV
python analyze_compression.py \
    --replot runs/Ising_VaTD_v0.15/DiscretePixelCNN_lr1e-3_e500_f2d43d/compression_42.csv
```

**Output:**
- `figs/<group>/weight_svd_spectra.png` — Weight SVD decay per residual block
- `figs/<group>/compression_test.png` — 2x2 figure (degradation curves, heatmap, Cv comparison, relative degradation)
- `figs/<group>/compression_per_block.png` — Per-block sensitivity heatmap (with `--per_layer`)
- `runs/<project>/<group>/compression_<seed>.csv`
- `runs/<project>/<group>/compression_per_block_<seed>.csv` (with `--per_layer`)

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | 200 | Samples per temperature |
| `--quick` | off | Reduced grid (20 temps, batch=100, 3 rank fracs) |
| `--per_layer` | off | Per-block sensitivity analysis (rank=50%) |
| `--replot <csv>` | - | Regenerate plots from existing data |

**Rank fractions tested:** 90%, 75%, 50%, 25%, 10% (quick: 75%, 50%, 25%)

### Experiment 3: ERF Rank (planned)

Verify correlation length divergence directly via Jacobian SVD of the Effective Receptive Field.

## Project Structure

```
.
├── main.py                  # Entry point: energy function, model dispatch
├── model.py                 # DiscretePixelCNN (masked conv, HC/mHC, diagonal/hilbert)
├── model_dfm.py             # DiscreteFlowMatcher (parallel sampling)
├── util.py                  # Trainer class (standard/accumulated/sequential modes)
├── util_dfm.py              # FlowMatchingTrainer class
├── config.py                # RunConfig, OptimizeConfig (YAML → importlib instantiation)
├── vatd_exact_partition.py  # Onsager exact partition function (16×16)
├── analyze.py               # Interactive thermodynamic analysis
├── analyze_rank.py          # Exp 1: Activation rank vs temperature
├── analyze_compression.py   # Exp 2: Weight SVD compression test
├── analyze_critical_temp.py # Critical temperature estimation
├── hyperbolic_lr.py         # ExpHyperbolicLR scheduler
├── splus.py                 # SPlus optimizer (matrix preconditioning)
├── muon.py                  # MuonWithAdamW optimizer (Newton-Schulz)
├── pruner.py                # PFLPruner for Optuna
├── mHC.cu/                  # MHCLayer submodule (doubly stochastic hyper-connections)
├── configs/                 # YAML configs (versioned: v0.13 ~ v0.16)
├── docs/                    # Research documents
│   ├── LOW_RANK.md          # Low-Rank Hypothesis specification
│   ├── path_generation_methods.md
│   └── ...
├── runs/                    # Saved models, configs, CSV results
└── figs/                    # Generated analysis figures
```

## References

- VaTD: Variational Thermodynamic Divergence
- Onsager (1944): Exact solution of the 2D Ising model
- Low-Rank Hypothesis: Nature Physics (2024), "Low-rank hypothesis of complex systems"
