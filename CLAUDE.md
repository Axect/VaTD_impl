# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VaTD (Variational Temperature Distribution) is a research implementation for training generative models to sample from the Boltzmann distribution of the 2D Ising model. It uses autoregressive neural networks (PixelCNN) with curriculum learning to generate spin configurations across multiple temperatures and predict partition functions.

The core physics: learn the exact partition function Z(T) for a 16x16 Ising lattice using the Onsager analytical solution for validation.

## Commands

```bash
# Activate environment
source .venv/bin/activate

# Single training run
python main.py --run_config configs/v0.9/ising_pixelcnn_best.yaml

# Training with device override
python main.py --run_config configs/v0.9/ising_pixelcnn_best.yaml --device cuda:0

# Hyperparameter optimization with Optuna
python main.py --run_config configs/v0.9/ising_pixelcnn.yaml \
               --optimize_config configs/v0.9/ising_pixelcnn_tpe.yaml

# Interactive analysis of trained models
python analyze.py

# Analysis scripts
python analyze_critical_temp.py --project MyProject --group my_group --seed 42
python compare_with_reference.py --project MyProject --group my_group --seed 42

# Run tests
python test_exact_partition.py
python test_film_ad.py
```

## Architecture

### Key Files
- `main.py` - Entry point; creates Ising energy function, loads exact partition values, orchestrates training/optimization
- `model.py` - DiscretePixelCNN with masked convolutions, temperature conditioning via FiLM or channel augmentation
- `util.py` - Trainer class with curriculum learning, validation logic, model loading
- `config.py` - RunConfig/OptimizeConfig dataclasses with dynamic class loading via importlib
- `vatd_exact_partition.py` - Onsager solution for exact log Z values (transfer matrix method)
- `pruner.py` - PFLPruner (Predicted Final Loss) for efficient hyperparameter search
- `hyperbolic_lr.py` - Custom learning rate schedulers (ExpHyperbolicLR)
- `analyze.py` - Interactive CLI for loading and testing trained models

### Configuration System
YAML configs in `configs/` define everything: model class paths, optimizer, scheduler, hyperparameters. Classes are loaded dynamically via importlib from string paths like `model.DiscretePixelCNN` or `torch.optim.AdamW`.

### Training Flow
1. Load config → create model/optimizer/scheduler via RunConfig methods
2. Compute exact log Z for validation temperatures using Onsager solution
3. Train with curriculum learning (3 phases):
   - Phase 1 (epochs 0-50): High temperature only (β ∈ [0.2, 0.35]) - easier disordered states
   - Phase 2 (epochs 50-150): Gradual expansion to full range via cosine annealing
   - Phase 3 (epochs 150+): Mixed sampling with 50% focus on critical region (Tc ≈ 2.269)
4. Loss = log_prob + β × energy (Boltzmann loss via REINFORCE)
5. Validate against exact log Z at fixed temperatures

### Model Architecture (DiscretePixelCNN)
- Masked convolutions (Type A/B) for autoregressive generation
- Temperature conditioning options:
  - `DiscretePixelCNN`: Channel augmentation (β concatenated to features)
  - `DiscretePixelCNNFiLM`: FiLM conditioning (Feature-wise Linear Modulation per layer)
- `TemperatureEmbedding`: Sinusoidal encoding with MLP refinement for rich β representation
- Output: binary spin logits (+1/-1)
- `fix_first=1` breaks Z2 symmetry by fixing the first spin

### Masked Convolution Pattern
- **Type A mask**: Cannot see current pixel (used in first layer)
- **Type B mask**: Can see current pixel (used in subsequent layers)
- Autoregressive order: row-major, left-to-right, top-to-bottom

## Key Concepts

- **β (beta)**: Inverse temperature = 1/T; higher β = lower temperature = more ordered
- **Critical temperature**: Tc ≈ 2.269 (βc ≈ 0.44) where phase transition occurs
- **Training range**: β ∈ [0.2, 1.0] (T ∈ [1.0, 5.0])
- **Validation range**: β ∈ [0.1, 2.0] is wider than training to test extrapolation
- **Tc-focused region**: β ∈ [0.38, 0.52] (T ∈ [1.92, 2.63]) around critical temperature
- **Runs directory**: `runs/` stores checkpoints and W&B logs

### WandB Logging Structure
Metrics are organized into sections:
- `train/`: Training loss and log probabilities
- `val/`: Validation metrics per temperature
- `exact/`: Errors vs exact partition function
- `curriculum/`: Current beta range during training
- `diversity/`: Sample diversity metrics

## Config Versioning

Configs are versioned in `configs/v0.0` through `configs/v0.10`. Current best: `configs/v0.9/ising_pixelcnn_best.yaml`
