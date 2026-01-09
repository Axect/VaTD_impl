# v0.15 Technical Report: Improved PixelCNN with Dilated Convolutions

**Date:** December 30, 2025
**Project:** VaTD Implementation (Ising Model)
**Version:** v0.15 (Dilated Architecture - Pure REINFORCE)

---

## 1. Executive Summary

Version 0.15 introduces a major architectural improvement to the `DiscretePixelCNN` aimed at resolving performance degradation near the **Critical Temperature ($T_c \approx 2.27$)** of the 2D Ising Model.

Prior iterations (v0.0 - v0.14) suffered from **Critical Slowing Down** and limited receptive fields due to standard convolutions. v0.15 introduces **Dilated Convolutions** to exponentially expand the receptive field, enabling the model to capture long-range correlations essential for critical point learning.

**Key Design Choice**: This version uses **pure REINFORCE (Self-Training)** without external MCMC guidance, testing whether architectural improvements alone can achieve accurate Boltzmann distribution learning.

---

## 2. Theoretical Background & Problem Analysis

### 2.1. The Critical Point Challenge
At the critical temperature $T_c$, the 2D Ising model undergoes a phase transition. The **correlation length ($\xi$)** diverges:
$$
\xi \propto |T - T_c|^{-\nu}
$$
This implies that spins are correlated across the *entire* lattice.

**Limitations of Standard PixelCNN:**

1.  **Linear Receptive Field:** A standard CNN with depth $L$ and kernel $K$ has a receptive field (RF) of $R \approx L \times K$. For a shallow network, $R < \text{Lattice Size}$. The model simply *cannot see* the long-range correlations required to define the critical state.
2.  **Causal Blindness:** PixelCNN generates pixels sequentially (Raster Scan). It captures local correlations well ($s_i$ depends on neighbors) but struggles to capture global geometric structures (fractal clusters) because the "future" pixels are invisible to "past" pixels.
3.  **Gradient Variance:** Training via REINFORCE on the Variational Free Energy $F = \mathbb{E}_q [\log q + \beta E]$ suffers from exploding variance at $T_c$ because the energy fluctuations (Specific Heat $C_v$) diverge.

---

## 3. Architectural Improvement: Dilated Convolutions

To address the limited receptive field, we replaced the standard residual blocks with **Dilated Residual Blocks** (inspired by WaveNet and PixelCNN++).

### 3.1. Mechanism
In a dilated convolution, the filter is applied over an area larger than its length by skipping input values with a certain step (dilation factor $d$).

*   **Standard Convolution ($d=1$):**
    *   Input: `x[0], x[1], x[2]`
*   **Dilated Convolution ($d=2$):**
    *   Input: `x[0], x[2], x[4]`

### 3.2. Exponential Receptive Field Expansion
We stack layers with exponentially increasing dilation factors: $d_i = 2^i$.

*   **Schedule:** $1, 2, 4, 8, 1, 2, 4, 8$ (Total 8 hidden layers)
*   **Kernel Size:** $k=3$

The Receptive Field size $R$ grows exponentially:
$$ R \approx 2^{L+1} $$

**Comparison:**
| Architecture | Depth | Kernel | Receptive Field | Coverage on 16x16 |
| :--- | :---: | :---: | :---: | :---: |
| v0.14 (Standard) | 5 | 3 | $11 \times 11$ | **Partial** (< 50% area) |
| **v0.15 (Dilated)** | **8** | **3** | **$61 \times 61$** | **Full** (> 300% area) |

This ensures that every generated spin $s_i$ has access to context from the **entire lattice boundaries**, allowing the model to make decisions based on the global phase (ordered vs. disordered).

### 3.3. Implementation

```python
# model.py - Dilated Residual Block
class DilatedResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )

    def forward(self, x):
        return x + self.conv(F.gelu(x))

# Dilation schedule: exponentially increasing
dilations = [1, 2, 4, 8, 1, 2, 4, 8]  # 8 layers total
```

---

## 4. Training Strategy: Pure REINFORCE

### 4.1. Self-Training Approach

v0.15 uses **pure variational inference** without external MCMC supervision:

$$
\mathcal{L} = \mathbb{E}_{x \sim q_\theta}[\log q_\theta(x|T) + \beta E(x)]
$$

This is minimized using the **REINFORCE gradient estimator**:

$$
\nabla_\theta \mathcal{L} = \mathbb{E}_{x \sim q_\theta}[\nabla_\theta \log q_\theta(x|T) \cdot (\log q_\theta(x|T) + \beta E(x) - b)]
$$

where $b$ is a baseline for variance reduction.

### 4.2. RLOO Baseline (Leave-One-Out)

To reduce gradient variance, we use the **RLOO baseline**:

$$
b_i = \frac{1}{N-1} \sum_{j \neq i} (\log q_\theta(x_j|T) + \beta E(x_j))
$$

This provides a sample-dependent baseline that significantly reduces variance compared to a simple mean baseline.

### 4.3. Why No MCMC?

| Approach | Pros | Cons |
|----------|------|------|
| MCMC-Guided | Stable supervision, correct targets | Requires external simulator, slow at $T_c$ |
| **Pure REINFORCE** | Self-contained, no external dependencies | Higher variance, potential mode collapse |

v0.15 tests whether **architectural improvements alone** (dilated convolutions with full receptive field) can overcome the limitations of pure self-training.

---

## 5. Curriculum Learning

### 5.1. 2-Phase Temperature Curriculum

To stabilize training, we gradually expand the temperature range:

```
            β_max
              │
       1.0 ───┤ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┬───────────
              │                               /│
              │                              / │
              │                             /  │
              │                            /   │
       0.35 ──┤────────────────────────────    │  Phase 2
              │         Phase 1           │    │  (cosine expansion)
              │      (high temp only)     │    │
       0.2 ───┼────────────────────────────────┼───────────
              │                           │    │
              0          100             250  500    epochs
                      phase1_epochs    +phase2_epochs
```

**Phase 1 (Epochs 0-100):**
- Train only on high temperatures ($T > T_c$)
- Model learns basic spin correlations in disordered phase
- $\beta_{max} = 0.35$ ($T_{min} \approx 2.86 > T_c$)

**Phase 2 (Epochs 100-250):**
- Cosine annealing expansion to full temperature range
- Model gradually encounters critical region and ordered phase
- $\beta_{max}$ increases from 0.35 to 1.0

---

## 6. Implementation Specifications

### `configs/v0.15/ising_pixelcnn_dilated_only.yaml`

| Parameter | Value | Description |
| :--- | :---: | :--- |
| `hidden_conv_layers` | **8** | Enables 4-step dilation cycle (1,2,4,8) twice |
| `hidden_channels` | **96** | Width of residual blocks |
| `kernel_size` | **7** | First convolution kernel size |
| `training_mode` | `sequential` | Sequential backward per temperature |
| `mcmc_enabled` | **false** | No MCMC guidance (pure REINFORCE) |
| `curriculum_enabled` | **true** | 2-phase temperature curriculum |
| `phase1_epochs` | **100** | High-temperature only phase |
| `phase1_beta_max` | **0.35** | $T_{min} \approx 2.86$ in Phase 1 |
| `phase2_epochs` | **150** | Gradual expansion phase |

### Full Configuration

```yaml
project: Ising_VaTD_v0.15
net: model.DiscretePixelCNN
optimizer: torch.optim.AdamW
scheduler: hyperbolic_lr.ExpHyperbolicLR
epochs: 500
batch_size: 512

net_config:
  size: 16
  fix_first: 1

  batch_size: 256
  num_beta: 8
  beta_min: 0.2
  beta_max: 1.0

  training_mode: sequential
  accumulation_steps: 64

  # MCMC Disabled
  mcmc_enabled: false

  # Curriculum
  curriculum_enabled: true
  phase1_epochs: 100
  phase1_beta_max: 0.35
  phase2_epochs: 150

  # Dilated Architecture
  kernel_size: 7
  hidden_channels: 96
  hidden_conv_layers: 8
  hidden_kernel_size: 3
  hidden_width: 128
  hidden_fc_layers: 2

  # Temperature Scaling
  logit_temp_scale: true
  temp_ref: 2.27
  temp_scale_power: 1.0

optimizer_config:
  lr: 1.e-3
  weight_decay: 1.e-4

scheduler_config:
  upper_bound: 600
  max_iter: 500
  infimum_lr: 1.e-6
```

---

## 7. Expected Results

### 7.1. Ablation Comparison

| Version | Architecture | Training | Expected Performance |
|---------|--------------|----------|---------------------|
| v0.13 | Standard (5 layers) | REINFORCE | Good at high-T, poor at $T_c$ |
| v0.15 | **Dilated (8 layers)** | **REINFORCE** | Improved at $T_c$ due to full RF |

### 7.2. Metrics

- **Free Energy Error**: $\Delta F = F_{model} - F_{exact}$ at each temperature
- **Critical Region**: $\beta \in [0.4, 0.5]$ (around $T_c \approx 2.27$)
- **Success Criterion**: $|\Delta F| < 0.1$ across all temperatures

---

## 8. Conclusion

v0.15 represents an **architecture-focused** improvement to the PixelCNN:

1. **Dilated Convolutions** expand the receptive field exponentially, allowing the model to see long-range correlations at the critical point.

2. **Pure REINFORCE** training tests whether self-training alone (without MCMC guidance) can learn the Boltzmann distribution when the architecture has sufficient capacity.

3. **Curriculum Learning** provides training stability by gradually introducing difficult temperatures.

This configuration serves as an **ablation study** to isolate the contribution of architectural improvements from algorithmic improvements (MCMC guidance). Results will inform whether future versions need external supervision or can rely on self-training with sufficiently powerful architectures.
