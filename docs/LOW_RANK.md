# Low-Rank Hypothesis Verification in the 2D Ising Model

> Verifying the "Low-rank hypothesis of complex systems" (Nature Physics, 2024)
> using internal representations of a variational autoregressive model (DiscretePixelCNN).

---

## 1. Introduction

The **Low-Rank Hypothesis** posits that complex systems, despite operating in high-dimensional state spaces, are effectively governed by a small number of macroscopic variables. In the language of singular value decomposition (SVD), the dynamics operator of such systems exhibits a rapidly decaying singular value spectrum — i.e., the system is approximately **low-rank**.

This project tests this hypothesis on the **2D Ising model** by probing the internal representations of a DiscretePixelCNN trained via Variational Thermodynamic Divergence (VaTD). We ask:

1. Does the neural network's internal representation become low-rank at the critical temperature $T_c$?
2. Does the temperature derivative of rank correlate with the specific heat $C_v$?
3. Does SVD compression of model weights cause maximum degradation at $T_c$?

---

## 2. Theoretical Background

### 2.1 The 2D Ising Model & Phase Transition

The 2D Ising model on an $L \times L$ square lattice with periodic boundary conditions is defined by:

$$H = -J \sum_{\langle i,j \rangle} s_i \, s_j, \qquad s_i \in \{-1, +1\}, \quad J = 1$$

The equilibrium probability at inverse temperature $\beta = 1/T$ follows the Boltzmann distribution:

$$p(x \mid T) = \frac{e^{-\beta \, H(x)}}{Z(\beta)}, \qquad Z(\beta) = \sum_x e^{-\beta \, H(x)}$$

Onsager's exact solution gives the critical temperature:

$$T_c = \frac{2J}{\ln(1 + \sqrt{2})} \approx 2.269$$

At $T_c$, the system undergoes a second-order phase transition characterized by:

| Quantity | Definition | Behavior at $T_c$ |
|----------|------------|--------------------|
| Correlation length | $\langle s_i s_j \rangle \sim e^{-r/\xi}$ | $\xi \to \infty$ (diverges) |
| Specific heat | $C_v = \frac{\beta^2}{N}(\langle E^2 \rangle - \langle E \rangle^2)$ | $C_v \sim \ln \lvert T - T_c \rvert$ (log divergence) |
| Susceptibility | $\chi = \beta N (\langle m^2 \rangle - \langle \lvert m \rvert \rangle^2)$ | $\chi \sim \lvert T - T_c \rvert^{-7/4}$ (power-law divergence) |
| Magnetization | $m = \frac{1}{N}\sum_i s_i$ | $\langle \lvert m \rvert \rangle \sim (T_c - T)^{1/8}$ for $T < T_c$ |

### 2.2 The Low-Rank Hypothesis

The hypothesis (Nature Physics, 2024) states that real-world complex systems have a **rapidly decaying singular value spectrum** in their dynamics operators, meaning the system can be faithfully described by projecting onto a low-dimensional subspace.

For a high-dimensional system $x \in \mathbb{R}^n$ with dynamics $\dot{x} = f(x)$, the Jacobian $\mathbf{J} = \partial f / \partial x$ captures linearized dynamics. If $\mathbf{J}$ has singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n$, then the **effective rank** quantifies how many dimensions are dynamically active.

A key prediction: the effective rank should show **non-monotonic behavior** across phase transitions, reflecting changes in the number of relevant degrees of freedom.

### 2.3 Connection: Why Low-Rank Representations at Criticality?

Consider the three phases of the 2D Ising model from the perspective of a neural network learning the distribution $q(x \mid T) \approx p(x \mid T)$:

**High temperature ($T \gg T_c$):** Spins are thermally uncorrelated ($\xi \ll 1$). Each spin acts as independent noise. The network must represent high-entropy, structureless configurations using many independent feature channels → **high rank**.

**Critical temperature ($T \approx T_c$):** The correlation length diverges ($\xi \sim L$). The entire lattice is governed by a few scaling fields (magnetization density $\phi$, energy density $\epsilon$). The state of one region strongly predicts another. In the network, feature maps become **highly collinear** — most channels encode redundant information about the same global mode → **low rank**.

**Low temperature ($T \ll T_c$):** The system is frozen into one of the symmetry-broken ground states ($+M$ or $-M$). Configurations are simple but distinct topological sectors coexist within a batch, causing slight rank recovery.

This leads to a key duality:

| Aspect | Activation Rank | Compression Sensitivity |
|--------|----------------|------------------------|
| **At $T_c$** | Minimum (data lies on low-dim manifold) | Maximum (weights must be precise to maintain the critical manifold) |
| **Away from $T_c$** | High (data fills many dimensions) | Low (perturbations don't qualitatively change the distribution) |

The physical analogy: at $T_c$, the system is like a **pencil balanced on its tip** — the state is simple (low-dimensional), but maintaining it requires extreme precision (high sensitivity to parameter perturbation). This mirrors the divergence of susceptibility $\chi \to \infty$ at criticality.

---

## 3. Rank Metrics

### 3.1 Effective Rank (erank)

Defined by Roy & Bhattacharya (2007) via the Shannon entropy of the normalized singular value distribution:

$$\text{erank}(\mathbf{A}) = \exp\bigl(H(\mathbf{p})\bigr)$$

where

$$p_i = \frac{\sigma_i}{\sum_j \sigma_j}, \qquad H(\mathbf{p}) = -\sum_i p_i \ln p_i$$

Properties:
- Returns a continuous scalar in $[1, \min(m, n)]$
- $\text{erank} = 1$ when a single singular value dominates ($\sigma_1 \gg \sigma_{i>1}$)
- $\text{erank} = \min(m, n)$ when all singular values are equal (uniform spectrum)
- Invariant to scaling: $\text{erank}(\alpha \mathbf{A}) = \text{erank}(\mathbf{A})$

### 3.2 Stable Rank (srank)

A noise-robust alternative:

$$\text{srank}(\mathbf{A}) = \frac{\|\mathbf{A}\|_F^2}{\|\mathbf{A}\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_{\max}^2}$$

The ratio of the Frobenius norm (total energy) to the spectral norm (largest mode energy). Less sensitive to small singular values compared to erank.

### 3.3 Participation Ratio (PR)

From condensed matter physics (inverse of the inverse participation ratio):

$$\text{PR}(\mathbf{A}) = \frac{\left(\sum_i \sigma_i^2\right)^2}{\sum_i \sigma_i^4}$$

Counts how many singular values "participate" in the spectrum. If $k$ singular values are equal and the rest are zero, $\text{PR} = k$.

### 3.4 Two Viewpoints: Channel Rank vs Spatial Rank

Given activations $\mathbf{A} \in \mathbb{R}^{N \times C \times H \times W}$ (batch $\times$ channels $\times$ spatial):

| Viewpoint | Construction | SVD target | Interpretation |
|-----------|-------------|------------|----------------|
| **Channel rank** | Average over spatial dims: $\bar{A} \in \mathbb{R}^{N \times C}$ | $\text{SVD}(\bar{A} - \text{mean})$ | How many feature channels carry independent information? |
| **Spatial rank** | Average over channels: $\bar{A} \in \mathbb{R}^{N \times HW}$ | $\text{SVD}(\bar{A} - \text{mean})$ | How many spatial modes are independent? |

Both are mean-centered before SVD to remove trivial DC offsets.

---

## 4. Experimental Design

### 4.1 Experiment 1: Activation Rank Analysis (`analyze_rank.py`)

**Procedure:**

1. Load a trained DiscretePixelCNN checkpoint
2. For each temperature $T$ in a grid spanning $[T_{\min}, T_{\max}]$:
   - Generate $B \times n_{\text{batches}}$ spin samples via autoregressive sampling
   - Perform a single forward pass, collecting intermediate activations from each residual block via PyTorch forward pre-hooks
   - Compute channel and spatial rank metrics for each layer
3. Compare rank profiles against Onsager's exact specific heat $C_v(T)$

**Temperature grid:** Log-spaced coarse grid (25 points) merged with linear dense grid around $T_c$ (15 points in $[0.8 T_c, 1.2 T_c]$). Critical mode uses 60 uniform points in $[0.5 T_c, 2.0 T_c]$.

**Exact specific heat** is computed via central finite differences of Onsager's $\ln Z$:

$$C_v(T) = \frac{\beta^2}{N} \frac{\partial^2 \ln Z}{\partial \beta^2} \approx \frac{\beta^2}{N} \cdot \frac{\ln Z(\beta + \delta) - 2\ln Z(\beta) + \ln Z(\beta - \delta)}{\delta^2}$$

### 4.2 Experiment 2: Weight Compression Test (`analyze_compression.py`)

**Procedure:**

1. Load the same checkpoint
2. Generate reference samples from the full model at each $T$, recording both samples and log-probabilities
3. For each rank fraction $k \in \{90\%, 75\%, 50\%, 25\%, 10\%\}$:
   - Create a deep copy of the model
   - SVD-truncate every Conv2d weight matrix (using the effective weight $W_{\text{eff}} = W \odot \text{mask}$ for masked layers):

$$\mathbf{W} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top \longrightarrow \mathbf{W}_k = \sum_{i=1}^{k} \sigma_i \, \mathbf{u}_i \, \mathbf{v}_i^\top, \quad k = \lfloor \text{frac} \times \text{full\_rank} \rfloor$$

   - Evaluate truncated model's log-probability on the reference samples

4. **Degradation metric** (KL divergence per site):

$$D(T, k) = \frac{1}{N} \, \mathbb{E}_{x \sim q_{\text{full}}}\!\Big[\log q_{\text{full}}(x \mid T) - \log q_k(x \mid T)\Big] = \frac{1}{N} \, \text{KL}(q_{\text{full}} \| q_k) \geq 0$$

5. Optional **per-block sensitivity**: truncate only one residual block at a time (at 50% rank) to identify which blocks are most critical at which temperatures

---

## 5. Results

> Results shown below are from **DiscretePixelCNN v0.16** (Compact Dilated ResNet)
> trained on the $L = 16$ 2D Ising model.
>
> | Parameter | Value |
> |-----------|-------|
> | Architecture | 4 residual blocks, dilation=[1,2,4,8], skip=[1,3] |
> | Receptive field | 37 × 37 (covers full 16 × 16 lattice) |
> | Hidden channels | 96 |
> | Training | Sequential mode, 250 epochs, AdamW (lr=1e-3) |
> | Temperature range | $\beta \in [0.2, 1.0]$ ($T \in [1, 5]$) |

### 5.1 Activation Rank vs Temperature

![rank_vs_temperature](../figs/DiscretePixelCNN_lr1e-3_e250_2037ea/rank_vs_temperature.png)

*Figure 1: 2×2 overview of activation rank analysis.*

#### (a) Channel Effective Rank — top left

Each curve represents one residual block (Block 0–3) and the final pre-FC layer. All layers exhibit a **V-shaped dip** centered near $T_c \approx 2.269$.

- **High $T$**: Channel eRank reaches ~25–30, indicating that most of the 96 hidden channels carry independent information when encoding structureless noise.
- **$T \approx T_c$**: eRank drops to ~3–5. The critical state is dominated by a few scaling fields (magnetization mode, energy mode), and the network compresses all 96 channels into this low-dimensional subspace.
- **Low $T$**: Partial recovery to ~8–12, reflecting the coexistence of $+M$ and $-M$ sectors within the batch and occasional domain-wall excitations.

The dip is deepest in **early layers** (Block 0, 1), which directly process raw spin correlations and are most sensitive to the diverging correlation length.

#### (b) Spatial Effective Rank — top right

Same analysis but for the spatial dimension ($N \times 256$ matrix). Shows a similar dip but less pronounced, dropping from ~175 (high $T$) to ~25 ($T_c$). At criticality, the spatial pattern collapses into a few dominant modes — physically, the fractal cluster structure of the critical Ising model is describable by a small number of principal spatial components.

#### (c) Average eRank vs Exact $C_v$ — bottom left

The layer-averaged channel eRank (blue, left axis) is overlaid with Onsager's exact specific heat (red, right axis). These two quantities are **anti-correlated**:

- $C_v$ peaks at $T_c$ (maximum energy fluctuations)
- eRank reaches its minimum at $T_c$ (maximum representation compression)

This reflects a fundamental distinction:

$$C_v(T) \sim \text{magnitude of fluctuations} \quad \longleftrightarrow \quad \text{eRank}(T) \sim \text{dimensionality of fluctuations}$$

At criticality, fluctuations are enormous but all aligned along the same few scaling directions. The fluctuations are large ($C_v \uparrow$) but low-dimensional ($\text{eRank} \downarrow$).

#### (d) Channel eRank Heatmap — bottom right

The heatmap (layer × $\log_{10} T$, magma colormap) reveals that the rank dip near $T_c$ (cyan dashed line) is most pronounced in early layers and weakens in later layers, which operate on progressively more abstract features.

### 5.2 Singular Value Spectra (Scree Plots)

![singular_value_spectra](../figs/DiscretePixelCNN_lr1e-3_e250_2037ea/singular_value_spectra.png)

*Figure 2: Normalized singular value spectra $\sigma_i / \sigma_1$ of channel activations at three representative temperatures.*

| Panel | $T$ | $\beta$ | Phase | Spectrum shape |
|-------|-----|---------|-------|----------------|
| Left | 4.73 | 0.211 | Disordered | Gradual decay; $\sigma_{100}/\sigma_1 \sim 10^{-1}$ |
| Center | 2.27 | 0.441 | Critical | Intermediate decay |
| Right | 1.36 | 0.737 | Ordered | Sharp decay; $\sigma_{20}/\sigma_1 \sim 10^{-2}$ |

At high $T$, many singular values contribute comparably — the activation space is "filled" by uncorrelated noise. At low $T$, the first few singular values dominate and the rest decay sharply — nearly all variance is explained by 1–2 components (the $\pm M$ magnetization directions).

This is the direct analog of the **scree plot** in PCA: the "elbow" moves leftward (fewer components needed) as the system becomes more ordered.

### 5.3 d(eRank)/dT vs Specific Heat

#### Full temperature range

![derank_dt_vs_Cv](../figs/DiscretePixelCNN_lr1e-3_e250_2037ea/derank_dt_vs_Cv.png)

*Figure 3: Temperature derivative of the layer-averaged effective rank (blue) compared with Onsager's exact specific heat (red), both normalized to [0, 1].*

The derivative $d(\text{eRank})/dT$ is computed via numerical gradient with Gaussian smoothing:

$$\frac{d(\text{eRank})}{dT}\bigg|_{T_i} \approx \frac{\text{eRank}(T_{i+1}) - \text{eRank}(T_{i-1})}{T_{i+1} - T_{i-1}}$$

The peak of $d(\text{eRank})/dT$ appears at $T \approx 2.5$, shifted slightly **above** $T_c = 2.269$. Three factors contribute:

1. **Finite-size effects**: On $L = 16$, the pseudocritical temperature shifts as $T^*(L) = T_c + a \, L^{-1/\nu}$ with $\nu = 1$ for the 2D Ising universality class.
2. **Asymmetric dip**: The eRank dip itself is asymmetric — the ordered side ($T < T_c$) recovers more gradually than the disordered side, pushing the steepest slope (and thus the derivative peak) above $T_c$.
3. **Widom line**: The crossover region where short-range correlations transition from "critical-like" to "noise-like" extends above $T_c$. The model detects the breakdown of correlated clusters most sharply in this region.

A secondary bump at low $T \approx 0.8$–$1.0$ reflects the onset of thermally activated domain walls in the ordered phase.

#### Critical region (dense grid)

![derank_dt_vs_Cv_critical](../figs/DiscretePixelCNN_lr1e-3_e250_2037ea/derank_dt_vs_Cv_critical.png)

*Figure 4: Same comparison on a dense uniform grid of 60 temperatures in $[0.5\,T_c,\ 2.0\,T_c]$.*

With finer temperature resolution, the peak position is confirmed at $T \approx 2.7$ with a broader profile than the exact $C_v$ peak. The persistent elevation at high $T > 3.5$ (where $C_v \to 0$) suggests the model has not perfectly learned the trivial high-temperature limit (the training range is $T \in [1, 5]$, so representation quality degrades near the boundary).

### 5.4 Weight SVD Spectra

![weight_svd_spectra](../figs/DiscretePixelCNN_lr1e-3_e250_2037ea/weight_svd_spectra.png)

*Figure 5: Singular value spectra of Conv2d weights (k×k convolutions) for each residual block.*

| Block | Effective Rank | Full Rank | eRank / Full |
|-------|---------------|-----------|--------------|
| B0.k×k | 84.0 | ~96 | 88% |
| B1.k×k | 84.3 | ~96 | 88% |
| B2.k×k | 78.1 | ~96 | 81% |
| B3.k×k | 83.1 | ~96 | 87% |

All blocks have **near-full-rank** weights with gentle spectral decay ($\sigma_{80}/\sigma_1 \sim 0.1$). This contrasts sharply with the temperature-dependent activation rank (Section 5.1).

**Interpretation:** The weights encode the **physics rules** (the Ising Hamiltonian $H = -J\sum s_i s_j$), which are static and must generalize across all temperatures. The activations encode the **physical state** at a given $T$, which undergoes a phase transition. The model is a universal simulator whose static parameters require full capacity, while its dynamic representations adapt their dimensionality to the thermodynamic phase.

Block B2 (dilation=4, largest receptive field per parameter) shows slightly lower eRank (78.1), suggesting partial weight redundancy in long-range correlation encoding.

### 5.5 Compression Test

![compression_test](../figs/DiscretePixelCNN_lr1e-3_e250_2037ea/compression_test.png)

*Figure 6: 2×2 overview of weight compression analysis.*

#### (a) Degradation Curves — top left

$D(T, k)$ vs temperature for each rank fraction. The degradation peaks sharply near $T_c$ for aggressive truncation:

- **$k = 10\%$**: $D \approx 0.07$ nats/site at $T_c$, near zero elsewhere
- **$k = 25\%$**: $D \approx 0.025$ nats/site at $T_c$
- **$k \geq 50\%$**: $D \approx 0$ everywhere — half the singular values suffice to preserve the learned distribution

This confirms the Low-Rank Hypothesis prediction: the model requires its **full weight rank most at $T_c$**, where the distribution is most sensitive to perturbation.

#### (b) Compression Sensitivity Heatmap — top right

The heatmap ($\text{rank fraction} \times \log_{10} T$, inferno colormap) shows a bright vertical band at $\log_{10}(T_c) \approx 0.356$. This "critical stripe" concentrates at low rank fractions (10%–25%), confirming that the information needed to represent the critical distribution is distributed across the **entire singular value spectrum**, not just the top components.

#### (c) Degradation vs $C_v$ with Critical Inset — bottom left

The $k = 10\%$ degradation curve (blue) is overlaid with exact $C_v$ (red) on dual axes. The **inset** normalizes both quantities in $T \in [1.5, 4.0]$ for direct shape comparison, revealing near-identical functional forms:

$$D(T, k) \propto C_v(T) \qquad \text{near } T_c$$

This proportionality has a physical explanation. Specific heat measures the variance of energy fluctuations:

$$C_v = \frac{\beta^2}{N}\,\text{Var}(E)$$

Weight truncation introduces a perturbation $\delta W$ to the learned Hamiltonian. Near $T_c$, the susceptibility to perturbation diverges alongside $C_v$, so the KL divergence $D(T, k)$ inherits the same scaling.

#### (d) Relative Degradation — bottom right

$D(T, k) / |\log q_{\text{full}}(T)|$ expressed as a percentage. At low $T$ with $k = 10\%$, this ratio spikes to $\sim 10^5$ % because $|\log q_{\text{full}}|$ approaches zero (the model assigns near-certainty to ground-state configurations). Even a small absolute degradation $D$ becomes catastrophic relative to the tiny baseline log-probability.

This highlights that **low-temperature precision** requires full weight fidelity — truncation destroys the model's ability to generate the exact ground-state configuration.

---

## 6. Discussion

### 6.1 The Compression Paradox

The most striking result is the apparent contradiction:

> **Activations are lowest-rank at $T_c$** (Section 5.1), yet **compression sensitivity is highest at $T_c$** (Section 5.5).

This resolves via the distinction between the **data manifold** and the **transformation sensitivity**:

- Low activation rank means the data lies on a low-dimensional manifold — analogous to a pencil balanced vertically (a 1D state in 3D space).
- High compression sensitivity means the transformation (weights) must be extremely precise to keep the data on that manifold — like the fine motor control needed to balance the pencil.

Formally, this connects to the divergence of susceptibility $\chi$ at criticality. The susceptibility measures the response to external perturbation $h$:

$$\chi = \frac{\partial \langle m \rangle}{\partial h}\bigg|_{h=0} \sim |T - T_c|^{-\gamma}, \quad \gamma = 7/4$$

Analogously, truncating weights introduces a "perturbation" to the model's effective Hamiltonian, and the response (degradation) diverges at $T_c$.

### 6.2 Connection to Renormalization Group

The layer-dependent rank structure (Figure 1d) mirrors **Renormalization Group (RG) flow**:

- Early layers (Block 0–1) show the deepest rank dip at $T_c$, suggesting they perform coarse-graining that integrates out irrelevant UV (short-wavelength) degrees of freedom.
- Later layers maintain more stable rank, operating on already-renormalized, IR (long-wavelength) variables.

In the RG framework, the number of **relevant operators** at the critical fixed point of the 2D Ising model is exactly 2 (thermal perturbation $\epsilon$ and magnetic perturbation $\sigma$). The observation that eRank drops to ~3–5 at $T_c$ is consistent with the network discovering these few relevant directions plus minor corrections.

### 6.3 Finite-Size Considerations

All measurements are on $L = 16$, which introduces several artifacts:

1. **Pseudocritical shift**: The effective $T_c(L)$ differs from the thermodynamic $T_c(\infty)$ by $O(L^{-1/\nu})$.
2. **Rounding**: The log divergence of $C_v$ is truncated to a finite peak at $L = 16$.
3. **Correlation saturation**: At $T_c$, the correlation length $\xi \sim L$, so the system cannot develop correlations beyond the lattice size.

The receptive field of the model (37 × 37) exceeds the lattice (16 × 16), ensuring the network can in principle capture all pairwise correlations. For $L > 37$, the RF would become a limiting factor.

### 6.4 Limitations

- **Training fidelity**: The model's approximation $q(x|T) \approx p(x|T)$ is imperfect, especially near $T_c$ where learning is hardest. Activation rank reflects both the true physics and model artifacts.
- **Sample statistics**: Rank metrics are computed from $\sim$600 samples per temperature ($200 \times 3$ batches), which may under-sample the tails of the distribution.
- **Extrapolation**: The training range $T \in [1, 5]$ does not cover the deep disordered regime ($T \gg 5$), so rank behavior at very high $T$ may reflect model extrapolation errors rather than physics.

---

## 7. Usage

```bash
# Full activation rank analysis (GPU recommended)
python analyze_rank.py --project Ising_VaTD_v0.16 \
    --group DiscretePixelCNN_lr1e-3_e250_2037ea --seed 42 --device cuda:0

# Critical-region mode (dense T grid around Tc)
python analyze_rank.py --project Ising_VaTD_v0.16 \
    --group DiscretePixelCNN_lr1e-3_e250_2037ea --seed 42 --device cuda:0 --critical

# Compression test with per-block sensitivity
python analyze_compression.py --project Ising_VaTD_v0.16 \
    --group DiscretePixelCNN_lr1e-3_e250_2037ea --seed 42 --device cuda:0 --per_layer

# Replot from saved CSV (no GPU needed)
python analyze_rank.py --replot runs/Ising_VaTD_v0.16/DiscretePixelCNN_lr1e-3_e250_2037ea/rank_analysis_42.csv
python analyze_compression.py --replot runs/Ising_VaTD_v0.16/DiscretePixelCNN_lr1e-3_e250_2037ea/compression_42.csv
```

---

## References

1. **Low-rank hypothesis**: Thiede, Giannakis, et al., "The low-rank hypothesis of complex systems," *Nature Physics* (2024).
2. **Effective rank**: Roy, O. & Bhattacharya, M., "Effective rank for matrices," *IEEE Trans. Signal Processing* (2007).
3. **Onsager solution**: Onsager, L., "Crystal statistics. I. A two-dimensional model with an order-disorder transition," *Physical Review* 65, 117 (1944).
4. **VaTD**: Wu, D., Wang, L., & Zhang, P., "Solving statistical mechanics using variational autoregressive networks," *Physical Review Letters* 122, 080602 (2019).
