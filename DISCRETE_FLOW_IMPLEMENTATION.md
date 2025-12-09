# Discrete Normalizing Flow Implementation Plan for Ising Model

## 1. Objective
To implement a **Discrete Normalizing Flow** specifically designed for the Ising Model (discrete spins $\{-1, 1\}$). This resolves the limitations of continuous Normalizing Flows (e.g., RealNVP) which require dequantization and suffer from "leaking" probability mass into invalid continuous states.

## 2. Theoretical Basis: XOR Coupling
Unlike continuous flows that use affine transformations ($y = x \cdot s + t$), discrete flows on binary data use **Modulo-2 Arithmetic (XOR)**.

### Forward Transformation
For a binary input $x \in \{0, 1\}$ split into $x_A$ (active) and $x_B$ (frozen):
$$ y_A = (x_A + \text{Round}(\mathcal{F}(x_B, T))) \pmod 2 $$
$$ y_B = x_B $$

- $\mathcal{F}$: A neural network (e.g., CNN) predicting a shift parameter.
- $\text{Round}$: Rounds the continuous output of $\mathcal{F}$ to the nearest integer $\{0, 1\}$.
- **Jacobian**: The transformation is volume-preserving discrete permutation. Log-determinant is always 0.

### Inverse Transformation
$$ x_A = (y_A - \text{Round}(\mathcal{F}(y_B, T))) \pmod 2 $$
Since in modulo 2 arithmetic subtraction is addition (XOR):
$$ x_A = (y_A + \text{Round}(\mathcal{F}(y_B, T))) \pmod 2 $$
The inverse operation is **identical** to the forward operation.

## 3. Implementation Steps

### Step 1: Define Helper Components in `model.py`

#### 1.1 `StraightThroughRound` Class
Since `torch.round` is non-differentiable, we need a custom autograd function or a straight-through estimator (STE) hook.

*   **Logic**:
    *   Forward: Return `torch.round(x)`
    *   Backward: Return `grad_output` (Identity)
*   **Usage**: `shift_discrete = shift_continuous + (shift_continuous.round() - shift_continuous).detach()`

### Step 2: Implement `BinaryCouplingLayer` in `model.py`

This class replaces `CheckerboardAffineCoupling` for the discrete case.

*   **Constructor Args**:
    *   `H`, `W`: Lattice dimensions.
    *   `parity`: 0 or 1 (for checkerboard masking).
    *   `hidden_channels`: For the internal CNN.
*   **Components**:
    *   `mask`: Checkerboard mask (buffer).
    *   `net`: `ConditionalCouplingNet` (reused from existing code).
        *   *Note*: The existing `ConditionalCouplingNet` outputs 2 channels (scale, shift). For discrete flow, we ignore the scale output and only use the shift.
*   **Forward Method (`x`, `T`)**:
    1.  Input `x` is in $\{0, 1\}$.
    2.  Mask `x` into `x_frozen` and `x_active`.
    3.  Pass `x_frozen` and `T` through `self.net` $\to$ get `raw_shift`.
    4.  Apply Sigmoid to `raw_shift` to get probability $p \in [0, 1]$.
    5.  Apply **STE Rounding** to $p$ to get `shift` $\in \{0, 1\}$.
    6.  Compute `z_active = (x_active + shift) % 2`.
    7.  Return `z`.
*   **Inverse Method**:
    *   Same logic as `Forward` (XOR is self-inverse).

### Step 3: Implement `DiscreteFlowModel` in `model.py`

This is the main model class matching the interface of `DiscretePixelCNN`.

*   **Constructor**:
    *   Accepts `hparams`.
    *   Initializes a list of `BinaryCouplingLayer`.
    *   **Base Distribution**: Unlike continuous flows (Gaussian), the base distribution $P(z)$ should be a **Factorized Bernoulli**.
        *   Parameters: `base_logits` (learnable).
        *   Ideally, $P(z|T)$ should depend on $T$. Use a small MLP: $T \to \text{logits}$.
*   **`log_prob(sample, T)`**:
    1.  Map input `sample` $\{-1, 1\} \to \{0, 1\}$ via $x_{bin} = (x + 1) / 2$.
    2.  Pass through `forward_flow` layers to get latent $z$.
    3.  Compute $\log P(z)$ using `BCEWithLogitsLoss` (negative sum) against the learned base distribution logits.
    4.  Return shape `(B, 1)`.
*   **`sample(batch_size, T)`**:
    1.  Sample $z$ from the base Bernoulli distribution using `torch.bernoulli(sigmoid(logits))`.
    2.  Pass through `inverse_flow` layers to get $x_{bin}$.
    3.  Map $\{0, 1\} \to \{-1, 1\}$ via $x = x_{bin} * 2 - 1$.

### Step 4: Configuration Updates in `config.py`

*   Update `FlowConfig` dataclass to include parameters relevant to discrete flows if necessary (though existing params like `num_flow_layers` usually suffice).
*   Alternatively, add a `model_type` field to distinguish between Continuous and Discrete flows if both are kept.

## 4. Code Snippets for Reference

### Straight Through Estimator
```python
def ste_round(x):
    return x + (x.round() - x).detach()
```

### XOR Coupling Logic
```python
# Inside BinaryCouplingLayer.forward
# net_out is the output of the CNN
shift_prob = torch.sigmoid(net_out)
shift = ste_round(shift_prob)
z = (x + shift) % 2  # Modulo 2 addition
```

### Base Distribution Logic
```python
# Inside DiscreteFlowModel
# T-dependent base distribution
base_logits = self.base_mlp(T) # (B, 1, H, W)
log_prob = -F.binary_cross_entropy_with_logits(base_logits, z, reduction='none').sum(dim=[1,2,3])
```

## 5. Integration Checklist

- [ ] Ensure `model.py` imports necessary modules.
- [ ] Add `DiscreteFlowModel` class.
- [ ] Verify `sample` method returns `{-1, 1}` tensor.
- [ ] Verify `log_prob` handles dimension broadcasting for `T`.
- [ ] Update `RunConfig.gen_group_name` in `config.py` if the model class name changes (e.g., to `DiscreteFlowModel`).
