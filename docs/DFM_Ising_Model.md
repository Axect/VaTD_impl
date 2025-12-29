# Discrete Flow Matching for 2D Ising Model

이 문서는 2D Ising 모델의 Boltzmann 분포를 학습하기 위한 Discrete Flow Matching (DFM) 구현을 상세하게 설명합니다.

---

## 목차

1. [개요](#1-개요)
2. [배경 지식](#2-배경-지식)
   - 2.1 [2D Ising 모델](#21-2d-ising-모델)
   - 2.2 [Flow Matching 기초](#22-flow-matching-기초)
3. [Discrete Flow Matching 이론](#3-discrete-flow-matching-이론)
   - 3.1 [Dirichlet 확률 경로](#31-dirichlet-확률-경로)
   - 3.2 [Velocity Field](#32-velocity-field)
   - 3.3 [Denoising Objective](#33-denoising-objective)
4. [아키텍처](#4-아키텍처)
   - 4.1 [ResConv2D 백본](#41-resconv2d-백본)
   - 4.2 [시간 임베딩](#42-시간-임베딩)
5. [학습 알고리즘](#5-학습-알고리즘)
   - 5.1 [Energy-Guided Training](#51-energy-guided-training)
   - 5.2 [Self-Imitation Learning](#52-self-imitation-learning)
   - 5.3 [Equilibration 알고리즘](#53-equilibration-알고리즘)
     - 5.3.1 [Swendsen-Wang 클러스터 알고리즘](#531-swendsen-wang-클러스터-알고리즘)
     - 5.3.2 [Metropolis-Hastings (fallback)](#532-metropolis-hastings-fallback)
   - 5.4 [Curriculum Learning](#54-curriculum-learning)
6. [샘플링 알고리즘](#6-샘플링-알고리즘)
7. [검증 및 평가](#7-검증-및-평가)
8. [구현 세부사항](#8-구현-세부사항)
9. [참고문헌](#9-참고문헌)

---

## 1. 개요

### 동기

2D Ising 모델의 Boltzmann 분포 $p(x) = \frac{1}{Z(\beta)} e^{-\beta E(x)}$를 학습하는 것은 통계물리학과 기계학습의 중요한 교차점입니다. 기존의 autoregressive 모델(PixelCNN)은 다음과 같은 한계가 있습니다:

| 문제점 | 설명 |
|--------|------|
| **순차적 샘플링** | $L \times L$ 격자에서 $L^2$번의 순차적 forward pass 필요 |
| **Masked Convolution** | 인과적 마스킹으로 인한 receptive field 제한 |
| **긴 의존성** | 멀리 떨어진 스핀 간의 상관관계 학습 어려움 |

### DFM의 장점

**Discrete Flow Matching (DFM)**은 이러한 한계를 극복합니다:

- **병렬 샘플링**: ODE integration으로 모든 위치를 동시에 업데이트
- **Full Receptive Field**: 마스킹 없이 전체 격자를 한 번에 처리
- **Energy Guidance**: Swendsen-Wang 클러스터 알고리즘으로 물리적으로 올바른 샘플 생성
- **Critical Point 학습**: SW 알고리즘으로 임계점(Tc) 근처에서도 정확한 equilibration

```
기존 PixelCNN:  [pixel 1] → [pixel 2] → ... → [pixel 256]  (순차적)
                    ↓           ↓                  ↓
DFM:            [uniform] ─────ODE────────→ [Boltzmann]   (병렬)
```

---

## 2. 배경 지식

### 2.1 2D Ising 모델

#### 정의

2D Ising 모델은 정사각 격자 $\Lambda = \{1, ..., L\}^2$ 위에 정의된 스핀 시스템입니다.

$$
x = \{s_{i,j}\}_{(i,j) \in \Lambda}, \quad s_{i,j} \in \{-1, +1\}
$$

#### 에너지 함수

주기적 경계조건(periodic boundary conditions)을 가진 Hamiltonian:

$$
E(x) = -J \sum_{\langle i,j \rangle} s_i \cdot s_j
$$

여기서 $\langle i,j \rangle$는 최근접 이웃 쌍을 의미하고, $J > 0$은 강자성 상호작용 상수입니다.

```python
# main.py:72-95 - Ising 에너지 함수 구현
def energy_fn(samples):
    """
    O(N) complexity using nearest-neighbor computation.

    Args:
        samples: (B, 1, H, W) tensor with values in {-1, 1}
    Returns:
        (B, 1) energy values
    """
    s = samples[:, 0]  # (B, H, W)

    # 주기적 경계조건으로 이웃 스핀 계산
    right = torch.roll(s, -1, dims=-1)  # 오른쪽 이웃
    down = torch.roll(s, -1, dims=-2)   # 아래쪽 이웃

    # E = -J * Σ s_i * s_j (J=1)
    energy = -(s * right + s * down).sum(dim=[1, 2])
    return energy.unsqueeze(-1)  # (B, 1)
```

#### Boltzmann 분포

역온도 $\beta = 1/T$에서의 평형 분포:

$$
p_\beta(x) = \frac{e^{-\beta E(x)}}{Z(\beta)}, \quad Z(\beta) = \sum_{x} e^{-\beta E(x)}
$$

#### 임계 온도 (Onsager Solution)

2D Ising 모델은 정확히 풀린 몇 안 되는 통계역학 모델 중 하나입니다:

$$
T_c = \frac{2J}{\ln(1 + \sqrt{2})} \approx 2.269
$$

```python
# vatd_exact_partition.py:194
CRITICAL_TEMPERATURE = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.269185
```

$T > T_c$ (high temperature): 무질서 상태, $\langle m \rangle = 0$
$T < T_c$ (low temperature): 질서 상태, $\langle m \rangle \neq 0$ (자발적 자화)

### 2.2 Flow Matching 기초

#### Continuous Normalizing Flows

Flow matching은 확률 분포 간의 연속적인 변환을 학습합니다:

$$
\frac{dx_t}{dt} = v_t(x_t), \quad t \in [0, T]
$$

여기서 $v_t$는 velocity field입니다.

#### 확률 밀도의 진화

velocity field에 의한 확률 밀도의 시간 진화는 continuity equation을 따릅니다:

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v_t) = 0
$$

#### Log-likelihood 변화

$$
\frac{d \log p_t(x_t)}{dt} = -\nabla \cdot v_t(x_t)
$$

이는 ODE를 따라 정확한 likelihood 계산을 가능하게 합니다.

---

## 3. Discrete Flow Matching 이론

### 3.1 Dirichlet 확률 경로

#### 이산 상태의 확률 표현

Ising 스핀 $s \in \{-1, +1\}$를 확률 simplex로 임베딩합니다:

$$
s = -1 \rightarrow \mathbf{e}_0 = (1, 0)
$$
$$
s = +1 \rightarrow \mathbf{e}_1 = (0, 1)
$$

연속 시간에서의 상태 $y_t \in \Delta^1$ (1-simplex, 즉 선분 $[0,1]$)

#### Dirichlet 분포

주어진 clean sample $x$에서 시간 $t$에서의 noisy 분포:

$$
p_t(y | x) = \text{Dir}(y; \alpha), \quad \alpha = \mathbf{1} + t \cdot \text{onehot}(x)
$$

**예시**: $x = +1$ (즉, $\mathbf{e}_1$)인 경우
- $t = 0$: $\alpha = (1, 1)$ → Uniform on simplex
- $t = 10$: $\alpha = (1, 11)$ → 거의 $\mathbf{e}_1$에 집중
- $t \to \infty$: $\alpha = (1, \infty)$ → Delta function at $\mathbf{e}_1$

```python
# model_dfm.py:330-372
def apply_dirichlet_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Apply Dirichlet noising along the probability path.

    p_t(y | x) = Dir(y; alpha = 1 + t * onehot(x))
    """
    B, C, H, W = x.shape

    # Convert {-1, +1} to {0, 1} indices
    x_idx = self.reverse_mapping(x).long()  # (B, 1, H, W)

    # Create one-hot encoding: (B, 2, H, W)
    x_onehot = F.one_hot(x_idx.squeeze(1), num_classes=2)
    x_onehot = x_onehot.permute(0, 3, 1, 2).float()

    # Dirichlet parameters: α = 1 + t * onehot(x)
    t_exp = t.view(B, 1, 1, 1)
    alpha = 1.0 + t_exp * x_onehot  # (B, 2, H, W)

    # Sample from Dirichlet for each spatial position
    alpha_flat = alpha.permute(0, 2, 3, 1).reshape(-1, 2)
    x_noisy_flat = torch.distributions.Dirichlet(alpha_flat).sample()
    x_noisy = x_noisy_flat.reshape(B, H, W, 2).permute(0, 3, 1, 2)

    return x_noisy  # (B, 2, H, W), 각 위치에서 확률 분포
```

#### 수렴 속도

시간 $t$에서 target 방향으로의 수렴도:

$$
\text{convergence}(t) = \frac{t}{2 + t}
$$

| $t$ | Convergence |
|-----|-------------|
| 0 | 0% (uniform) |
| 2 | 50% |
| 10 | 83% |
| 50 | 96% |
| 100 | 98% |

### 3.2 Velocity Field

#### 조건부 Velocity

Dirichlet 경로에서 조건부 velocity field는:

$$
u_t(y | x) = \frac{\mathbf{e}_x - y}{2 + t}
$$

여기서 $\mathbf{e}_x$는 clean sample $x$의 one-hot 인코딩입니다.

#### 물리적 해석

- **방향**: 현재 상태 $y$에서 목표 상태 $\mathbf{e}_x$를 향함
- **크기**: 시간이 지남에 따라 감소 ($1/(2+t)$ 스케일링)
- **수렴**: $t \to \infty$일 때 velocity는 0으로 수렴

```python
# model_dfm.py:375-428
def velocity_field(self, x: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Compute the probability velocity field for Discrete Flow Matching.

    For Dirichlet path: u_t(y|x) = (e_x - y) / (2 + t)
    """
    B, _, H, W = x.shape

    # Time embedding
    t_emb = self.time_embedding(t)

    # Temperature channel
    T_channel = T.view(B, 1, 1, 1).expand(B, 1, H, W)
    x_input = torch.cat([x, T_channel], dim=1)

    # Network predicts target distribution
    logits = self.net(x_input, t_emb).squeeze(2)
    target_prob = F.softmax(logits, dim=1)

    # Time scaling factor: 1 / (2 + t)
    time_scale = 1.0 / (2.0 + t.view(B, 1, 1, 1))

    # Velocity = (predicted_target - current_state) * time_scale
    velocity = (target_prob - x) * time_scale

    return velocity  # (B, 2, H, W)
```

### 3.3 Denoising Objective

#### Cross-Entropy Loss

네트워크가 noisy input에서 clean target을 예측하도록 학습:

$$
\mathcal{L}_{\text{denoise}} = \mathbb{E}_{t, x, y_t} \left[ -\log p_\theta(x | y_t, t, T) \right]
$$

```python
# model_dfm.py:741-774
def denoise(self, x_noisy: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Predict clean state from noisy state."""
    B, _, H, W = x_noisy.shape

    t_emb = self.time_embedding(t)
    T_channel = T.view(B, 1, 1, 1).expand(B, 1, H, W)
    x_input = torch.cat([x_noisy, T_channel], dim=1)

    logits = self.net(x_input, t_emb).squeeze(2)
    return logits  # (B, 2, H, W) - unnormalized log probabilities
```

#### 첫 번째 스핀 고정 (Symmetry Breaking)

Ising 모델의 $\mathbb{Z}_2$ 대칭성을 깨기 위해 첫 번째 스핀을 고정:

```python
# 학습 시 첫 번째 위치 마스킹
if self.fix_first is not None:
    mask = torch.ones(B, H, W, device=self.device)
    mask[:, 0, 0] = 0  # 첫 번째 위치 제외
    ce_loss = F.cross_entropy(logits, target, reduction='none')
    denoise_loss = (ce_loss * mask).sum() / mask.sum()
```

---

## 4. 아키텍처

### 4.1 ResConv2D 백본

DFM은 **마스킹 없는 ResNet** 구조를 사용합니다:

```
Input: [prob_simplex (2) + temperature (1)] = 3 channels
       ↓
    First Conv (7x7) → GELU
       ↓
    + Time Embedding (broadcast)
       ↓
    ResBlock × N (1x1 → 3x3 → 1x1)
       ↓
    FC Layers (1x1 conv)
       ↓
Output: logits (2 channels)
```

```python
# model_dfm.py:73-221 - ResConv2D 클래스
class ResConv2D(nn.Module):
    def __init__(
        self,
        channel: int = 1,           # 데이터 채널 수
        category: int = 2,          # 카테고리 수 (이진 스핀)
        time_dim: int = 64,         # 시간 임베딩 차원
        hidden_channels: int = 64,  # ResBlock 너비
        hidden_conv_layers: int = 5,# ResBlock 개수
        hidden_kernel_size: int = 3,# 공간 커널 크기
        hidden_width: int = 128,    # FC 레이어 너비
        hidden_fc_layers: int = 2,  # FC 레이어 수
        augment_channels: int = 1,  # 온도 채널
    ):
        super().__init__()

        # 입력: probability simplex + temperature
        in_channels = category * channel + augment_channels  # 2 + 1 = 3

        self.first_conv = nn.Conv2d(
            in_channels, 2 * hidden_channels,
            kernel_size=7, padding=3  # 마스킹 없음!
        )

        # Residual blocks (bottleneck 구조)
        for _ in range(hidden_conv_layers):
            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(2*hidden_channels, hidden_channels, 1),  # 압축
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),  # 공간
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, 2*hidden_channels, 1),  # 확장
                )
            )
```

**PixelCNN과의 비교:**

| 특성 | PixelCNN | DFM (ResConv2D) |
|------|----------|-----------------|
| 마스킹 | Type A/B 마스킹 필수 | 마스킹 없음 |
| Receptive Field | 제한적 (causal) | 전체 격자 |
| 입력 | 이전 픽셀들 | 전체 확률 분포 + 시간 |

### 4.2 시간 임베딩

Transformer의 positional encoding을 시간에 적용:

$$
\text{emb}(t)_{2k} = \sin(t / 10000^{2k/d})
$$
$$
\text{emb}(t)_{2k+1} = \cos(t / 10000^{2k/d})
$$

```python
# model_dfm.py:26-70
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64, max_period: float = 10000.0):
        super().__init__()
        half_dim = dim // 2
        # 주파수 밴드 사전 계산
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim) / half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → embedding: (B, dim)"""
        args = t.unsqueeze(-1) * self.freqs  # (B, half_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding  # (B, dim)
```

시간 임베딩은 MLP를 통해 처리된 후 첫 번째 convolution 출력에 더해집니다:

```python
# Time embedding projection
self.time_mlp = nn.Sequential(
    nn.Linear(time_dim, hidden_channels * 2),
    nn.GELU(),
    nn.Linear(hidden_channels * 2, hidden_channels * 2),
)

# Forward에서
t_proj = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
h = self.first_conv(x) + t_proj  # Broadcast over spatial dimensions
```

---

## 5. 학습 알고리즘

### 5.1 Energy-Guided Training

DFM의 핵심 학습 전략은 **energy-guided training**입니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Energy-Guided Training                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 초기화 결정 (90% 모델 / 10% 랜덤)                           │
│                     ↓                                           │
│   2. Equilibration 알고리즘으로 물리적 샘플 생성                  │
│      - use_cluster=true: Swendsen-Wang (기본, critical point OK)│
│      - use_cluster=false: Metropolis-Hastings (fallback)       │
│                     ↓                                           │
│   3. Equilibrated 샘플을 target으로 denoising loss 계산          │
│                     ↓                                           │
│   4. Velocity matching loss 계산 (보조)                         │
│                     ↓                                           │
│   5. 총 손실 = denoise_loss + 0.1 * velocity_loss              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
# model_dfm.py - training_loss 함수
def training_loss(self, samples, T, energy_fn, training_mode="energy_guided",
                  use_cluster=True, n_clusters=10, mh_steps=200, ...):

    if training_mode == "energy_guided":
        # 1. 초기화 결정
        use_model_init = torch.rand(B) < self.mh_model_init_prob  # 0.9
        initial_samples = use_model_init * samples + (1 - use_model_init) * random_samples

        # 2. Equilibration (핵심 변경!)
        if use_cluster:
            # Swendsen-Wang cluster algorithm - no critical slowing down!
            equilibrated_samples = self.swendsen_wang_update(
                initial_samples, T, n_sweeps=n_clusters
            )
        else:
            # MH with checkerboard updates (fallback)
            equilibrated_samples = self.improve_samples_with_energy(
                initial_samples, T, energy_fn, n_steps=mh_steps
            )

        # 3. Denoising loss
        t = torch.rand(B) * (self.t_max - self.t_min) + self.t_min
        x_noisy = self.apply_dirichlet_noise(equilibrated_samples, t)
        logits = self.denoise(x_noisy, t, T)
        target = self.reverse_mapping(equilibrated_samples).squeeze(1).long()
        denoise_loss = F.cross_entropy(logits, target)

        # 4. Velocity matching loss
        v_learned = self.velocity_field(x_noisy, t, T)
        v_target = (target_onehot - x_noisy) * time_scale
        velocity_loss = ((v_learned - v_target.detach()) ** 2).mean()

        # 5. 총 손실
        total_loss = denoise_loss + energy_weight * 0.1 * velocity_loss

        return total_loss, metrics
```

### 5.2 Self-Imitation Learning

**핵심 아이디어**: 모델이 생성한 샘플을 MH로 개선한 후, 개선된 샘플을 모방하도록 학습

```
┌───────────────────────────────────────────────────────┐
│              Self-Imitation Learning                  │
├───────────────────────────────────────────────────────┤
│                                                       │
│   mh_model_init_prob = 0.9                           │
│                                                       │
│   ┌─────────────┐     ┌─────────────┐                │
│   │  90% 확률   │     │  10% 확률   │                │
│   │ 모델 샘플   │     │ 랜덤 샘플   │                │
│   └──────┬──────┘     └──────┬──────┘                │
│          │                   │                       │
│          └────────┬──────────┘                       │
│                   ↓                                   │
│          ┌───────────────┐                           │
│          │  MH 샘플링    │ (equilibration)           │
│          └───────┬───────┘                           │
│                  ↓                                   │
│          ┌───────────────┐                           │
│          │ 개선된 샘플   │ (학습 target)              │
│          └───────────────┘                           │
│                                                       │
│   장점:                                              │
│   - 90% 모델: 저온에서 빠른 equilibration            │
│   - 10% 랜덤: 다양성 유지, mode collapse 방지        │
│                                                       │
└───────────────────────────────────────────────────────┘
```

**왜 이것이 중요한가?**

1. **순수 랜덤 초기화의 문제**: 저온에서 랜덤 상태가 equilibrium에 도달하려면 매우 많은 MH step 필요
2. **순수 모델 초기화의 문제**: 모델의 bias를 강화하는 positive feedback loop 발생
3. **하이브리드 접근**: 둘의 장점을 결합

### 5.3 Equilibration 알고리즘

학습 중 equilibrium 샘플을 생성하기 위한 두 가지 알고리즘을 제공합니다:

| 알고리즘 | 임계점 성능 | 속도 | 기본값 |
|----------|-------------|------|--------|
| **Swendsen-Wang** | 우수 (τ ~ O(1)) | 빠름 | ✓ |
| Metropolis-Hastings | 느림 (τ ~ L^z, z≈2.2) | 느림 | fallback |

#### 5.3.1 Swendsen-Wang 클러스터 알고리즘

**Critical Slowing Down 문제**

Metropolis-Hastings (MH) 알고리즘은 임계점(Tc ≈ 2.27) 근처에서 심각한 성능 저하를 겪습니다:

- **자기상관 시간**: τ ~ L^z (z ≈ 2.2)
- **16×16 격자에서**: τ ≈ 16^2.2 ≈ 500 sweeps
- **MH 200 sweeps 시 오차**: ~20 에너지 단위

Swendsen-Wang (SW) 클러스터 알고리즘은 이 문제를 해결합니다:
- **자기상관 시간**: τ ~ O(1) (격자 크기에 거의 무관)
- **SW 10 sweeps 시 오차**: ~1 에너지 단위

**알고리즘 원리**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Swendsen-Wang Algorithm                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Bond Activation (확률적 연결)                               │
│      - 같은 스핀을 가진 이웃 쌍에 대해                            │
│      - 확률 p = 1 - exp(-2β)로 bond 활성화                       │
│                                                                  │
│   2. Cluster Identification                                      │
│      - 활성화된 bond로 연결된 스핀들 → 하나의 클러스터            │
│      - Label propagation으로 병렬 처리                           │
│                                                                  │
│   3. Cluster Flipping                                            │
│      - 각 클러스터를 독립적으로 50% 확률로 뒤집음                 │
│      - 대규모 스핀 업데이트 → critical slowing down 해결!        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**구현 코드**

```python
# model_dfm.py - swendsen_wang_update
def swendsen_wang_update(self, samples, T, n_sweeps=1):
    """
    Swendsen-Wang cluster algorithm - fully parallel version.

    Eliminates critical slowing down: τ ~ O(1) vs τ ~ L^z for MH
    """
    B, C, H, W = samples.shape
    spins = samples[:, 0].clone()  # (B, H, W), values in {-1, +1}

    # Bond activation probability: p = 1 - exp(-2β)
    beta = 1.0 / T.view(B, 1, 1)
    p_bond = 1.0 - torch.exp(-2.0 * beta)

    for _ in range(n_sweeps):
        # === Step 1: Activate bonds between same-spin neighbors ===
        same_right = (spins == torch.roll(spins, -1, dims=2))
        same_down = (spins == torch.roll(spins, -1, dims=1))

        # Randomly activate bonds with probability p_bond
        bond_right = same_right & (torch.rand(B, H, W, device=device) < p_bond)
        bond_down = same_down & (torch.rand(B, H, W, device=device) < p_bond)

        # === Step 2: Find connected components via label propagation ===
        labels = torch.arange(H * W, device=device).view(1, H, W).expand(B, -1, -1)

        for _ in range(int(math.log2(H * W)) + 5):  # Convergence iterations
            # Propagate labels through active bonds
            labels_right = torch.roll(labels, 1, dims=2)
            labels_left = torch.roll(labels, -1, dims=2)
            labels_up = torch.roll(labels, 1, dims=1)
            labels_down = torch.roll(labels, -1, dims=1)

            # Take minimum label among connected neighbors
            new_labels = labels.clone()
            new_labels = torch.where(torch.roll(bond_right, 1, dims=2),
                                     torch.minimum(new_labels, labels_right), new_labels)
            new_labels = torch.where(bond_right,
                                     torch.minimum(new_labels, labels_left), new_labels)
            # ... (similar for vertical bonds)

            labels = new_labels

        # === Step 3: Flip each cluster with 50% probability ===
        unique_labels = labels.unique()
        flip_decisions = torch.rand(len(unique_labels)) < 0.5

        for i, label in enumerate(unique_labels):
            if flip_decisions[i]:
                mask = (labels == label)
                spins[mask] *= -1

    return spins.unsqueeze(1)  # (B, 1, H, W)
```

**성능 비교 (T = Tc = 2.27)**

```
┌──────────────────────┬────────────┬────────────┬─────────────┐
│ 알고리즘              │ Sweeps     │ 에너지 오차 │ 시간 (ms)   │
├──────────────────────┼────────────┼────────────┼─────────────┤
│ MH (checkerboard)    │ 200        │ -20.3      │ 108         │
│ Swendsen-Wang        │ 10         │ -1.3       │ 28          │
│ Swendsen-Wang        │ 50         │ -0.4       │ 140         │
└──────────────────────┴────────────┴────────────┴─────────────┘
```

#### 5.3.2 Metropolis-Hastings (fallback)

임계점에서 떨어진 온도에서는 MH도 효율적입니다. `use_cluster: false`로 설정하면 MH를 사용합니다.

**Checkerboard 업데이트**

2D 격자에서 효율적인 병렬 업데이트를 위해 checkerboard 패턴 사용:

```
  ┌───┬───┬───┬───┐
  │ W │ B │ W │ B │     W = White (even)
  ├───┼───┼───┼───┤     B = Black (odd)
  │ B │ W │ B │ W │
  ├───┼───┼───┼───┤     같은 색의 사이트들은
  │ W │ B │ W │ B │     이웃을 공유하지 않음
  ├───┼───┼───┼───┤     → 완전 병렬 업데이트 가능!
  │ B │ W │ B │ W │
  └───┴───┴───┴───┘
```

```python
# model_dfm.py - improve_samples_with_energy
def improve_samples_with_energy(self, samples, T, energy_fn, n_steps=10, use_checkerboard=True):
    """
    Metropolis-Hastings with checkerboard updates for parallel sampling.
    """
    B, C, H, W = samples.shape
    improved = samples.clone()
    beta = 1.0 / T.view(B, 1, 1)

    # Checkerboard 마스크 생성
    row_idx = torch.arange(H).view(1, H, 1)
    col_idx = torch.arange(W).view(1, 1, W)
    even_mask = ((row_idx + col_idx) % 2 == 0).expand(B, H, W)
    odd_mask = ~even_mask

    for _ in range(n_steps):
        for mask in [even_mask, odd_mask]:
            spins = improved[:, 0]

            # 이웃 스핀의 합 (local field)
            neighbors = (
                torch.roll(spins, 1, dims=1) +
                torch.roll(spins, -1, dims=1) +
                torch.roll(spins, 1, dims=2) +
                torch.roll(spins, -1, dims=2)
            )

            # 에너지 변화: ΔE = 2 * s * h
            delta_E = 2 * spins * neighbors

            # Metropolis 수락 확률
            accept_prob = torch.where(
                delta_E < 0,
                torch.ones_like(delta_E),
                torch.exp(-beta * delta_E)
            )

            accept = torch.rand_like(accept_prob) < accept_prob
            accept = accept & mask
            improved[:, 0] = torch.where(accept, -spins, spins)

    return improved
```

**효율성 분석:**

| 방식 | 한 sweep의 연산량 | 병렬화 |
|------|------------------|--------|
| Single-site MH | $L^2$ 순차 연산 | 불가 |
| Checkerboard MH | 2번의 병렬 연산 | GPU 최적화 |

### 5.4 Curriculum Learning

온도 범위를 점진적으로 확장하는 2단계 curriculum:

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
              0          75              175  300    epochs
                      phase1_epochs    +phase2_epochs
```

```python
# util_dfm.py:224-241 - get_curriculum_beta_range
def get_curriculum_beta_range(self):
    """Get current beta range based on curriculum phase."""
    if not self.curriculum_enabled:
        return self.model.beta_min, self.model.beta_max

    if self.current_epoch < self.phase1_epochs:
        # Phase 1: 고온만 학습 (β_max = 0.35, T_min ≈ 2.86 > T_c)
        return self.model.beta_min, self.phase1_beta_max
    else:
        # Phase 2: Cosine annealing으로 점진적 확장
        epochs_in_phase2 = self.current_epoch - self.phase1_epochs
        progress = min(1.0, epochs_in_phase2 / self.phase2_epochs)
        cos_progress = 0.5 * (1 - math.cos(math.pi * progress))

        effective_beta_max = (
            self.phase1_beta_max +
            (self.model.beta_max - self.phase1_beta_max) * cos_progress
        )
        return self.model.beta_min, effective_beta_max
```

**왜 Curriculum이 중요한가?**

| 온도 영역 | 특성 | 학습 난이도 |
|----------|------|------------|
| 고온 ($T > T_c$) | 무질서, 상관관계 약함 | 쉬움 |
| 임계점 ($T \approx T_c$) | 상전이, 장거리 상관관계 | 어려움 |
| 저온 ($T < T_c$) | 질서, 강한 상관관계 | 어려움 (mode collapse 위험) |

---

## 6. 샘플링 알고리즘

### ODE Integration (기본 방법)

Euler 방법으로 ODE를 적분하여 샘플 생성:

$$
x_{t+\Delta t} = x_t + v_t(x_t) \cdot \Delta t
$$

```python
# model_dfm.py:786-845 - sample_ode
def sample_ode(self, batch_size=None, T=None):
    """Generate samples via Euler ODE integration."""
    H, W = self.size

    # 1. Uniform 분포에서 초기화 (Dirichlet(1,1))
    alpha = torch.ones(batch_size, H, W, 2, device=self.device)
    x = torch.distributions.Dirichlet(alpha).sample()
    x = x.permute(0, 3, 1, 2)  # (B, 2, H, W)

    # 2. 시간 그리드 설정
    dt = (self.t_max - self.t_min) / self.num_flow_steps

    # 3. Euler integration
    for step in range(self.num_flow_steps):
        t = self.t_min + step * dt
        t_batch = torch.full((batch_size,), t, device=self.device)

        # Velocity 계산
        with torch.no_grad():
            v = self.velocity_field(x, t_batch, T)

        # Euler step
        x = x + v * dt

        # Simplex 제약 유지 (softmax를 통한 projection)
        x = F.softmax(torch.log(x.clamp(min=1e-8)), dim=1)

    # 4. 이산화: argmax 또는 sampling
    samples = (x[:, 1] > 0.5).float()  # p(+1) > 0.5면 +1
    samples = samples.unsqueeze(1)

    # 첫 번째 스핀 고정
    if self.fix_first is not None:
        samples[:, 0, 0, 0] = float(self.fix_first)

    # {0, 1} → {-1, +1} 변환
    samples = self.mapping(samples)  # 2*x - 1

    return samples  # (B, 1, H, W)
```

### 샘플링 과정 시각화

```
t = 0 (uniform)          t = t_max/2              t = t_max (converged)
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ ░░▓░░▓▓░▓░▓░░▓░ │     │ ▓▓▓▓░░▓▓▓▓▓░░▓▓ │     │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ▓░▓░▓░▓░▓░░▓░▓░ │     │ ▓▓▓▓░░░▓▓▓▓▓░▓▓ │     │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ░▓░░░▓░▓▓░▓▓░▓░ │ --> │ ▓▓▓░░░░▓▓▓▓▓▓▓▓ │ --> │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ░░▓░▓░░▓░▓░▓▓▓░ │     │ ▓▓░░░░░░▓▓▓▓▓▓▓ │     │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ▓░▓▓░░░▓░░░░░▓░ │     │ ░░░░░░░░░▓▓▓▓▓▓ │     │ ░░░░░░░░▓▓▓▓▓▓▓ │
└─────────────────┘     └─────────────────┘     └─────────────────┘
   50% random              도메인 형성              저온 평형상태
```

---

## 7. 검증 및 평가

### Onsager Exact Solution

2D Ising 모델의 정확한 분배함수:

```python
# vatd_exact_partition.py:77-174
def logZ(n, j, beta):
    """
    Exact log partition function using transfer matrix method.

    Z = 2 * (2 sinh(2βJ))^(N/2) * Σ [products of eigenvalues]
    """
    # 4개의 항 계산 (서로 다른 경계조건)
    # Term 1: even r, cosh
    # Term 2: even r, sinh
    # Term 3: odd r, cosh
    # Term 4: odd r, sinh

    result = (
        torch.logsumexp(torch.cat(terms), dim=0)
        - torch.log(torch.tensor(2.0))
        + 0.5 * n**2 * torch.log(2 * torch.sinh(2 * h(j, beta)))
    )
    return result
```

### Variational Free Energy

모델 검증을 위한 variational free energy:

$$
F_q = \mathbb{E}_{x \sim q}[\log q(x) + \beta E(x)] \geq -\log Z(\beta)
$$

등호는 $q(x) = p_\beta(x)$일 때 성립합니다.

```python
# util_dfm.py:344-408 - val_epoch
def val_epoch(self, energy_fn):
    """Validation with fixed beta values."""
    with torch.no_grad():
        # 모델에서 샘플 생성
        samples = self.model.sample(batch_size=total_size, T=T_expanded)

        # Log probability 계산
        log_prob = self.model.log_prob(samples, T=T_expanded)

        # 에너지 계산
        energy = energy_fn(samples)

        # Variational free energy: F = E[log q + βE]
        beta = (1.0 / T_expanded).unsqueeze(-1)
        loss_raw = log_prob + beta * energy

        # 정확한 값과 비교
        if self.exact_logz_values is not None:
            for i in range(num_beta):
                exact_logz = self.exact_logz_values[i]
                model_loss = val_dict[f"val_loss_beta_{i}"]
                error = model_loss + exact_logz  # 이상적으로 0
                val_dict[f"val_error_exact_beta_{i}"] = error
```

### Thermodynamic Integration

분배함수 추정을 위한 열역학적 적분:

$$
\log Z(\beta) = \log Z(0) - \int_0^\beta \langle E \rangle_{\beta'} d\beta'
$$

여기서 $\log Z(0) = (L^2 - 1) \ln 2$ (첫 스핀 고정 시)

```python
# partition_estimation.py:25-100
def thermodynamic_integration(model, energy_fn, beta_target, ...):
    """
    Compute log Z via thermodynamic integration.

    ∂/∂β log Z(β) = -⟨E⟩_β

    Integrating: log Z(β) = log Z(0) - ∫₀^β ⟨E⟩_β' dβ'
    """
    # log Z(0) = (N-1) * log(2) for first spin fixed
    log_Z0 = (N - 1) * math.log(2)

    # Beta grid
    betas = torch.linspace(0.001, beta_target, n_points)

    # Monte Carlo estimation of ⟨E⟩ at each β
    for beta in betas:
        T = 1.0 / beta.item()
        samples = model.sample(batch_size=n_samples, T=T_tensor)
        energies = energy_fn(samples)
        mean_energies.append(energies.mean().item())

    # Trapezoidal integration
    integral = np.trapz(mean_energies, betas.numpy())

    return log_Z0 - integral
```

---

## 8. 구현 세부사항

### 설정 파일 (configs/v0.14/ising_dfm.yaml)

```yaml
# 프로젝트 설정
project: Ising_DFM_v0.14
device: cuda:0
net: model_dfm.DiscreteFlowMatcher
optimizer: splus.SPlus
scheduler: hyperbolic_lr.ExpHyperbolicLR
epochs: 300
seeds: [42]

net_config:
  # 격자 설정
  size: 16
  fix_first: 1              # 첫 스핀 고정 (+1)

  # 온도 샘플링
  batch_size: 64            # 온도당 샘플 수
  num_beta: 4               # step당 온도 개수
  beta_min: 0.2             # T_max = 5
  beta_max: 1.0             # T_min = 1

  # Flow Matching
  num_flow_steps: 100       # ODE 적분 스텝
  t_max: 50.0               # 최대 적분 시간
  t_min: 0.01               # 최소 시간 (특이점 회피)
  time_dim: 64              # 시간 임베딩 차원

  # 학습 모드
  training_mode: energy_guided
  energy_weight: 1.0
  use_cluster: true         # Swendsen-Wang 클러스터 알고리즘 사용 (critical slowing down 해결!)
  n_clusters: 10            # SW sweeps 수 (10이면 Tc에서도 충분)
  mh_steps: 200             # MH sweeps (use_cluster=false일 때 fallback)
  mh_model_init_prob: 0.9   # Self-imitation 비율
  accumulation_steps: 8

  # Curriculum
  curriculum_enabled: true
  phase1_epochs: 75         # 고온 학습
  phase1_beta_max: 0.35     # T_min = 2.86 > T_c
  phase2_epochs: 100        # 점진적 확장

  # 모델 아키텍처
  hidden_channels: 64
  hidden_conv_layers: 5
  hidden_kernel_size: 3
  hidden_width: 128
  hidden_fc_layers: 2

optimizer_config:
  lr: 2.e-1
  eps: 1.e-10

scheduler_config:
  upper_bound: 350
  max_iter: 300
  infimum_lr: 1.e-6
```

### 실행 방법

```bash
# 기본 실행
python main.py --run_config configs/v0.14/ising_dfm.yaml --device cuda:0

# Hyperparameter optimization
python main.py --run_config configs/v0.14/ising_dfm.yaml \
               --optimize_config configs/optimize_template.yaml \
               --device cuda:0
```

### 모델 파라미터 수

| 컴포넌트 | 파라미터 |
|----------|----------|
| Time embedding MLP | 64 × 128 × 2 = 16,384 |
| First conv (7×7) | 3 × 128 × 49 = 18,816 |
| ResBlocks (×5) | 5 × (128×64 + 64×64×9 + 64×128) ≈ 250,000 |
| FC layers | 128 × 128 × 2 ≈ 32,768 |
| Final conv | 128 × 2 = 256 |
| **총계** | **~320,000** |

---

## 9. 참고문헌

### Flow Matching

1. **Discrete Flow Matching** - Gat et al., NeurIPS 2024
   - 이산 데이터에 대한 flow matching 일반화
   - Dirichlet probability path 제안

2. **Flow Matching for Generative Modeling** - Lipman et al., ICLR 2023
   - Continuous normalizing flows의 효율적 학습
   - Optimal transport 관점의 해석

### Ising Model

3. **Crystal Statistics I** - Onsager, Physical Review 1944
   - 2D Ising 모델의 정확한 해
   - 임계 온도 유도

4. **Statistical Mechanics: Theory and Molecular Simulation** - Tuckerman, Oxford 2010
   - Metropolis-Hastings 알고리즘
   - 열역학적 적분

### Cluster Algorithms

5. **Nonuniversal critical dynamics in Monte Carlo simulations** - Swendsen & Wang, Physical Review Letters 1987
   - Swendsen-Wang 클러스터 알고리즘 최초 제안
   - Critical slowing down 해결: τ ~ O(1) vs τ ~ L^z

6. **Collective Monte Carlo updating for spin systems** - Wolff, Physical Review Letters 1989
   - Wolff 단일 클러스터 알고리즘
   - 대형 격자에서 더 효율적

### 관련 구현

7. **PixelCNN** - van den Oord et al., ICML 2016
   - Autoregressive 이미지 모델링
   - Masked convolution

8. **VaTD (Variational Thermodynamic Divergence)** - 본 프로젝트의 이론적 기반
   - 온도 의존 생성 모델 학습

---

## 부록 A: 수학적 유도

### Dirichlet 분포의 평균과 분산

$Y \sim \text{Dir}(\alpha_1, \alpha_2)$일 때:

$$
\mathbb{E}[Y_i] = \frac{\alpha_i}{\alpha_0}, \quad \text{Var}[Y_i] = \frac{\alpha_i(\alpha_0 - \alpha_i)}{\alpha_0^2(\alpha_0 + 1)}
$$

여기서 $\alpha_0 = \sum_i \alpha_i$

### Ising 에너지 변화

단일 스핀 $s_i$를 뒤집을 때의 에너지 변화:

$$
\Delta E_i = E(\text{after}) - E(\text{before}) = 2 s_i \sum_{j \in \mathcal{N}(i)} s_j = 2 s_i h_i
$$

여기서 $h_i = \sum_{j \in \mathcal{N}(i)} s_j$는 local field

### Metropolis 수락 확률

$$
P_{\text{accept}} = \min\left(1, e^{-\beta \Delta E}\right) = \begin{cases}
1 & \text{if } \Delta E \leq 0 \\
e^{-\beta \Delta E} & \text{if } \Delta E > 0
\end{cases}
$$

---

## 부록 B: 코드 구조

```
VaTD_impl/
├── model_dfm.py           # DFM 모델 구현
│   ├── SinusoidalTimeEmbedding    # 시간 임베딩
│   ├── ResConv2D                  # ResNet 백본
│   └── DiscreteFlowMatcher
│       ├── sample_ode()           # ODE 적분 샘플링
│       ├── training_loss()        # 학습 손실 계산
│       ├── swendsen_wang_update() # SW 클러스터 알고리즘 (기본)
│       ├── improve_samples_with_energy()  # MH 샘플링 (fallback)
│       ├── apply_dirichlet_noise()        # Dirichlet 노이징
│       └── velocity_field()       # Velocity 예측
├── util_dfm.py            # 학습 유틸리티
│   ├── FlowMatchingTrainer
│   │   ├── use_cluster            # SW 사용 여부
│   │   ├── n_clusters             # SW sweeps 수
│   │   └── train_epoch()          # 에폭 학습
│   └── run()
├── vatd_exact_partition.py # Onsager exact solution
│   ├── logZ()
│   └── CRITICAL_TEMPERATURE
├── partition_estimation.py # 분배함수 추정
│   └── thermodynamic_integration()
├── main.py                # 실행 진입점
│   ├── create_ising_energy_fn()
│   └── main()
└── configs/v0.14/
    └── ising_dfm.yaml     # 설정 파일
```
