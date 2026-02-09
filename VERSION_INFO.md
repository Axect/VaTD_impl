# VaTD Version History

> Variational Thermodynamic Divergence for 2D Ising Model
> Development Period: November 24, 2025 ~ January 15, 2026

---

## Overview

| Phase | Versions | Period | Focus |
|-------|----------|--------|-------|
| **Exploration** | v0.0 ~ v0.7 | Nov 24 ~ Dec 9 | 기본 구조 탐색, 분산 감소 기법 |
| **Stabilization** | v0.8 ~ v0.12 | Dec 13 ~ Dec 26 | Curriculum Learning, 온도 스케일링 |
| **Architecture** | v0.13 ~ v0.14 | Dec 28 ~ Dec 29 | Sequential 훈련, Discrete Flow Matching |
| **Refinement** | v0.15 ~ v0.16 | Dec 30 ~ Jan 15 | Dilated Conv, Skip Connection, Hyper-Connection |

---

## Phase 1: Exploration (v0.0 ~ v0.7)

### v0.0 — Initial Implementation
**Date**: Nov 24, 2025
**Commit**: `f081896` (Initial commit) → `548ed8a` → ... → `74f40cc`

- DiscretePixelCNN 기본 구현 (Masked Convolution Type A/B)
- 2D Ising 에너지 함수 구현 (주기적 경계 조건)
- SPlus 옵티마이저 + ExpHyperbolicLR 스케줄러 도입
- Onsager 정확한 분할함수 솔루션 구현 (`vatd_exact_partition.py`)

**Config Baseline**:
```
Optimizer: SPlus (lr=1e-3)
Batch: 64 samples/temp, num_beta=8
Beta range: [0.1, 2.0] (T ∈ [0.5, 10])
Architecture: kernel_size=7, hidden_channels=64, 5 conv layers
Epochs: 10
```

### v0.1 — Batch Size & LR Tuning
**Date**: Dec 1, 2025 | **Commit**: `1c4bee6`

- batch_size를 64 → **256**으로 증가 (분산 감소)
- Learning rate를 1e-3 → **1e-1**로 증가 (SPlus에 적합)
- Optuna TPE 기반 하이퍼파라미터 탐색 도입

### v0.2 — Early Hyperparameter Search
**Date**: Dec 1, 2025 | **Commit**: `f36a84c`

- TPE(Tree-structured Parzen Estimator) 기반 탐색 고도화
- Best config 별도 저장 체계 수립

### v0.3 — Temperature Range Fix + RealNVP 실험
**Date**: Dec 2, 2025 | **Commit**: `26747b2`

- **Beta 범위 수정**: [0.1, 2.0] → **[0.2, 1.0]** (T ∈ [1, 5])
  - 물리적으로 의미있는 온도 범위로 축소 (Tc ≈ 2.269 포함)
- **RealNVP (Normalizing Flow)** 대안 모델 실험 (`model.RealNVP`)
  - Affine coupling layer 기반 Flow 모델
  - PixelCNN 대비 성능 미흡으로 이후 중단
- Extrapolation test 추가 (학습 범위 바깥 온도 검증)

### v0.4 — Detach & Beta-Weighted Loss
**Date**: Dec 8, 2025 | **Commit**: `15bff35`

- **Gradient detachment** 메커니즘 도입
  - REINFORCE에서 baseline 계산 시 gradient 분리
- **β-가중 손실함수**: 온도별 손실에 β 가중치 적용
  - 저온(높은 β)에 더 큰 가중치 → 질서상(ordered phase) 학습 강화
- RealNVP 병행 실험 계속

### v0.5 — Normalized by Pixels
**Date**: Dec 8, 2025 | **Commit**: `380dc49`

- **Pixel 정규화**: 손실함수를 격자 크기(16×16=256)로 나누어 정규화
  - 격자 크기에 독립적인 손실 스케일 확보
- β 의존성 제거 실험 (손실에서 β 가중치 제거)
- RealNVP 실험 계속 (최종적으로 PixelCNN이 우세)

### v0.6 — Gumbel-Softmax Exploration
**Date**: Dec 9, 2025 | **Commit**: `4821bbc`

- **Gumbel-Softmax 샘플링** 도입 시도
  - Discrete sampling의 미분 가능한 근사
  - REINFORCE 분산 감소 목적
- Normalizing Flow 제거 (Glow 포함), PixelCNN으로 통일
- 최종적으로 Gumbel-Softmax는 discrete 샘플 품질 이슈로 미채택

### v0.7 — RLOO Baseline
**Date**: Dec 9, 2025 | **Commit**: `564c450`

- **RLOO (Leave-One-Out) Baseline** 구현
  - REINFORCE gradient estimator의 분산 감소 핵심 기법
  - 각 샘플의 baseline을 나머지 샘플의 평균으로 계산
  - 기존 단순 평균 baseline 대비 유의미한 분산 감소
- 이후 모든 버전의 표준 baseline 기법으로 자리잡음

---

## Phase 2: Stabilization (v0.8 ~ v0.12)

### v0.8 — Curriculum Learning 도입
**Date**: Dec 13, 2025 | **Commit**: `d3f67f6`

- **2-Phase Curriculum Learning** — 이 프로젝트의 핵심 전략
  - **Phase 1**: 높은 온도(낮은 β)만 학습 → 무질서상(disordered)부터 시작
  - **Phase 2**: β_max를 점진적으로 확장 → 임계온도 영역 포함
  - 물리적 직관: 무질서상은 학습이 쉽고, 질서상은 long-range correlation 필요
- **옵티마이저 전환**: SPlus → **AdamW** (lr=1e-3, weight_decay=1e-4)
- Epochs 증가: 10 → **20**

**Config 변경점**:
```yaml
curriculum_enabled: true
curriculum_warmup_epochs: 75
curriculum_start_beta_max: 0.425  # T_min ≈ 2.35 (Tc 바로 위)
optimizer: AdamW (lr=1e-3)
```

### v0.9 — 3-Phase Curriculum + Tc Focus
**Date**: Dec 14, 2025 | **Commit**: `927ff13`

- **3-Phase Curriculum Learning**으로 확장
  - Phase 1 (50 epochs): 높은 온도만 (β_max = 0.4)
  - Phase 2 (100 epochs): 점진적 범위 확장
  - **Phase 3**: 50% 전체 범위 + 50% Tc 집중 샘플링
    - Tc 근방: β ∈ [0.38, 0.52] (T ∈ [1.92, 2.63])
- 임계온도 영역의 학습 난이도가 높아 명시적 집중 전략 시도
- `util.py`에 3-phase 커리큘럼 로직 대규모 추가 (+130 lines)

### v0.10 — FiLM Conditioning (실험적)
**Date**: Dec 22, 2025 | **Commit**: `9f261b1` (v1.0 → v0.10 리네이밍)

- **Temperature Embedding + FiLM (Feature-wise Linear Modulation)** 실험
  - β 기반 sinusoidal encoding → MLP → 각 레이어별 γ(T), β(T) 생성
  - `MaskedResConv2DFiLM`: FiLM 조건부 convolution
  - `DiscretePixelCNNFiLM`: 별도 모델 클래스
- 별도 브랜치(`claude/fix-critical-temperature`)에서 탐색
- 메인 브랜치에서는 다른 경로(v0.11)로 진행

> **Note**: v0.10 config 디렉토리는 이후 삭제됨. FiLM은 메인 경로에 미병합.

### v0.11 — Gradient Accumulation Training
**Date**: Dec 22, 2025 | **Commit**: `fc3de73`

- **Gradient Accumulation** 훈련 모드 도입 (`training_mode: accumulated`)
  - 레퍼런스 VaTD 구현체의 19 optimizer steps/epoch 방식 참고
  - `accumulation_steps: 16` → epoch당 16회 옵티마이저 업데이트
  - GPU 메모리 내에서 실질적 대규모 배치 효과
- Epochs 대폭 증가: 20 → **500**
- 모델 용량 확장: hidden_channels 64→**96**, conv_layers 5→**6**, hidden_kernel_size 3→**5**
- AMP(혼합 정밀도) 비활성화 (REINFORCE 수치 안정성)
- 3-Phase Curriculum 유지 (Phase 1: 100ep, Phase 2: 150ep)

**Config 변경점**:
```yaml
training_mode: accumulated
accumulation_steps: 16
epochs: 500
hidden_channels: 96, hidden_conv_layers: 6
```

### v0.12 — Temperature-Dependent Output Scaling
**Date**: Dec 26, 2025 | **Commit**: `1ccdf8b`

- **온도 의존 출력 스케일링** — 이 프로젝트의 또 다른 핵심 혁신
  - `scale = (T_ref / T)^power` (T_ref = 2.27 ≈ Tc)
  - 고온 (T=5): scale < 1 → logits 축소 → softmax ≈ 0.5 (무질서)
  - 저온 (T=1): scale > 1 → logits 확대 → softmax 날카롭게 (질서)
  - **핵심 통찰**: 고온에서 p≈0.5여야 하나, 255개 스핀에 걸친 미세한 bias 축적이 큰 오차 유발
- **Zero bias initialization**: 모든 Conv2D bias를 0으로 초기화
  - 초기 예측이 정확히 p=0.5 (대칭)
- **3-Phase → 2-Phase Curriculum** 단순화 (Tc focus 제거)
  - Tc focus가 오히려 고온 샘플 부족 유발 → 제거

**Config 변경점**:
```yaml
logit_temp_scale: true
temp_ref: 2.27        # Tc
temp_scale_power: 1.0 # v0.12.1에서 0.5→1.0 증가
# 3-Phase의 tc_focus 관련 설정 제거
```

---

## Phase 3: Architecture (v0.13 ~ v0.14)

### v0.13 — Sequential Backward Training + OneCycleLR
**Date**: Dec 28, 2025 | **Commit**: `0790146`

- **Sequential Training Mode** (`training_mode: sequential`)
  - 온도별로 순차적으로 forward/backward 수행
  - 온도 간 "crosstalk" (gradient 간섭) 방지
  - `accumulated` 모드 대비 안정적 학습
- `accumulation_steps: 64`로 증가 (더 많은 optimizer step/epoch)
- **OneCycleLR 스케줄러** 도입 (v0.13.2)
  - Warmup (30%) → Peak LR → Cosine Annealing
  - max_lr=3e-3, initial=3e-4, final=3e-6
  - REINFORCE의 noisy gradient에 적합한 3단계 LR 전략
- **MCMC Guided Training** 옵션 추가 (Swendsen-Wang)
  - `mcmc_freq: 5` (5 epoch마다 MCMC correction)
  - PixelCNN의 주요 참조 버전으로 자리잡음

**주요 Config Variants**:
| Config | Scheduler | MCMC | 특징 |
|--------|-----------|------|------|
| `ising_pixelcnn_accumulated.yaml` | ExpHyperbolicLR | Off | 기본 sequential |
| `ising_pixelcnn_onecycle.yaml` | **OneCycleLR** | Off | 권장 기본값 |
| `ising_pixelcnn_mcmc.yaml` | ExpHyperbolicLR | On | SW correction |

### v0.14 — Discrete Flow Matching (DFM)
**Date**: Dec 29, 2025 | **Commit**: `aa9f91c`

- **완전히 새로운 모델 아키텍처**: `DiscreteFlowMatcher`
  - Autoregressive PixelCNN을 대체하는 Flow 기반 모델
  - **새 파일**: `model_dfm.py` (749 lines), `util_dfm.py` (681 lines), `partition_estimation.py` (466 lines)
- **핵심 차이점 vs PixelCNN**:
  - **병렬 샘플링**: Euler integration으로 ~3-5x 빠른 샘플링
  - **Dirichlet Probability Path**: Discrete 분포를 위한 노이징 경로
  - **Energy-Guided Training**: MH-improved 타겟 + velocity matching
  - **마스킹 불필요**: Autoregressive 제약 없음
- **Dual Trainer 아키텍처**:
  - `main.py`가 모델 타입 감지 후 라우팅
  - PixelCNN → `util.run()` → `Trainer`
  - DFM → `util_dfm.run()` → `FlowMatchingTrainer`
- SPlus 옵티마이저 복귀 (DFM에 적합)

**Config 핵심**:
```yaml
net: model_dfm.DiscreteFlowMatcher
training_mode: energy_guided
num_flow_steps: 100    # Euler integration steps
t_max: 50.0            # Integration time (Dirichlet convergence ~96%)
use_cluster: true      # Swendsen-Wang equilibration
optimizer: SPlus (lr=2e-1)
```

**Variants**: `ising_dfm.yaml` (표준), `ising_dfm_large.yaml` (고용량)

---

## Phase 4: Refinement (v0.15 ~ v0.16)

### v0.15 — Dilated Convolutions + MCMC Ablation
**Date**: Dec 30, 2025 | **Commit**: `bcb4504`

- **Dilated Convolution** 도입 — Receptive Field의 지수적 확장
  - Dilation pattern: [1, 2, 4, 8]
  - RF: 7×7 → **61×61** (kernel_size 증가 없이)
  - `hidden_conv_layers: 8` (v0.13의 6에서 증가)
  - `hidden_kernel_size: 3` (5에서 축소 — dilation이 RF 담당)
  - 임계온도의 long-range correlation 학습에 핵심
- **두 가지 Variant로 Ablation Study**:

| Config | MCMC | 목적 |
|--------|------|------|
| `ising_pixelcnn_improved.yaml` | **On** (매 epoch, 20 SW sweeps) | Dilated + MCMC 시너지 |
| `ising_pixelcnn_dilated_only.yaml` | **Off** | **Dilated Conv 단독 효과 측정** |

- Pure REINFORCE(MCMC 없음)로도 Dilated Conv만으로 Tc 학습 개선 확인
- **XY 모델 확장** (`xy_cfm.yaml`): Continuous Flow Matching for XY model
  - Von Mises 분포 기반, BKT 전이 (T ≈ 0.89) 연구용

### v0.16 — Compact Architecture + Experimental Variants
**Date**: Jan 6 ~ 15, 2026 | **Commit**: `f4459cd` ~ `33371de`

- **Compact Dilated ResNet**: v0.15 대비 효율 개선
  - `hidden_conv_layers: 4` (8에서 축소, **~46% 파라미터 감소**)
  - RF: 61×61 → **37×37** (여전히 16×16 격자보다 큼)
  - `accumulation_steps: 16` (64에서 축소)
  - Epochs: 500 → **250**
- **Multi-Scale Skip Connection**: `skip_connection_indices: [1, 3]`
  - Layer 1 (d=1,2 이후): 로컬 피처 (RF≈11)
  - Layer 3 (d=1,2,4,8 이후): 글로벌 피처 (RF≈37)
- **Explicit Dilation Pattern**: `dilation_pattern: [1, 2, 4, 8]` 명시적 선언

**실험적 Variants**:

| Config | 핵심 아이디어 | 설명 |
|--------|-------------|------|
| `ising_pixelcnn_compact.yaml` | Baseline Compact | 4-layer dilated + skip connection |
| `ising_pixelcnn_diagonal.yaml` | **Diagonal Scan** | Anti-diagonal 순서: 31 step (vs 256), ~8x 샘플링 속도 |
| `ising_pixelcnn_hilbert.yaml` | **Hilbert Curve** | 공간 채움 곡선으로 2D locality 보존 |
| `ising_pixelcnn_mhc.yaml` | **mHC (Manifold HC)** | Sinkhorn-Knopp 정규화, Doubly Stochastic 믹싱 |
| `ising_pixelcnn_mhc2.yaml` | **Dynamic mHC** | 배치별 동적 H 행렬 학습 |
| `ising_pixelcnn_hc.yaml` | **HC (Unconstrained)** | Manifold 제약 없는 Hyper-Connection |
| `ising_pixelcnn_muon.yaml` | **Muon Optimizer** | Newton-Schulz 직교화, 2D weights용 Muon + 1D용 AdamW |

> **Hyper-Connection (HC/mHC)**: DeepSeek-AI (2025) 참조.
> Skip connection의 융합 가중치를 학습하는 기법으로,
> mHC는 doubly stochastic constraint를 통해 신호 폭발/소멸 방지.

---

## Version-by-Version Config Evolution

### Architecture

| Version | Layers | Channels | Kernel | Dilation | Skip | RF |
|---------|--------|----------|--------|----------|------|----|
| v0.0~v0.7 | 5 | 64 | k7/h3 | - | - | ~15×15 |
| v0.8~v0.9 | 5 | 64 | k7/h3 | - | - | ~15×15 |
| v0.11~v0.13 | **6** | **96** | k7/**h5** | - | - | ~27×27 |
| v0.14 (DFM) | 5 | 64 | h3 | - | - | N/A |
| v0.15 | **8** | 96 | k7/**h3** | **[1,2,4,8]** | - | **~61×61** |
| v0.16 | **4** | 96 | k7/h3 | [1,2,4,8] | **[1,3]** | **~37×37** |

### Training Strategy

| Version | Mode | Accum Steps | Curriculum | Optimizer |
|---------|------|-------------|------------|-----------|
| v0.0~v0.7 | standard | - | None | SPlus |
| v0.8~v0.9 | standard | - | 2-Phase / 3-Phase | AdamW |
| v0.11 | **accumulated** | 16 | 3-Phase | AdamW |
| v0.12 | accumulated | 16 | **2-Phase** | AdamW |
| v0.13 | **sequential** | **64** | 2-Phase | AdamW |
| v0.14 | **energy_guided** | 8 | 2-Phase | SPlus |
| v0.15 | sequential | 64 | 2-Phase | AdamW |
| v0.16 | sequential | **16** | 2-Phase | AdamW / Muon |

### Key Feature Timeline

| Feature | Introduced | Description |
|---------|-----------|-------------|
| REINFORCE + RLOO | v0.7 | Leave-One-Out baseline for variance reduction |
| Curriculum Learning | v0.8 | Phase 1: high-T → Phase 2: expand |
| Tc Focus Sampling | v0.9 | Phase 3: 50% Tc region (later removed) |
| Gradient Accumulation | v0.11 | Multiple optimizer steps per epoch |
| Temperature Scaling | v0.12 | `scale = (T_ref/T)^power` for logits |
| Sequential Training | v0.13 | Per-temperature forward/backward |
| OneCycleLR | v0.13 | Warmup → Peak → Cosine Annealing |
| Discrete Flow Matching | v0.14 | Parallel sampling, energy-guided |
| Dilated Convolution | v0.15 | Exponential RF expansion |
| Skip Connections | v0.16 | Multi-scale feature fusion |
| Hyper-Connections | v0.16 | Learned mixing weights (HC/mHC) |

---

## Physics Constants (All Versions)

```
2D Ising Model: E = -J Σ_{<i,j>} s_i · s_j  (J = 1)
Lattice:        16 × 16, Periodic Boundary Conditions
Temperature:    T ∈ [1, 5]  →  β ∈ [0.2, 1.0]
Critical Temp:  Tc ≈ 2.269  →  βc ≈ 0.441
Symmetry:       fix_first = 1 (첫 번째 스핀 고정으로 Z₂ 대칭 파괴)
Validation:     Onsager's exact partition function
```
