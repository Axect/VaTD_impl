# VaTD — Variational Thermodynamic Divergence

**Neural networks discover the operator content of conformal field theories**

> 2D 격자 스핀 시스템(Ising, Potts)의 Boltzmann 분포를 변분 자유에너지 최소화로 학습하여,
> 임계점에서의 CFT 구조를 자동으로 발견하는 연구 프로젝트.

---

## 1. 연구 개요

VaTD는 자기회귀 심층 생성 모델(PixelCNN, Transformer, Mamba 등)을 사용하여 2D 스핀 격자 시스템의 Boltzmann 분포를 학습한다. REINFORCE 기반의 변분 자유에너지 최소화를 통해 훈련된 모델의 내부 표현을 분석함으로써, 다음 세 가지 핵심 발견을 검증한다:

| # | 발견 | 물리적 의미 |
|---|------|------------|
| **Result 1** | 임계 온도 Tc에서 활성화 유효 랭크(eRank)의 최솟값이 CFT 관련 연산자 수와 일치 | Ising: 3, 3-Potts: 6, 4-Potts: 8 |
| **Result 2** | 은닉층 간 교차 공분산 SVD 엔트로피가 Calabrese-Cardy 공식을 따름 | 중심 전하 c 추출 |
| **Result 3** | eRank ~ \|T−Tc\|^{−φ} 보편적 멱법칙 | 상관 길이 발산과 대응 |

이 연구는 **기계학습**, **통계 물리학**, **등각장론(CFT)** 을 연결하며, 신경망이 임계 현상의 저차원(low-rank) 구조를 자연스럽게 발견함을 보여준다.

---

## 2. 물리적 배경

### 2.1 에너지 함수

**Ising 모델 (q = 2)**
```
E(s) = −J Σ_{⟨i,j⟩} sᵢsⱼ,    sᵢ ∈ {−1, +1},   J = 1
```

**q-state Potts 모델 (q ≥ 2)**
```
E(s) = −J Σ_{⟨i,j⟩} δ(sᵢ, sⱼ),    sᵢ ∈ {0, 1, …, q−1}
```

격자: 16×16, 주기적 경계 조건 (periodic boundary conditions)

### 2.2 임계 온도 & CFT 데이터

| 모델 | q | Tc | 중심 전하 c | 관련 연산자 수 |
|------|---|-----|-----------|-------------|
| Ising | 2 | 2/ln(1+√2) ≈ 2.269 | 1/2 | 3 |
| 3-Potts | 3 | 1/ln(1+√3) ≈ 0.995 | 4/5 | 6 |
| 4-Potts | 4 | 1/ln(1+√4) ≈ 0.910 | 1 | 8 |

### 2.3 저차원 가설 (Low-Rank Hypothesis)

Thibeault et al. (Nature Physics, 2024)의 "복잡계의 저차원 가설"을 2D Ising 모델에서 검증:

- **활성화 랭크 최소**: Tc에서 소수의 스케일링 장이 지배 → 내부 표현이 저차원
- **압축 민감도 최대**: Tc에서 가중치 SVD 절단 시 최대 성능 저하 → 가중치 정밀도가 가장 중요
- **"압축 역설"**: 활성화는 가장 단순하나(저차원), 가중치는 가장 정밀해야 함

---

## 3. 모델 아키텍처

### 3.1 DiscretePixelCNN (주력 모델)

```
model.py — v0.0~v0.19
```

- 마스크 합성곱(TypeA/TypeB)을 사용한 자기회귀 픽셀 생성
- 팽창(Dilated) ResNet 백본 (수용장 = 37 for compact)
- **Hyper-Connections (HC/mHC)**: 잔차 블록 간 doubly-stochastic 융합 레이어
- **스캔 경로**: 행 우선, 대각선(~8배 샘플링 가속), 힐버트 곡선
- **온도 의존 로짓 스케일링**: scale = (T_ref / T)^power
- `category` 매개변수로 Ising(2) / Potts(q>2) 전환

### 3.2 LatticeGPT (Transformer)

```
model_transformer.py — v0.20~v0.21
```

- 16×16 → 256-토큰 시퀀스, 인과적(causal) Transformer
- AdaLN 온도 조건화, FlashAttention + KV-cache
- 주기적 2D 상대 위치 바이어스
- 아키텍처 독립성 검증 목적

### 3.3 LatticeMamba (SSM)

```
model_mamba.py — v0.22
```

- Mamba 상태 공간 모델 (d_state = 32)
- AdaLN 블록별 온도 조건화
- 대형 격자(L=32/64) 확장을 위한 O(L²) 스케일링

### 3.4 LatticeFLA / GLA

```
model_fla.py — v0.23
```

- Fusion-Linearity-Attention 및 Gated Linear Attention 변형
- 최신 실험 아키텍처

---

## 4. 훈련 파이프라인

### 4.1 핵심 알고리즘

```
변분 자유에너지 최소화:
  F[q] = ⟨E(s)⟩_q − T·H[q]  ≥  F_exact
  ∇F = REINFORCE with RLOO baseline
```

- **손실 함수**: 에너지 기댓값 − 온도 × 엔트로피 (정확한 분배함수 대비 KL 발산과 동치)
- **그래디언트 추정**: REINFORCE + RLOO(Leave-One-Out) 기준선
- **Ising**: Onsager 정확 해가 있어 ground-truth KL 계산 가능
- **Potts (q>2)**: 유한 격자 정확 해 없음 → REINFORCE만 사용

### 4.2 훈련 모드

| 모드 | 설명 |
|------|------|
| `standard` | 모든 β를 매 에폭마다 샘플링 |
| `accumulated` | 그래디언트 누적 (배치 분할) |
| `sequential` | 온도별 순차 훈련 |
| `energy_guided` | DFM 스타일 에너지 가이드 |

### 4.3 커리큘럼 학습

```
Phase 1 (고온부): T ≥ 1/β_max  →  상자성 영역에서 빠른 수렴
Phase 2 (전체):   T ∈ [T_min, T_max]  →  임계점 포함 전 범위
```

### 4.4 실행 방법

```bash
# 환경 설정
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 훈련
python main.py --run_config configs/v0.19/potts3_pixelcnn.yaml --device cuda:0

# 하이퍼파라미터 최적화 (Optuna)
python main.py --run_config <run>.yaml --optimize_config <opt>.yaml --device cuda:0
```

---

## 5. 분석 실험

### 5.1 활성화 랭크 분석 (Experiment 1)

```bash
python analyze_rank.py --project <project> --group <group> --seed 42
python analyze_rank.py --critical    # Tc 근처 60점 밀도 그리드
python analyze_rank.py --replot <csv>
```

- 잔차 블록의 출력 활성화에 대해 SVD 수행
- 9가지 랭크 메트릭 계산 (eRank, Stable Rank, Participation Ratio 등)
- **핵심 결과**: 채널 eRank가 고온(~25–30)에서 Tc(~3–5)로 급감

### 5.2 가중치 압축 테스트 (Experiment 2)

```bash
python analyze_compression.py --project <project> --group <group> --seed 42
python analyze_compression.py --per_layer  # 블록별 민감도
```

- Conv2d 가중치를 SVD로 절단 (90%, 75%, 50%, 25%, 10%)
- 성능 저하 D(T,k) = KL(q_full ∥ q_k) / N 측정
- **핵심 결과**: D(T,k)가 Tc에서 첨예하게 피크, 비열 Cv와 비례

### 5.3 교차 모델 비교 (PRL Result 1)

```bash
python analyze_cross_model.py
```

- Ising, 3-Potts, 4-Potts 모델의 eRank_min을 비교
- CFT 관련 연산자 수와의 일치 검증

### 5.4 신경 얽힘 엔트로피 (PRL Result 2)

```bash
python analyze_entanglement.py
```

- 은닉층 간 교차 공분산 SVD → 엔트로피 계산
- Calabrese-Cardy 공식에 대한 피팅으로 중심 전하 c 추출

### 5.5 랭크-지수 멱법칙 (PRL Result 3)

```bash
python analyze_rank_exponent.py
```

- eRank ~ |T−Tc|^{−φ} 피팅
- 상관 길이 ξ ~ |T−Tc|^{−ν}와의 대응

---

## 6. 프로젝트 구조

```
VaTD_impl/
├── 핵심 모델
│   ├── model.py              # DiscretePixelCNN (주력)
│   ├── model_transformer.py  # LatticeGPT
│   ├── model_mamba.py        # LatticeMamba
│   ├── model_fla.py          # LatticeFLA / GLA
│   └── model_dfm.py          # DiscreteFlowMatcher (미사용)
│
├── 훈련 & 설정
│   ├── main.py               # 진입점: 모델 디스패치 & 에너지 설정
│   ├── util.py               # Trainer (REINFORCE, 커리큘럼)
│   ├── util_dfm.py           # FlowMatchingTrainer
│   ├── config.py             # RunConfig / OptimizeConfig
│   └── configs/              # 68개 YAML 설정 (v0.0~v0.23)
│
├── 물리 & 정확 해
│   ├── ising.py              # Ising 에너지 + MCMC (SW, MH, Wolff)
│   ├── potts.py              # Potts 에너지 + MCMC
│   ├── vatd_exact_partition.py   # Onsager 정확 분배함수
│   └── potts_exact_partition.py  # Potts Tc, CFT 데이터
│
├── 분석 (11개 스크립트)
│   ├── analyze_rank.py            # 활성화 랭크 분석
│   ├── analyze_compression.py     # 가중치 압축 테스트
│   ├── analyze_cross_model.py     # 교차 모델 비교 (PRL R1)
│   ├── analyze_entanglement.py    # 신경 얽힘 엔트로피 (PRL R2)
│   ├── analyze_rank_exponent.py   # 랭크-지수 멱법칙 (PRL R3)
│   ├── analyze_prl_paper.py       # PRL 출판용 통합 그림
│   ├── analyze.py                 # 대화형 열역학 분석 UI
│   └── ...                        # 기타 진단 스크립트
│
├── 최적화 & 유틸리티
│   ├── splus.py              # SPlus 옵티마이저
│   ├── muon.py               # MuonWithAdamW
│   ├── hyperbolic_lr.py      # ExpHyperbolicLR 스케줄러
│   ├── pruner.py             # Optuna 프루너
│   └── unified_rank_metrics.py   # 9가지 랭크 메트릭
│
├── 문서
│   ├── docs/LOW_RANK.md              # 저차원 가설 배경
│   └── docs/LOW_RANK_COMPANION.md    # 한국어 확장판
│
└── 출력 (gitignored)
    ├── runs/          # 저장된 모델 & 결과
    ├── figs/          # 생성된 그림
    └── checkpoints/   # 훈련 체크포인트
```

---

## 7. 기술 스택

| 범주 | 도구 |
|------|------|
| 언어 | Python 3, PyTorch (CUDA 12.x) |
| 패키지 관리 | uv |
| 실험 추적 | Weights & Biases (wandb) |
| 설정 | YAML + 타입 데이터클래스 |
| HPO | Optuna (TPE 샘플러, 커스텀 프루너) |
| 시각화 | matplotlib + scienceplots |
| SSM | mambapy 1.2.0 |

---

## 8. 버전 히스토리

| 버전 | 시기 | 핵심 변경 |
|------|------|----------|
| v0.0–v0.9 | 2024.11 | 기본 PixelCNN, REINFORCE, 커리큘럼 학습 |
| v0.10–v0.13 | 2024.12 | 순차 훈련 모드, OneCycleLR |
| v0.14 | 2024.12 | DiscreteFlowMatcher (DFM) 도입 |
| v0.15 | 2025.01 | 팽창 합성곱, MCMC 가이드 |
| v0.16 | 2025.01 | Compact ResNet + Hyper-Connections |
| v0.18–v0.19 | 2025.02 | A100 최적화, q-state Potts 지원 |
| v0.20–v0.21 | 2025.02 | LatticeGPT (Transformer) |
| v0.22 | 2025.02 | LatticeMamba (SSM) |
| v0.23 | 2025.03 | LatticeFLA / GLA |

---

## 9. 주요 참고 문헌

1. Thibeault, V. et al., "The low-rank hypothesis of complex systems," *Nature Physics* **20**, 294–302 (2024)
2. Roy, O. & Vetterli, M., "The effective rank," *EUSIPCO* (2007)
3. Gavish, M. & Donoho, D., "The optimal hard threshold for singular values is 4/√3," *IEEE Trans. Inf. Theory* (2014)
4. Calabrese, P. & Cardy, J., "Entanglement entropy and quantum field theory," *J. Stat. Mech.* (2004)
5. Di Francesco, P. et al., *Conformal Field Theory*, Springer (1997)
