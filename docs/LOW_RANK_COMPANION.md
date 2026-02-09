# Low-Rank Hypothesis 해설서: 배경지식 총정리

> **목적**: `LOW_RANK.md` 문서를 읽기 위해 필요한 모든 배경지식을 정리한 해설서.
>
> **대상 독자**:
> - **현상론 물리학 배경의 ML-oriented 대학원생**: 열역학·통계역학 수준의 물리 배경과 딥러닝 실무 경험을 갖추고 있으며, 물리와 ML의 교차점에 관심이 있는 연구자
> - **머신러닝 연구자**: 딥러닝, 최적화, 생성 모델에 익숙하지만, 통계물리학적 배경을 보충하고 싶은 독자
> - **물리학 연구자**: QFT/통계역학에 익숙하지만, 신경망 내부의 물리적 구조에 관심이 있는 독자
>
> 각 섹션에는 **물리 → ML 대응**을 명시하여, 물리 개념이 신경망 학습과 어떻게 연결되는지 직관적으로 이해할 수 있도록 구성했습니다.

---

## 목차

1. [통계역학의 기본 구조: 경로적분과의 대응](#1-통계역학의-기본-구조-경로적분과의-대응)
2. [2D Ising 모델: 가장 단순한 상전이 시스템](#2-2d-ising-모델-가장-단순한-상전이-시스템)
3. [상전이와 임계현상](#3-상전이와-임계현상)
4. [Onsager 엄밀해와 그 역사적 의미](#4-onsager-엄밀해와-그-역사적-의미)
5. [재규격화군(RG): Wilson의 블록 스핀과 HEP의 RG](#5-재규격화군rg-wilson의-블록-스핀과-hep의-rg)
6. [2D Ising CFT: c = 1/2 최소 모델](#6-2d-ising-cft-c--12-최소-모델)
7. [Low-Rank Hypothesis: Nature Physics (2024)](#7-low-rank-hypothesis-nature-physics-2024)
8. [SVD와 랭크 지표들](#8-svd와-랭크-지표들)
9. [자기회귀 모델과 PixelCNN](#9-자기회귀-모델과-pixelcnn)
10. [변분 열역학 발산(VaTD)과 자유에너지 최소화](#10-변분-열역학-발산vatd과-자유에너지-최소화)
11. [REINFORCE와 RLOO 기울기 추정](#11-reinforce와-rloo-기울기-추정)
12. [핵심 연결: 왜 임계점에서 Low-Rank인가?](#12-핵심-연결-왜-임계점에서-low-rank인가)
13. [용어 대조표](#13-용어-대조표)
14. [참고문헌](#14-참고문헌)

---

## 1. 통계역학의 기본 구조: 경로적분과의 대응

> **HEP 물리학자를 위한 핵심 메시지**: 통계역학은 유클리드 장론(Euclidean QFT)과 **수학적으로 동치**입니다.

### 1.1 분배함수 = 유클리드 경로적분

고에너지 물리학에서 익숙한 경로적분을 떠올려 봅시다:

$$
Z[J] = \int \mathcal{D}\phi \, e^{-S_E[\phi]}
$$

여기서 $S_E$는 유클리드 작용(Euclidean action)입니다. 통계역학의 분배함수는 정확히 이 구조를 가집니다:

$$
Z(\beta) = \sum_{\{s\}} e^{-\beta H(s)} = \text{Tr}\left(e^{-\beta H}\right)
$$

**대응 관계:**

| 유클리드 장론 (HEP) | 통계역학 |
|---|---|
| 유클리드 작용 $S_E[\phi]$ | $\beta H(\mathbf{s})$ (역온도 $\times$ 해밀토니안) |
| 장 배위 $\phi(x)$ | 스핀 배위 $\{s_i\}$ |
| 경로적분 $\int \mathcal{D}\phi$ | 배위 합 $\sum_{\{s\}}$ |
| 소스항 $J\phi$ | 외부 자기장 $h \sum_i s_i$ |
| 연결 상관함수 $\langle \phi(x)\phi(y) \rangle_c$ | 스핀-스핀 상관함수 $\langle s_i s_j \rangle - \langle s_i \rangle \langle s_j \rangle$ |
| 진공에너지 $-\ln Z$ | 자유에너지 $F = -T \ln Z$ |

이 대응은 **Wick 회전** ($t \to -i\tau$)에 의해 성립합니다. HEP에서 민코프스키 시공간의 장론을 유클리드 공간으로 해석적 연속하면, 그 수학적 구조가 통계역학의 분배함수와 정확히 일치합니다.

#### ML 관점에서 본 통계역학

위의 경로적분 대응은 HEP 배경이 있는 독자를 위한 것입니다. ML 배경의 독자를 위해, 통계역학의 핵심 개념을 딥러닝 용어로 재해석합니다:

| 통계역학 | 머신러닝 대응 | 직관 |
|---|---|---|
| Boltzmann 분포 $p(\mathbf{x}) = e^{-\beta E(\mathbf{x})}/Z$ | **Softmax with $2^N$ classes** | $N=256$ 스핀의 모든 가능한 배위에 대한 softmax. 각 배위의 "로짓"이 $-\beta E(\mathbf{x})$ |
| 분배함수 $Z = \sum_{\mathbf{x}} e^{-\beta E(\mathbf{x})}$ | **Softmax의 분모** (정규화 상수) | LogSumExp: $\log Z = \text{LSE}(-\beta E(\mathbf{x}))$. $2^{256}$개 항의 합이므로 직접 계산 불가 |
| 에너지 함수 $E(\mathbf{x})$ | **Unnormalized negative log-probability** | EBM(에너지 기반 모델)의 에너지와 동일: $E(\mathbf{x}) = -\log \tilde{p}(\mathbf{x})$ |
| 역온도 $\beta = 1/T$ | **Softmax temperature의 역수** | LLM의 `temperature` 파라미터와 동일한 역할. $\beta \uparrow$ → 분포가 날카로워짐 (greedy), $\beta \downarrow$ → 균일해짐 (random) |
| 자유에너지 최소화 | **ELBO 최대화** (부호 반전) | $F_q = F_{\text{exact}} + T \cdot D_{\text{KL}}(q \| p)$: VAE의 ELBO와 동일한 구조 |

> **핵심 직관**: Boltzmann 분포는 "에너지가 낮은 상태일수록 높은 확률을 부여하는 softmax"이고, 분배함수 $Z$는 이 softmax의 분모입니다. VaTD 훈련은 이 거대한 softmax를 자기회귀 모델로 근사하는 과정입니다.

### 1.2 자유에너지와 열역학적 포텐셜

> **ML 전문가를 위한 설명**: 자유에너지는 "에너지 최소화"와 "엔트로피 최대화" 사이의 균형을 정량화하는 손실함수입니다.

**Helmholtz 자유에너지**는 통계역학의 중심 양입니다:

$$
F = -T \ln Z = \langle E \rangle - T S
$$

여기서 $S = -\sum_{\mathbf{x}} p(\mathbf{x}) \ln p(\mathbf{x})$는 열역학적 엔트로피입니다. 이 공식은 두 가지 경쟁하는 힘을 포착합니다:

- **낮은 온도** ($T \to 0$): 에너지 $\langle E \rangle$ 항이 지배 → 시스템은 에너지가 가장 낮은 상태(바닥 상태)를 선호 → **질서 상**
- **높은 온도** ($T \to \infty$): 엔트로피 $-TS$ 항이 지배 → 시스템은 가능한 한 많은 상태에 퍼짐 → **무질서 상**
- **임계 온도** ($T = T_c$): 두 힘이 정확히 균형을 이루며, 스케일 불변성이 출현

**VAE의 ELBO와의 대응**: 자유에너지의 변분 원리를 VAE의 ELBO(Evidence Lower Bound)와 나란히 비교하면 구조적 동치성이 명확해집니다:

$$
\underbrace{-\text{ELBO}}_{\text{VAE 손실}} = \underbrace{\mathbb{E}_{q_\phi}[-\log p(x|z)]}_{\text{재구성 오차}} + \underbrace{D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\text{사전분포 정칙화}}
$$

$$
\underbrace{F_q}_{\text{변분 자유에너지}} = \underbrace{\langle E \rangle_q}_{\text{평균 에너지} \leftrightarrow \text{재구성 오차}} + \underbrace{T \langle \log q \rangle_q}_{\text{음의 엔트로피} \leftrightarrow \text{KL 정칙화}}
$$

두 경우 모두 (1) 데이터/물리에 맞는 방향과 (2) 분포를 퍼뜨리는 정칙화 사이의 트레이드오프이며, 온도 $T$가 이 균형을 제어합니다. VAE에서 KL 가중치를 조절하는 $\beta$-VAE의 $\beta$는 물리의 역온도 $\beta = 1/T$와 정확히 같은 역할을 합니다.

### 1.3 Boltzmann 분포

주어진 온도 $T$에서 배위 $\mathbf{x}$가 나타날 확률은:

$$
p(\mathbf{x} \mid T) = \frac{e^{-\beta H(\mathbf{x})}}{Z(\beta)}, \quad \beta = \frac{1}{T}
$$

이것은 기계학습에서 **에너지 기반 모델(EBM)**의 softmax 분포와 동일한 형태입니다. 분모 $Z(\beta)$는 정규화 상수이며, 2D Ising 모델의 경우 $2^N$개의 항을 합산해야 하므로 ($L = 16$이면 $2^{256}$개), 직접 계산은 불가능합니다.

**Softmax temperature와의 관계**: LLM이나 분류 모델에서 사용하는 temperature scaling은 물리학의 온도와 동일한 수학적 구조입니다:

$$
\text{softmax}_T(z_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}} \quad \longleftrightarrow \quad p(\mathbf{x} | T) = \frac{e^{-E(\mathbf{x}) / T}}{Z(T)}
$$

로짓 $z_i$가 음의 에너지 $-E(\mathbf{x})$에 대응합니다. 본 프로젝트의 `logit_temp_scale: true` 설정은 PixelCNN의 출력 로짓에 $\text{scale} = 1/T$를 곱하는데, 이것은 정확히 물리적 Boltzmann 분포의 $\beta = 1/T$ 스케일링을 구현한 것입니다. 고온($T \gg 1$)에서 출력이 $P \approx 0.5$에 가까워지고, 저온($T \ll 1$)에서 날카로워지는 것은 softmax temperature와 동일한 행동입니다.

### 1.4 요동-소산 정리 (Fluctuation-Dissipation Theorem)

> **HEP 물리학자를 위한 설명**: 이것은 열적 QFT의 Kubo 공식과 동일합니다.

통계역학에서 **응답함수**(response function)는 평형 **요동**(fluctuation)과 직접적으로 연결됩니다:

$$
C_v = \frac{\partial \langle E \rangle}{\partial T} = \frac{\beta^2}{N} \text{Var}(E) = \frac{\beta^2}{N}\left(\langle E^2 \rangle - \langle E \rangle^2\right)
$$

비열(specific heat) $C_v$는 에너지 요동의 분산에 비례합니다. 마찬가지로 자기감수율(susceptibility)은 자화(magnetization)의 요동에 비례합니다:

$$
\chi = \frac{\beta}{N}\left(\langle m^2 \rangle - \langle |m| \rangle^2\right) \sim \frac{\partial \langle m \rangle}{\partial h}\bigg|_{h=0}
$$

QFT 언어로 번역하면: $C_v$는 에너지 밀도 연산자의 2점 상관함수의 적분이고, $\chi$는 자화 밀도 연산자의 2점 상관함수의 적분입니다. 임계점에서 이 상관함수들이 장거리(long-range)로 변하면서 응답함수가 발산합니다.

---

## 2. 2D Ising 모델: 가장 단순한 상전이 시스템

### 2.1 해밀토니안

2D Ising 모델은 $L \times L$ 정방격자 위에서 정의됩니다:

$$
H = -J \sum_{\langle i,j \rangle} s_i \, s_j, \quad s_i \in \{-1, +1\}, \quad J = 1
$$

여기서 $\langle i,j \rangle$는 최근접 이웃 쌍(nearest-neighbor pair)을 의미하며, 주기 경계조건(periodic boundary conditions)이 적용됩니다. 에너지는 `torch.roll`을 사용하여 $O(N)$에 계산됩니다.

### 2.2 $Z_2$ 대칭과 자발적 대칭 깨짐

> **HEP 물리학자를 위한 핵심 포인트**: Ising $Z_2$는 **전역(global) 대칭**이며, **게이지(gauge) 대칭이 아닙니다**.

해밀토니안은 **모든 스핀을 동시에 뒤집는** 변환 $s_i \to -s_i$ 아래 불변입니다. 이것은 이산 전역 $Z_2$ 대칭입니다.

**HEP와의 대응:**

| 개념 | HEP | 2D Ising |
|---|---|---|
| $Z_2$ 대칭 | $\phi \to -\phi$ (실수 스칼라장) | $s_i \to -s_i$ (전체 스핀 뒤집기) |
| 질서변수 | Higgs VEV $\langle \phi \rangle = v$ | 자화 $M = \langle \sum_i s_i \rangle / N$ |
| 대칭 상 | $\langle \phi \rangle = 0$ (복원된 대칭) | $T > T_c$: $M = 0$ (무질서 상) |
| 깨진 상 | $\langle \phi \rangle = \pm v$ (두 진공) | $T < T_c$: $M = \pm M_0$ (질서 상) |
| SSB의 결과 | 질량 있는 Higgs + (게이지일 때) 먹힌 Goldstone | **Goldstone 보손 없음** (이산 대칭) |

**핵심 차이**: Higgs 메커니즘에서는 **국소(local) 게이지 대칭**이 깨지므로 Goldstone 보손이 게이지 보손에 "먹혀서" 질량을 줍니다. 반면 Ising 모델에서는 **전역 이산 대칭**이 깨지므로 연속 대칭의 Goldstone 정리가 적용되지 않고, 금보손(Goldstone boson)이 존재하지 않습니다.

#### ML 관점: 대칭 깨짐과 Mode Collapse

ML에서도 대칭 깨짐은 핵심적인 문제입니다:

| 물리학 개념 | ML 대응 | 본 프로젝트 |
|---|---|---|
| $Z_2$ 대칭 ($s_i \to -s_i$) | **출력 분포의 다봉성(multimodality)** | $+M$과 $-M$ 두 상태가 동등 |
| 자발적 대칭 깨짐 ($T < T_c$) | **Mode collapse** (GAN) / mode dropping | 모델이 한 모드만 학습 |
| 명시적 대칭 깨짐 (`fix_first: 1`) | **조건화(conditioning)를 통한 모드 선택** | 첫 스핀 $= +1$ 고정 → $+M$ 섹터만 학습 |
| 뉴런 순열 대칭 | **가중치 대칭** (모든 은닉 뉴런을 치환해도 동일 함수) | 학습 과정에서 자발적으로 깨짐 |

`fix_first: 1`은 물리적으로는 $Z_2$ 대칭을 깨뜨리는 것이고, ML 관점에서는 **mode collapse를 구조적으로 방지하는 전략**입니다. 생성 모델이 $+M$과 $-M$을 동시에 학습하면 모델 용량의 절반이 중복되므로, 한 섹터를 선택함으로써 효율적인 학습이 가능합니다.

**유효 포텐셜 관점**: Landau-Ginzburg 자유에너지 범함수는

$$
\mathcal{F}[\phi] = \int d^2x \left[ \frac{1}{2}(\nabla \phi)^2 + \frac{r}{2}\phi^2 + \frac{u}{4}\phi^4 \right]
$$

여기서 $r \propto (T - T_c)$이며, $r < 0$일 때 이중 우물 포텐셜이 형성됩니다. 이것은 HEP에서 익숙한 $\phi^4$ 이론의 유클리드 버전입니다.

### 2.3 코드베이스에서의 대칭 깨짐 처리

본 프로젝트에서는 `fix_first: 1` 파라미터로 첫 번째 스핀을 $+1$로 고정하여 $Z_2$ 대칭을 명시적으로 깨뜨립니다. 이는 HEP에서 게이지 고정(gauge fixing)을 하여 불필요한 중복(redundancy)을 제거하는 것과 유사한 전략입니다. 생성 모델이 $+M$과 $-M$ 상태를 동시에 학습하느라 용량을 낭비하는 것을 방지합니다.

### 2.4 상관함수와 전파자(Propagator)

> **HEP 물리학자를 위한 설명**: 스핀-스핀 상관함수는 유클리드 전파자(Euclidean propagator)입니다.

2점 상관함수:

$$
G(r) = \langle s_i s_j \rangle - \langle s_i \rangle \langle s_j \rangle
$$

이것의 행동은 QFT의 스칼라 전파자와 정확히 대응합니다:

**$T \neq T_c$ (유한한 상관길이, 질량 있는 이론):**

$$
G(r) \sim \frac{e^{-r/\xi}}{r^{(d-1)/2}}, \quad \text{여기서 } m = 1/\xi \text{ (질량)}
$$

QFT 대응: 질량 $m$인 스칼라장의 유클리드 전파자

$$
\Delta(x) \sim \int \frac{d^dk}{(2\pi)^d} \frac{e^{ikx}}{k^2 + m^2} \sim e^{-mr}
$$

**$T = T_c$ (상관길이 발산, 질량 없는 이론):**

$$
G(r) \sim \frac{1}{r^{d-2+\eta}} = \frac{1}{r^{1/4}} \quad (d=2, \, \eta=1/4)
$$

임계점에서 상관길이가 발산한다는 것은 QFT 언어로 **질량 갭(mass gap)이 사라지는 것**을 의미합니다. 시스템은 **질량 없는 이론**이 되고, 이것이 **등각 불변성(conformal invariance)**을 가져옵니다.

> **ML 전문가를 위한 해석**: $T_c$에서 이미지의 모든 픽셀이 장거리 상관관계를 갖습니다. 이는 작은 커널의 국소적 CNN이 포착하기 가장 어려운 레짐이며, 확장 합성곱(dilated convolution)이 필수적인 이유입니다.

**상관길이와 수용 영역(Receptive Field)의 직접 연결**:

상관함수 $G(r) \sim e^{-r/\xi}$에서 상관길이 $\xi$는 "정보가 유의미하게 전파되는 거리"입니다. 신경망의 수용 영역(RF)은 하나의 출력 뉴런이 참조할 수 있는 입력 범위입니다. 물리적으로, 모델이 올바른 조건부 확률 $q(s_i | s_{<i})$를 학습하려면:

$$
\text{RF} \geq \xi(T), \quad \text{특히 } T \approx T_c \text{에서 } \xi \sim L \text{이므로 RF} \geq L
$$

| 온도 영역 | 상관길이 $\xi$ | 필요한 RF | 학습 난이도 |
|---|---|---|---|
| 고온 ($T \gg T_c$) | $\xi \ll 1$ (거의 독립) | 작은 RF도 충분 | 쉬움 (독립 Bernoulli에 가까움) |
| 임계 ($T \approx T_c$) | $\xi \sim L = 16$ | **RF $\geq L$ 필수** | 가장 어려움 (장거리 상관) |
| 저온 ($T \ll T_c$) | $\xi \ll 1$ (정렬됨) | 국소 도메인 벽만 포착 필요 | 쉬움 (거의 모든 스핀 동일) |

이것이 본 프로젝트에서 확장 패턴 `[1,2,4,8]`로 RF $= 37 \times 37 > 16 \times 16$을 확보하는 물리적 이유입니다.

---

## 3. 상전이와 임계현상

### 3.1 2차 상전이 (Second-Order Phase Transition)

Ising 모델의 상전이는 **2차(연속) 상전이**입니다. 1차 상전이(물의 끓음처럼 잠열이 있는)와 달리, 2차 상전이에서는:

- 질서변수가 **연속적으로** 0에 접근합니다 ($T \to T_c^-$에서 $M \to 0$)
- **상관길이가 발산**합니다 ($\xi \to \infty$)
- 열역학적 양들이 **발산하거나 비해석적(non-analytic)** 행동을 보입니다

### 3.2 임계지수 (Critical Exponents)

축소온도(reduced temperature)를 $t = (T - T_c)/T_c$로 정의하면, 다양한 열역학적 양들이 거듭제곱 법칙으로 발산하거나 소멸합니다:

| 지수 | 물리량 | 정의 | 2D Ising (엄밀) | 평균장(Landau) |
|------|--------|------|-----------------|---------------|
| $\alpha$ | 비열 | $C_v \sim \|t\|^{-\alpha}$ | **0** (로그 발산) | 0 (불연속) |
| $\beta$ | 자화 | $M \sim (-t)^{\beta}$ | **1/8** | 1/2 |
| $\gamma$ | 감수율 | $\chi \sim \|t\|^{-\gamma}$ | **7/4** | 1 |
| $\delta$ | 상태방정식 | $H \sim M^{\delta}$ at $t=0$ | **15** | 3 |
| $\nu$ | 상관길이 | $\xi \sim \|t\|^{-\nu}$ | **1** | 1/2 |
| $\eta$ | 이상차원 | $G(r) \sim r^{-(d-2+\eta)}$ | **1/4** | 0 |

**임계지수가 신경망 학습에 미치는 실질적 영향**:

각 임계지수는 $T_c$ 근방에서 신경망이 직면하는 구체적인 어려움과 직결됩니다:

- **$\nu = 1$ (상관길이 발산)**: $\xi \sim |t|^{-1} \to \infty$이므로, **수용 영역이 격자 전체를 커버해야** 합니다. 국소 커널만으로는 장거리 상관을 포착할 수 없으며, 이것이 확장 합성곱(`dilation_pattern: [1,2,4,8]`)이 필수적인 물리적 근거입니다.

- **$\gamma = 7/4$ (감수율 발산)**: 감수율은 외부 섭동에 대한 시스템의 응답입니다. 이것은 **가중치의 작은 변화(섭동)에 대한 모델 출력의 민감도가 $T_c$에서 발산**한다는 것을 의미합니다. `analyze_compression.py`에서 관측하는 "가중치 압축 시 $T_c$에서 최대 열화"가 정확히 이 현상입니다.

- **$\alpha = 0$ (비열 로그 발산)**: 에너지 요동의 분산이 발산하므로, REINFORCE 기울기의 분산이 $T_c$에서 최대가 됩니다. 이것이 **온도별 RLOO 기준선**이 필요한 물리적 이유입니다.

- **$\beta = 1/8$ (자화 지수)**: 질서변수가 매우 천천히 ($|t|^{1/8}$) 성장하므로, $T_c$ 바로 아래에서 자화 신호가 약하여 모델이 질서/무질서 상을 구분하기 어렵습니다.

### 3.3 평균장 이론은 왜 실패하는가?

> **HEP 물리학자를 위한 설명**: 이것은 **상위 임계 차원(upper critical dimension)** 아래에서 요동이 지배적이라는 것과 관련됩니다.

평균장 이론은 국소적 스핀 환경을 자기무순(self-consistent) 평균장으로 대체하여 공간적 요동을 무시합니다. **Ginzburg 기준**(Ginzburg criterion)에 따르면 요동이 지배적인 조건은 $d < d_{uc}$ (상위 임계 차원)입니다. Ising 보편성류에서 $d_{uc} = 4$이므로:

- $d = 4$: 평균장 지수가 정확 (로그 보정 있음)
- $d = 3$: 평균장에서 상당히 벗어남 ($\alpha \approx 0.110$, $\beta \approx 0.326$)
- $d = 2$: 평균장이 **극적으로** 실패 ($\beta = 1/8$ vs 평균장 $1/2$)

HEP에서 차원적 정칙화(dimensional regularization)로 $d = 4 - \epsilon$ 전개를 하는 것과 같은 맥락입니다. $\epsilon$이 클수록(낮은 차원일수록) 루프 보정이 중요해집니다.

#### ML 관점: 평균장 ↔ 독립 픽셀 모델

평균장 근사는 ML에서 **각 픽셀이 독립이라고 가정하는 모델**과 정확히 대응합니다:

$$
\underbrace{p_{\text{MF}}(\mathbf{s}) = \prod_i p(s_i | \langle s \rangle)}_{\text{평균장: 독립 스핀}} \quad \longleftrightarrow \quad \underbrace{q_{\text{indep}}(\mathbf{x}) = \prod_i q(x_i)}_{\text{독립 픽셀 모델}}
$$

평균장이 실패하는 이유는 **공간적 상관관계를 무시**하기 때문이며, 이는 독립 픽셀 모델이 실제 이미지 분포를 포착하지 못하는 것과 같습니다. 자기회귀 모델(PixelCNN)은 조건부 의존성 $q(x_i | x_{<i})$를 통해 공간 상관관계를 명시적으로 모델링하므로, 평균장을 넘어서는 "요동 보정"을 자연스럽게 포함합니다.

### 3.4 스케일링 관계 (Scaling Relations)

6개의 임계지수는 독립이 아닙니다. 4개의 엄밀한 스케일링 관계가 존재하여 독립 지수는 2개뿐입니다(예: $\nu$와 $\eta$):

$$
\text{Rushbrooke:} \quad \alpha + 2\beta + \gamma = 2 \qquad (0 + 2 \cdot \tfrac{1}{8} + \tfrac{7}{4} = 2 \; \checkmark)
$$

$$
\text{Widom:} \quad \gamma = \beta(\delta - 1) \qquad (\tfrac{7}{4} = \tfrac{1}{8} \cdot 14 = \tfrac{7}{4} \; \checkmark)
$$

$$
\text{Fisher:} \quad \gamma = \nu(2 - \eta) \qquad (\tfrac{7}{4} = 1 \cdot (2 - \tfrac{1}{4}) = \tfrac{7}{4} \; \checkmark)
$$

$$
\text{Josephson (초스케일링):} \quad \nu d = 2 - \alpha \qquad (1 \cdot 2 = 2 - 0 = 2 \; \checkmark)
$$

> **HEP 물리학자를 위한 해석**: 이 관계들은 **자유에너지의 동차성**(homogeneity)에서 유도됩니다. 게이지 이론의 Ward 항등식이 그린 함수를 제약하듯이, 스케일링 관계는 스케일 불변성으로부터 임계지수를 제약합니다.

### 3.5 보편성류 (Universality Classes)

> **HEP 물리학자를 위한 핵심 아이디어**: 다른 미시적(UV) 이론들이 같은 거시적(IR) 고정점으로 흐를 수 있습니다.

**보편성류**란 미시적으로 완전히 다른 시스템들이 임계점에서 **동일한 임계지수**를 공유하는 현상입니다. 예를 들어:

- 정방격자 Ising 모델
- 삼각격자 Ising 모델
- 이원 합금의 질서-무질서 전이
- $CO_2$의 액체-기체 임계점

이들은 모두 2D Ising 보편성류에 속합니다. HEP 언어로: 다른 **UV 완비 이론**(UV completions)이 같은 **IR 고정점**으로 RG 흐름(flow)하는 것입니다. 격자 구조, 결합 상수의 부호, 심지어 물리적 차원(자성 vs 유체)까지 모두 RG 의미에서 **무관한(irrelevant) 연산자**에 해당합니다.

#### ML 관점: 보편성류와 Transfer Learning

보편성류는 ML의 **전이 학습(transfer learning)**과 깊은 구조적 유사성을 가집니다:

- **보편성류**: 미시적 세부사항(격자 구조, 상호작용 형태)이 다르더라도 **임계점의 거시적 행동**은 동일
- **Transfer learning**: 사전학습 도메인(ImageNet, 텍스트)이 다르더라도 **학습된 특성 표현**은 새로운 태스크에 전이 가능

두 현상 모두 같은 메커니즘으로 설명됩니다: **RG 흐름이 무관한 세부사항을 제거하고 관련 정보만 보존**합니다. 사전학습된 CNN의 초기 층(에지, 텍스처 검출)은 도메인에 무관한(irrelevant) 특성이 이미 제거된 보편적 표현이며, 이것이 다양한 도메인에서 전이가 가능한 이유입니다.

| RG 개념 | Transfer Learning 대응 |
|---|---|
| 무관 연산자 (IR에서 소멸) | 도메인 특수적 특성 (전이 시 제거됨) |
| 관련 연산자 (IR에서 성장) | 보편적 특성 (에지, 텍스처 — 전이됨) |
| 같은 보편성류 = 같은 임계지수 | 같은 "특성 공간" = 같은 전이 성능 |

### 3.6 비열의 로그 발산: 왜 특별한가?

2D Ising 모델의 비열은 $\alpha = 0$으로, 거듭제곱 법칙 대신 **로그 발산**을 보입니다:

$$
C_v \sim -A_{\pm} \ln|t| + B_{\pm}
$$

이것은 **한계적(marginal) 경우**입니다:
- $\alpha > 0$: 진정한 거듭제곱 발산 (예: 3D Ising, $\alpha \approx 0.110$)
- $\alpha = 0$: 로그 발산 (2D Ising)
- $\alpha < 0$: 비열이 첨점(cusp)을 보이나 유한 (예: 3D XY 모델)

RG 관점에서, $\alpha = 0$은 초스케일링 관계 $\nu d = 2 - \alpha$에서 $\nu d = 2$를 의미합니다. 상관 부피(correlation volume) $\xi^d$가 자유에너지 밀도 특이성과 정확히 같은 스케일링을 가지며, 거듭제곱 $|t|^0 = 1$의 모호성이 로그를 발생시킵니다.

**유한 크기 효과**: $L = 16$ 격자에서 비열 피크의 높이는 $C_v^{\max}(L) \sim \ln(L) \sim 2.77$로, 꽤 완만합니다. 이는 반면 감수율 피크 $\chi_{\max} \sim L^{7/4} \sim 128$이 훨씬 강한 신호를 제공하는 이유입니다.

---

## 4. Onsager 엄밀해와 그 역사적 의미

### 4.1 역사적 의의

Lars Onsager의 1944년 해는 이론물리학에서 가장 중요한 결과 중 하나입니다:

1. **연속 상전이의 최초 엄밀한 증명**. 1925년 Ernst Ising이 1D 모델에 상전이가 없음을 보인 후, 모든 차원에서 상전이가 없다고 잘못 추측했습니다. Onsager는 2D에서 상전이가 존재함을 증명했습니다.

2. **평균장 이론의 질적 실패를 최초로 입증**. 엄밀한 임계지수가 Landau 예측과 극적으로 다르다는 것을 보여, 낮은 차원에서 요동이 물리를 근본적으로 변경함을 증명했습니다.

3. **재규격화군의 선구자**. Onsager가 밝힌 비자명한 지수들은 Wilson의 RG 프로그램(1970년대 초)이 등장하기까지 약 30년간 기존 프레임워크로 설명할 수 없었습니다.

### 4.2 임계온도의 도출: Kramers-Wannier 쌍대성

Kramers와 Wannier(1941)는 Onsager보다 2년 앞서, 분배함수가 고온 전개와 저온 전개 사이의 **쌍대성(duality)**을 만족함을 보였습니다. 고온과 저온이 서로 매핑되며, 상전이가 유일하다면 자기쌍대점(self-dual point)에서 발생해야 합니다:

$$
\sinh(2J/T_c) = 1 \quad \Longrightarrow \quad T_c = \frac{2J}{\ln(1 + \sqrt{2})} \approx 2.269
$$

> **물리 배경 참고**: 이 쌍대성은 고온($T \to \infty$)과 저온($T \to 0$)의 분배함수를 서로 교환하는 변환입니다. HEP의 S-쌍대성($g \to 1/g$)과 구조적으로 유사하지만, 여기서 중요한 것은 실용적 결과입니다.

**실용적 의미**: 이 관계로부터 $T_c \approx 2.269$ ($\beta_c \approx 0.441$)가 엄밀하게 결정됩니다. 본 프로젝트의 모든 분석에서 이 값이 기준점으로 사용되며, `analyze_rank.py`와 `analyze_compression.py`에서 eRank 곡선의 극값 위치를 이 정확한 $T_c$와 비교합니다. 유한 크기($L = 16$) 효과로 인한 유사임계온도 이동 $T_c(L) \approx T_c + O(L^{-1})$도 이 엄밀해를 기준으로 정량화됩니다.

### 4.3 엄밀한 결과들

**자발 자화** (Onsager 발표 1948, Yang 증명 1952):

$$
M(T) = \begin{cases} \left[1 - \sinh^{-4}(2J/T)\right]^{1/8} & T < T_c \\ 0 & T \geq T_c \end{cases}
$$

지수 $1/8$은 당시 어떤 이론 프레임워크로도 유도할 수 없었습니다.

### 4.4 이 코드베이스에서의 활용

본 프로젝트는 `vatd_exact_partition.py`에서 유한 크기 격자($L = 16$)에 대한 Onsager의 정확한 분배함수를 구현합니다. 이를 통해:
- 신경망이 학습한 변분 자유에너지를 정확한 해와 비교할 수 있음
- 수치 2차 미분으로 정확한 $C_v(T)$를 얻어, 활성화 랭크와 비교하는 기준선으로 사용

---

## 5. 재규격화군(RG): Wilson의 블록 스핀과 HEP의 RG

> **HEP 물리학자를 위한 핵심 메시지**: 당신이 아는 RG(달리는 결합상수, 베타 함수)는 **운동량 공간 RG**입니다. Wilson의 **실공간 RG**(블록 스핀 변환)는 이것의 실공간 버전입니다.

### 5.1 HEP의 RG vs Wilson의 RG

| 관점 | HEP (운동량 공간) | Wilson (실공간) |
|------|------------------|----------------|
| 절차 | 높은 운동량 모드 ($k > \Lambda$) 적분 | $2 \times 2$ 스핀 블록을 하나의 "초-스핀"으로 |
| 효과 | 결합상수의 달림 $g(\mu)$ | 유효 해밀토니안 파라미터의 변화 |
| 고정점 | $\beta(g) = 0$ | 임계점 $T = T_c$ |
| UV 자유도 제거 | 루프 적분의 정칙화 | 격자 조대화(coarse-graining) |

### 5.2 관련(Relevant), 무관(Irrelevant), 한계(Marginal) 연산자

RG 고정점 근방에서 연산자를 분류합니다:

- **관련 연산자** (eigenvalue $\lambda > 1$): IR에서 **성장** → 시스템을 고정점에서 벗어나게 함
  - HEP 비유: $d = 4$에서 질량항 $m^2 \phi^2$ (관련 연산자)
  - Ising: 열적 섭동 $\epsilon$ (온도 방향)과 자기적 섭동 $\sigma$ (외부장 방향)
  - 2D Ising 임계 고정점에는 **정확히 2개**의 관련 연산자가 있음

- **무관 연산자** (eigenvalue $\lambda < 1$): IR에서 **소멸** → 미시적 세부사항의 "잊혀짐"
  - HEP 비유: $d = 4$에서 $\phi^6/M^2$ (비재규격화 가능 항)
  - Ising: 격자 구조, 다음-최근접 이웃 결합 등 → **보편성**의 기원

- **한계 연산자** (eigenvalue $\lambda = 1$): 일정하게 유지 (또는 로그적으로 달림)

### 5.3 CNN과 RG의 유사성

> **ML 전문가를 위한 핵심 연결**: CNN의 풀링(pooling)은 본질적으로 Wilson의 블록 스핀 변환과 같은 구조입니다.

| CNN 구조 | RG 대응 |
|----------|---------|
| Max/Average 풀링 | 블록 스핀 변환 (조대화) |
| 스트라이드 합성곱 | RG 스텝 (해상도 감소) |
| 깊은 층 | 반복된 RG 변환 |
| 잔차(residual) 연결 | 관련 연산자의 보존 |

**Mehta & Schwab (2014)의 정확한 매핑**: 이 유사성은 단순한 비유가 아닙니다. Mehta와 Schwab은 2D Ising 모델에서 학습된 RBM(Restricted Boltzmann Machine)의 가중치 행렬이 실공간 RG 변환(블록 스핀)과 **정확히 일치**함을 보였습니다:

- RBM의 은닉 뉴런 $h_j$ = 블록 스핀 변수 (조대화된 자유도)
- RBM의 가중치 행렬 $W_{ij}$ = 다수결(majority rule) 조대화 커널
- RBM 층의 깊이 = RG 스텝의 횟수

이는 **심층 신경망이 RG를 자연스럽게 수행한다**는 것을 의미합니다: 각 층이 미세 스케일의 자유도를 제거하고 거시적 집단 변수만 보존합니다.

**깊이와 Low-Rank Bias**: Arora et al. (2019)는 깊은 행렬 분해(deep matrix factorization)에서 경사 하강법이 자연스럽게 저랭크 해를 찾는다는 것을 증명했습니다. 이것은 RG 해석과 결합하면 강력한 예측을 제공합니다: **깊은 층을 거칠수록 무관 연산자가 소멸하므로 유효 랭크가 감소**합니다. 이것이 `LOW_RANK.md`에서 관측하는 층별 랭크 구조의 이론적 배경입니다.

**한계점 — CNN ≠ RG**: 이 유사성에는 중요한 차이가 있습니다:
- RG는 **비가역적**(irreversible) 조대화입니다 — UV 정보가 영구히 손실됩니다
- CNN은 **정보를 파괴하지 않습니다** — 잔차 연결이나 스킵 연결을 통해 세부 정보가 보존될 수 있습니다
- RG 흐름은 결합상수 공간에서의 궤적이지만, CNN의 가중치 갱신은 매개변수 공간에서의 최적화입니다
- 따라서 CNN-RG 유사성은 **정보 흐름의 구조적 유사성**이지, 정확한 수학적 동치는 아닙니다

`LOW_RANK.md`의 결과(Section 5.1)에서 **초기 층(Block 0-1)이 $T_c$에서 가장 깊은 랭크 하락**을 보이는 것은 RG 해석과 일치합니다: 초기 층은 UV(단파장) 자유도를 적분해 내는 조대화를 수행하고, 후기 층은 이미 재규격화된 IR(장파장) 변수를 다룹니다.

### 5.4 딥러닝에서의 상전이 현상

물리학의 상전이 개념은 딥러닝 연구에서도 핵심적인 현상을 설명하는 데 활용되고 있습니다:

#### Grokking과 Effective Rank

**Grokking** (Power et al., 2022)은 신경망이 훈련 데이터를 암기한 후, 추가 훈련을 통해 갑자기 일반화 성능을 획득하는 현상입니다. 이것은 **상전이와 유사한 급격한 전환**입니다:

- 암기 단계: 가중치의 effective rank가 높음 (과적합, 무질서 상과 유사)
- 일반화 전환: effective rank가 **급격히 하락** → 가중치가 저차원 구조로 수렴
- 전환점: 손실 곡선에서 급격한 변화 (임계점과 유사)

이것은 본 프로젝트의 관찰과 직접 연결됩니다: $T_c$에서 활성화 랭크가 급락하는 것은, 신경망이 임계점의 저차원 구조를 "발견"하는 일종의 grokking입니다.

#### Neural Collapse

**Neural collapse** (Papyan et al., 2020)은 분류 신경망의 마지막 층에서 클래스 특성이 정규 심플렉스(simplex)로 수렴하는 현상입니다:

- 마지막 층 활성화의 클래스 내 분산 → 0 (rank collapse)
- 클래스 평균 특성이 **정확히 $K$개의 방향**으로 수렴 ($K$ = 클래스 수)
- 이것은 물리적으로 **완전한 대칭 깨짐** — 연속적 특성 공간이 이산적 구조로 응축

Neural collapse에서의 랭크 $= K$(클래스 수)는, 2D Ising에서의 eRank $\approx 3$–$5$(관련 연산자 수 + 보정)와 같은 물리적 메커니즘입니다: 두 경우 모두 시스템의 실질적 자유도 수로 차원이 축소됩니다.

#### Edge of Chaos와 보편성류

신경망 초기화에서 **edge of chaos** (Poole et al., 2016)는 신호가 폭발하지도 소멸하지도 않는 임계 초기화 조건입니다. 최근 연구 (Roberts et al., *Phys. Rev. Research*, 2025)는 이 임계점이 물리적 상전이와 **같은 보편성류**에 속할 수 있음을 보여주었습니다:

- 가중치 분산 $\sigma_w^2$가 임계값 아래: 신호 소멸 (질서 상)
- 가중치 분산 $\sigma_w^2$가 임계값 위: 신호 폭발 (무질서 상/혼돈)
- 임계점: 정보가 최대한 깊이 전파 = 가장 효과적인 학습

이는 "좋은 초기화"가 물리학적 임계점에 해당한다는 것을 의미하며, 본 프로젝트에서 임계 온도의 학습이 가장 어려운 것과 연결됩니다.

---

## 6. 2D Ising CFT: c = 1/2 최소 모델

> **ML 독자를 위한 도입**: 이 섹션은 수학적으로 밀도가 높지만, 핵심 메시지는 간단합니다:
>
> **"임계점에서 시스템의 자유도가 몇 개인가?"에 대한 정확한 답을 제공합니다.**
>
> CFT(등각장론)는 임계점의 물리를 완전히 분류합니다. 2D Ising의 경우:
> - **관련 연산자 = 2개** ($\sigma$: 자화, $\epsilon$: 에너지 밀도)
> - 이것은 네트워크가 $T_c$에서 포착해야 할 **독립적인 물리 모드의 수**입니다
> - `analyze_rank.py`에서 eRank가 $T_c$ 부근에서 3–5로 떨어지는 것 = **2개 관련 연산자 + 항등원 + 유한 크기 보정**의 물리적 반영
>
> 수학적 세부사항을 건너뛰더라도, "관련 연산자 수 → eRank 하한"이라는 핵심 연결만 기억하시면 됩니다.

### 6.1 임계점에서의 등각 불변성

임계점에서 상관길이가 발산하면 ($\xi \to \infty$, 즉 질량 갭 $m \to 0$), 시스템은 **스케일 불변(scale-invariant)**이 됩니다. 2차원에서 스케일 불변성은 일반적으로 훨씬 큰 **등각 대칭(conformal symmetry)** — 각도를 보존하는 변환에 대한 불변성 — 을 수반합니다.

### 6.2 중심 전하 c = 1/2

2D Ising 임계점은 **M(4,3) 최소 모델(minimal model)**로 기술되며, 중심 전하(central charge)는:

$$
c = \frac{1}{2}
$$

**중심 전하의 직관적 의미**: $c$는 시스템의 **유효 자유도 수**를 세는 양입니다.
- $c = 1/2$: 자유 Majorana 페르미온 **1개** (가장 단순한 비자명 CFT)
- $c = 1$: 자유 보손 **1개** (또는 Dirac 페르미온 1개)
- $c = 7/10$: 3-상태 Potts 모델 (더 복잡한 임계점)

$c$가 작을수록 임계점의 물리가 **단순하고 제약적**입니다. $c = 1/2$는 가능한 가장 단순한 임계점으로, 일차 연산자가 겨우 3개뿐입니다. 이것이 2D Ising이 "가장 단순한 상전이 시스템"인 이유입니다.

> **수학적 배경** (선택적): 등각 대칭의 생성자는 Virasoro 대수 $[L_m, L_n] = (m-n)L_{m+n} + \frac{c}{12}m(m^2 - 1)\delta_{m+n,0}$를 만족합니다. $c$는 이 대수의 중심 확장(central extension)으로, 양자 이상(quantum anomaly)에서 기원합니다.

물리적으로, $c = 1/2$는 2D Ising 임계 모델이 **질량 없는 Majorana 페르미온 장론**과 동치임을 의미합니다. $T \neq T_c$에서 페르미온은 질량 $m \sim |t|^\nu = |t|$를 획득하며, 이것이 역 상관길이입니다.

### 6.3 일차 연산자와 Kac 테이블

M(4,3) 최소 모델에는 정확히 **3개의 일차 연산자(primary operator)**가 있습니다:

| 연산자 | 기호 | 스케일링 차원 $\Delta$ | 물리적 의미 | **신경망에서의 의미** |
|--------|------|----------------------|------------|---------------------|
| 항등원 | $\mathbb{1}$ | 0 | 진공 (기저 상태) | 평균 배경 (DC 성분) |
| 스핀 | $\sigma$ | $1/8$ | 자화 밀도 | **전역 자화 모드** — 가장 중요한 특성 |
| 에너지 | $\epsilon$ | $1$ | 에너지 밀도 섭동 | **국소 에너지 요동 모드** — 두 번째 중요 특성 |

**ML 관점에서의 재해석**: 이 3개의 일차 연산자는 **신경망이 $T_c$에서 학습해야 할 독립 특성**입니다:

- $\mathbb{1}$ (항등원): 평균 배경 — 모든 온도에서 존재하는 기본 모드
- $\sigma$ (스핀 연산자): 전역 자화 패턴 — 저온에서 질서를 형성하는 모드
- $\epsilon$ (에너지 연산자): 국소 에너지 요동 — 인접 스핀 간의 정렬/비정렬 패턴

임계점에서 이 **2개의 관련 연산자** ($\sigma$, $\epsilon$)만으로 시스템의 모든 장거리 물리가 결정됩니다. 나머지 무한히 많은 연산자(고차 보정)는 무관 연산자로서 RG 흐름 아래에서 소멸합니다. 이것이 `analyze_rank.py`에서 eRank가 $T_c$에서 3–5로 떨어지는 물리적 기원입니다: **2개 관련 연산자 + 항등원 + 유한 크기 보정**.

스케일링 차원으로부터 임계지수가 직접 결정됩니다:

$$
\langle \sigma(0) \sigma(r) \rangle \sim r^{-2\Delta_\sigma} = r^{-1/4} \quad \Longrightarrow \quad \eta = 1/4
$$

$$
\beta = \Delta_\sigma \cdot \nu = \frac{1}{8} \cdot 1 = \frac{1}{8}
$$

### 6.4 OPE와 등각 부트스트랩

융합 규칙(fusion rules)은:

$$
\sigma \times \sigma = \mathbb{1} + \epsilon, \quad \sigma \times \epsilon = \sigma, \quad \epsilon \times \epsilon = \mathbb{1}
$$

이것은 QFT의 **연산자곱 전개(OPE)**의 이산 버전입니다. 핵심 메시지는: 3개의 일차 연산자와 이 융합 규칙만으로 **모든 상관함수가 완전히 결정**된다는 것입니다. 이것이 2D Ising이 "정확히 풀 수 있는(exactly solvable)" 이유입니다.

> **등각 부트스트랩** (심화 참고): 현대 등각 부트스트랩 프로그램(El-Showk et al., 2012)은 대칭성 제약조건만으로 3D Ising 임계지수를 높은 정밀도로 결정했습니다. 이것은 ML의 "제약 최적화"와 유사합니다 — 물리적 일관성 조건(유니타리성, 교차 대칭)이 허용 가능한 해를 극도로 제한합니다.

---

## 7. Low-Rank Hypothesis: Nature Physics (2024)

> **논문**: Thibeault, V., Allard, A., & Desrosiers, P., "The low-rank hypothesis of complex systems," *Nature Physics* **20**, 294–302 (2024).
>
> (**주의**: `LOW_RANK.md`의 참고문헌 1에서 저자가 "Thiede, Giannakis"로 표기되어 있으나, 정확한 저자는 **Thibeault, Allard, Desrosiers**입니다.)

### 7.1 핵심 주장

이 논문은 복잡계 과학에서 널리 사용되지만 거의 명시적으로 진술되지 않는 가정을 정의하고 검증합니다:

> **Low-Rank Hypothesis**: 네트워크 위의 고차원 비선형 역학계의 결합 행렬(coupling matrix)은 **빠르게 감소하는 특이값 스펙트럼**을 가지며, 따라서 저차원 부분공간으로의 투영이 역학을 충실하게 기술한다.

### 7.2 수학적 프레임워크

네트워크 위의 역학계:

$$
\dot{x}_i = F(x_i) + \sum_j W_{ij} G(x_i, x_j)
$$

여기서 $W$는 $N \times N$ 결합 행렬입니다. SVD를 수행하면:

$$
W = U\Sigma V^\top = \sum_{k=1}^{N} \sigma_k \mathbf{u}_k \mathbf{v}_k^\top
$$

Low-rank hypothesis는 $\sigma_k$가 $k$에 대해 빠르게(종종 지수적으로) 감소하므로, 랭크-$r$ 절단 ($r \ll N$)이 $W$와 그 위의 역학을 충실하게 근사한다고 주장합니다.

### 7.3 검증 결과

- **600개 이상의 실세계 네트워크**: 안정 랭크(stable rank)가 일반적으로 ambient 차원의 **10% 이하**
- **다양한 역학 모델**에서 검증: SIS 역병 모델, Kuramoto 진동자 동기화, Wilson-Cowan 신경 역학, 일반화된 Lotka-Volterra, 순환 신경망(RNN)
- **RNN의 정확한 차원 축소**: 랭크-$n$ 연결 행렬 $W$를 가진 RNN은 $n$차원 축소 시스템으로 **정확하게**(근사 오차 0) 기술 가능

### 7.4 핵심 발견: 고차 상호작용의 자연적 출현

가장 놀라운 결과 중 하나: 쌍별(pairwise) 네트워크 역학의 최적 차원 축소를 수행하면, **고차(beyond-pairwise) 상호작용**이 축소된 방정식에서 텐서 항으로 자연스럽게 출현합니다. 이는 복잡계에서 고차 상호작용의 기원에 대한 이론적 설명을 제공합니다.

### 7.5 본 프로젝트와의 관계

Thibeault 논문은 **네트워크 결합 행렬**의 특이값 스펙트럼을 분석합니다. 본 프로젝트(`LOW_RANK.md`)는 이를 세 가지 방향으로 확장합니다:

**확장 1: 정적 구조 → 동적 표현**. Thibeault는 결합 행렬 $W$의 정적 특이값 스펙트럼을 분석하지만, 본 프로젝트는 신경망의 **활성화(activation)**가 온도에 따라 어떻게 변하는지를 추적합니다. 같은 가중치를 가진 네트워크가 다른 온도의 입력에 대해 다른 유효 랭크를 보이는 것은, low-rank 구조가 네트워크 구조가 아닌 **입력 데이터의 물리적 특성**에서 기원함을 보여줍니다.

**확장 2: 물리계 → 학습된 표현**. 원래 hypothesis는 "물리 시스템의 결합 행렬이 low-rank"라는 주장입니다. 본 프로젝트는 더 나아가 "물리 시스템을 **학습한 신경망의 내부 표현**이 그 low-rank 구조를 반영한다"는 것을 실험적으로 보여줍니다.

**확장 3: 온도 의존적 랭크 프로파일**. 가장 중요한 새 발견은 eRank의 온도 의존성입니다: 고온에서 높고, $T_c$에서 급락하며, 저온에서 부분 회복합니다. 이 비단조적 프로파일은 Thibeault 논문에서 다루지 않는 것으로, RG 흐름과 CFT의 관련 연산자 수로 설명됩니다.

### 7.6 Koopman 연산자와의 관계

Low-rank hypothesis는 **Koopman 연산자 이론**과 병행하지만 구별되는 접근입니다:

| 측면 | Thibeault (Low-Rank) | Koopman 이론 |
|------|---------------------|-------------|
| 분석 대상 | 결합 행렬 $W$ (네트워크 구조) | 진화 연산자 $K$ (역학) |
| 성질 | 유한 차원 행렬 | 무한 차원 선형 연산자 |
| Low-rank 의미 | 상호작용 위상의 소수 지배적 특이 모드 | 역학의 소수 지배적 고유함수 |

두 프레임워크 모두 같은 물리적 현상의 표현입니다: 전이 근처의 복잡계는 소수의 집단적 자유도에 의해 지배됩니다.

### 7.7 Low-Rank Hypothesis와 현대 ML 연구의 접점

Low-rank hypothesis는 물리학에서 출발했지만, 현대 ML 연구의 핵심 결과들과 깊이 연결됩니다:

#### Implicit Rank Regularization (Arora et al., 2019)

경사 하강법(GD)으로 행렬 분해를 학습하면, 명시적 정칙화 없이도 **저랭크 해가 자연스럽게 선호**됩니다. 심층 행렬 분해 $W = W_L \cdots W_2 W_1$에서 깊이가 증가할수록 이 implicit bias가 강화됩니다. 이것은 low-rank hypothesis의 ML 내부 메커니즘입니다: 물리 시스템이 저랭크인 것뿐 아니라, **신경망 최적화 자체가 저랭크를 선호**합니다.

#### LoRA: Low-Rank Adaptation (Hu et al., 2021)

대규모 언어 모델의 미세조정에서 가중치 업데이트 $\Delta W = BA$ ($B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$)로 제한해도 전체 미세조정과 동등한 성능을 달성합니다. 이것이 가능한 이유는:

- 미세조정 시 가중치 변화가 **본질적으로 저랭크 부분공간**에 놓여 있음
- 이는 사전학습된 모델이 이미 "관련 연산자"를 학습했고, 미세조정은 소수의 방향만 조정하면 되기 때문
- Low-rank hypothesis의 직접적 응용: 실질적 자유도 $r \ll d$

#### ARSVD: Adaptive Rank SVD (2024)

특이값 스펙트럼의 **스펙트럼 엔트로피**를 기반으로 최적 랭크를 적응적으로 결정하는 방법입니다. 본 프로젝트의 eRank (Roy & Vetterli, 2007)와 같은 정보이론적 접근이지만, 모델 압축에 직접 활용됩니다. 핵심 아이디어: "얼마나 압축할 수 있는가?"는 물리학의 "관련 자유도가 몇 개인가?"와 같은 질문입니다.

#### AlphaPruning (NeurIPS 2024)

가중치 행렬의 특이값 분포가 **heavy-tailed power law** $p(\sigma) \sim \sigma^{-\alpha}$를 따른다는 관찰에 기반한 가지치기(pruning) 방법입니다. 꼬리 지수 $\alpha$가 작을수록 해당 층이 더 잘 학습되었음(더 low-rank)을 나타내며, 이를 기반으로 층별 가지치기 비율을 결정합니다. 이것은 RG 관점에서 자연스럽습니다: 잘 학습된 층은 무관 연산자를 효과적으로 제거했으므로 더 공격적으로 압축할 수 있습니다.

---

## 8. SVD와 랭크 지표들

### 8.1 특이값 분해 (SVD) 기초

> **ML 전문가를 위한 설명**: SVD는 PCA의 일반화이며, 행렬의 "정보 구조"를 분해합니다.

행렬 $A \in \mathbb{R}^{m \times n}$의 SVD:

$$
A = U \Sigma V^\top = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

여기서 $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$은 직교 행렬, $\Sigma$는 대각 행렬, $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$은 특이값입니다.

**PCA와의 관계**: 데이터 행렬 $X$를 mean-centering한 후 SVD를 적용하면 $X = U\Sigma V^\top$이며, $V$의 열벡터가 주성분(principal components), $\sigma_i^2/(N-1)$이 분산에 비례합니다.

### 8.2 절단 SVD와 Eckart-Young 정리

> **핵심 정리**: 랭크-$k$ 절단 $A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$는 Frobenius 노름에서 **최적의** 랭크-$k$ 근사입니다.

$$
A_k = \underset{\text{rank}(B) \leq k}{\arg\min} \|A - B\|_F, \qquad \|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2
$$

이 정리는 `analyze_compression.py`에서 가중치 행렬을 압축할 때 사용되는 이론적 기반입니다.

**현대 모델 압축과의 연결**: Eckart-Young 정리는 LoRA를 포함한 거의 모든 SVD 기반 모델 압축의 이론적 토대입니다:

- **LoRA adapter**: 가중치 업데이트 $\Delta W = BA$는 Eckart-Young의 최적 랭크-$r$ 근사를 파라미터화한 것입니다
- **SVD pruning**: 작은 특이값을 제거하는 것은 무관 연산자(RG 언어)를 폐기하는 것과 동치
- **Implicit low-rank bias**: GD로 학습된 가중치가 이미 저랭크이므로, SVD 절단의 정보 손실이 작습니다

본 프로젝트의 `analyze_compression.py`는 이 연결을 직접 검증합니다: SVD 절단 비율(90%, 75%, 50%, 25%, 10%)에 따른 자유에너지 열화를 온도별로 측정하여, **$T_c$에서 압축 민감도가 최대**임을 보여줍니다.

### 8.3 유효 랭크 (Effective Rank, erank)

Roy & Vetterli (2007)가 정의한, 특이값의 정규화 분포에 대한 Shannon 엔트로피를 사용한 연속 랭크 측도:

$$
\text{erank}(A) = \exp\!\left(H(\mathbf{p})\right), \quad p_i = \frac{\sigma_i}{\sum_j \sigma_j}, \quad H(\mathbf{p}) = -\sum_i p_i \ln p_i
$$

**성질:**
- **연속 스칼라**: $[1, \min(m, n)]$ 범위의 실수값 반환
- $\text{erank} = 1$: 단일 특이값이 지배 (완전 랭크-1)
- $\text{erank} = \min(m, n)$: 모든 특이값이 균일 (최대 랭크)
- **스케일 불변**: $\text{erank}(\alpha A) = \text{erank}(A)$

**왜 단순한 랭크가 아닌 erank를 사용하는가?** 수학적 랭크는 $\sigma_i > 0$인 $i$의 수를 세지만, 수치 노이즈로 인해 거의 항상 $\text{rank} = \min(m,n)$입니다. erank는 특이값의 **분포 균일도**를 측정하여 "실질적으로 독립적인 차원의 수"를 연속적으로 포착합니다.

### 8.4 안정 랭크 (Stable Rank, srank)

노이즈에 강건한 대안:

$$
\text{srank}(A) = \frac{\|A\|_F^2}{\|A\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_{\max}^2}
$$

Frobenius 노름(총 에너지)과 스펙트럼 노름(최대 모드 에너지)의 비율입니다. 특이값을 제곱으로 가중하므로 작은 특이값에 덜 민감합니다. 행렬 집중 부등식(matrix concentration inequalities)에서 자연스럽게 등장합니다.

### 8.5 참여비 (Participation Ratio, PR)

> **HEP/응집물질 물리학자를 위한 배경**: 참여비는 원래 **Anderson 국소화**(Anderson localization) 이론에서 파동함수가 얼마나 많은 격자점에 퍼져 있는지를 측정하는 데 사용되었습니다.

$$
\text{PR}(A) = \frac{\left(\sum_i \sigma_i^2\right)^2}{\sum_i \sigma_i^4}
$$

$k$개의 특이값이 같고 나머지가 0이면 $\text{PR} = k$. 스펙트럼에 "참여하는" 특이값의 유효 개수를 셉니다.

### 8.6 스크리 플롯 (Scree Plot) 읽는 법

> **ML 전문가를 위한 설명**: PCA의 "explained variance ratio" 플롯과 동일합니다.

스크리 플롯은 정규화된 특이값 $\sigma_i / \sigma_1$을 인덱스 $i$에 대해 그린 것입니다:

- **평탄한 감소**: 많은 차원이 비슷하게 기여 → 높은 유효 랭크 (고온 Ising)
- **급격한 감소 (엘보)**: 소수 차원이 지배 → 낮은 유효 랭크 (저온 Ising, 임계 Ising)

`LOW_RANK.md` Figure 2에서 고온($T = 4.73$)의 완만한 감소와 저온($T = 1.36$)의 급격한 감소를 비교하면 이 차이를 명확히 볼 수 있습니다.

### 8.7 채널 랭크 vs 공간 랭크

활성화 텐서 $A \in \mathbb{R}^{N \times C \times H \times W}$에서 SVD를 적용하는 두 가지 관점:

| 관점 | 구성 | SVD 대상 | 해석 |
|------|------|---------|------|
| **채널 랭크** | 공간 차원 평균: $\bar{A} \in \mathbb{R}^{N \times C}$ | $\text{SVD}(\bar{A} - \text{mean})$ | 독립적 정보를 전달하는 특성 채널의 수 |
| **공간 랭크** | 채널 차원 평균: $\bar{A} \in \mathbb{R}^{N \times HW}$ | $\text{SVD}(\bar{A} - \text{mean})$ | 독립적인 공간 모드의 수 |

### 8.8 신경망 가중치의 SVD 압축

Conv2D 가중치 텐서 $(C_{\text{out}}, C_{\text{in}}, k, k)$를 2D 행렬 $(C_{\text{out}}, C_{\text{in}} \cdot k^2)$로 리쉐이핑한 후 SVD를 적용합니다. 마스크된 합성곱의 경우, 유효 가중치 $W_{\text{eff}} = W \odot M$ (원소별 곱)에 대해 SVD를 수행해야 합니다.

---

## 9. 자기회귀 모델과 PixelCNN

### 9.1 확률의 체인 규칙과 정확한 우도

> **ML 전문가를 위한 핵심**: 자기회귀 모델의 최대 장점은 **정확한(exact) 로그 우도** 계산이 가능하다는 것입니다.

임의의 결합 분포는 체인 규칙으로 분해됩니다:

$$
q_\theta(\mathbf{x}) = \prod_{i=1}^{N} q_\theta(x_i \mid x_1, x_2, \ldots, x_{i-1})
$$

따라서 로그 우도는:

$$
\log q_\theta(\mathbf{x}) = \sum_{i=1}^{N} \log q_\theta(x_i \mid x_{<i})
$$

VAE와 달리 ELBO(Evidence Lower Bound)가 아닌 **정확한 로그 우도**를 제공합니다. 정규화 흐름(normalizing flows)과 달리 가역성(invertibility) 제약이 없습니다.

### 9.2 마스크된 합성곱 (Masked Convolution)

2D 격자에서 자기회귀 순서를 강제하기 위해 합성곱 커널에 바이너리 마스크를 적용합니다.

**래스터 스캔 순서**: 왼쪽→오른쪽, 위→아래. 위치 $(i, j)$의 출력은 이 순서에서 앞서는 위치에만 의존해야 합니다.

**Type A 마스크 (첫 번째 층)**:

```
1  1  1
1 [0] 0
0  0  0
```

중심 픽셀 **자신을 제외**합니다. 첫 번째 층에서 자기 자신을 보면 사소한 항등 지름길(identity shortcut)이 되어, 모델이 조건부 분포를 학습하지 않고 입력값을 복사하게 됩니다.

**Type B 마스크 (후속 층)**:

```
1  1  1
1 [1] 0
0  0  0
```

중심 픽셀을 **포함**합니다. 첫 번째 층의 Type A 마스크 덕분에 이 위치의 은닉 표현은 이미 $x_{<(i,j)}$의 정보만 포함하므로, Type B를 사용해도 정보 누출이 없습니다.

### 9.3 확장 합성곱 (Dilated Convolution)과 수용 영역

> **물리학자를 위한 동기**: 임계점에서 상관관계가 격자 전체에 걸치므로, 모델의 수용 영역이 격자 크기 이상이어야 합니다.

확장(dilation)은 커널 원소 사이에 간격을 삽입합니다. 확장 패턴 $[1, 2, 4, 8]$로 4개의 층을 쌓으면:

$$
\text{RF}_{\text{total}} = 1 + \sum_l (k_l - 1) \cdot d_l
$$

$3 \times 3$ 커널과 확장 $[1, 2, 4, 8]$을 8개 층에 적용하면 수용 영역은 $61 \times 61$이 되어, $16 \times 16$ 격자를 완전히 포괄합니다. 4개 층이면 $37 \times 37$입니다.

**훈련과 샘플링의 비대칭**:
- **훈련**: 모든 조건부 확률이 합성곱을 통해 동시에 계산됨 → $O(1)$ (시퀀스 길이에 독립)
- **샘플링**: 순차적으로 한 스핀씩 생성 → $L^2 = 256$ 번의 순방향 전달 필요 (주요 계산 병목)

### 9.4 온도 조건화

온도 $\beta$는 스핀 배위에 추가 입력 채널로 연결됩니다. 이 증강 채널은 자기회귀 마스크의 적용을 받지 않습니다 — 전역 조건 정보이므로 모든 위치에서 접근 가능해야 합니다.

### 9.5 대안적 스캔 순서

- **대각선 스캔** (`DiagonalMaskedConv2D`): 반대각선 ($i + j = \text{const}$) 단위로 병렬 처리. 같은 반대각선의 픽셀은 서로 독립 → 샘플링 스텝 수가 $O(L^2) \to O(2L-1)$로 감소 (~8배 속도 향상)
- **Hilbert 곡선**: 2D 국소성을 최대한 보존하는 공간 채움 곡선. 실험적 단계.

---

## 10. 변분 열역학 발산(VaTD)과 자유에너지 최소화

### 10.1 핵심 아이디어

> **ML 전문가를 위한 설명**: VaTD는 **훈련 데이터 없이** 목표 분포(Boltzmann)를 학습합니다. 에너지 함수가 알려져 있으므로, 모델이 자체 생성한 샘플로 자기 강화(self-play) 학습을 합니다.

일반적인 생성 모델링에서는 목표 분포의 샘플(훈련 데이터)이 필요합니다. VaTD에서는:
- 에너지 함수 $E(\mathbf{x})$는 해석적으로 알려져 있음 (Ising 해밀토니안)
- 모델이 배위를 생성하고, Boltzmann 분포와의 차이를 줄이는 방향으로 학습
- 이것은 본질적으로 **강화학습(RL) / 자기 대국(self-play)** 설정

**RL 프레이밍**: VaTD를 강화학습의 언어로 정확히 대응시키면:

| RL 개념 | VaTD 대응 | 구체적 구현 |
|---|---|---|
| **에이전트 (Agent)** | PixelCNN | 자기회귀 신경망 |
| **상태 (State)** | 이전 스핀들 $s_{<i}$ | 래스터 순서로 생성된 앞선 스핀 |
| **행동 (Action)** | 현재 스핀 선택 $s_i \in \{-1, +1\}$ | Bernoulli 확률로 샘플링 |
| **에피소드** | 완전한 스핀 배위 $\mathbf{x}$ | 256개 스핀의 순차 생성 |
| **보상 (Reward)** | $-(\log q_\theta(\mathbf{x}) + \beta E(\mathbf{x}))$ | 낮은 에너지 + 높은 엔트로피 |
| **환경 (Environment)** | 물리 법칙 (Ising 해밀토니안) | `energy(samples, beta)` |

핵심적인 차이: 일반적인 RL에서는 환경과의 상호작용이 필요하지만, VaTD에서는 에너지 함수가 해석적이므로 **환경 시뮬레이션이 불필요**합니다. 이것이 "데이터 없는 생성 모델"의 의미입니다 — 보상 함수(에너지)가 물리에서 직접 주어집니다.

### 10.2 KL 발산에서 자유에너지로

학습된 분포 $q_\theta(\mathbf{x})$와 목표 Boltzmann 분포 $p_\beta(\mathbf{x}) = e^{-\beta E(\mathbf{x})}/Z(\beta)$ 사이의 KL 발산:

$$
D_{\text{KL}}(q_\theta \| p_\beta) = \mathbb{E}_{q_\theta}\!\left[\log q_\theta(\mathbf{x}) + \beta E(\mathbf{x})\right] + \log Z(\beta)
$$

**변분 자유에너지**를 정의하면:

$$
F_q \equiv \langle E \rangle_q + T \langle \log q \rangle_q = \langle E \rangle_q - T S_q
$$

그러면:

$$
F_q = F_{\text{exact}} + T \cdot D_{\text{KL}}(q_\theta \| p_\beta) \geq F_{\text{exact}}
$$

$D_{\text{KL}} \geq 0$이므로 $F_q \geq F$이고, **등호 조건은 $q_\theta = p_\beta$일 때**입니다.

**VAE ELBO vs VaTD 변분 자유에너지의 명시적 비교**:

$$
\boxed{
\begin{aligned}
\text{VAE:} \quad & -\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[-\log p_\theta(x|z)]}_{\text{재구성 오차}} + \underbrace{D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\text{사전분포 정칙화}} \\[6pt]
\text{VaTD:} \quad & \beta F_q = \underbrace{\mathbb{E}_{q_\theta}[\beta E(\mathbf{x})]}_{\text{평균 에너지} \leftrightarrow \text{재구성 오차}} + \underbrace{\mathbb{E}_{q_\theta}[\log q_\theta(\mathbf{x})]}_{\text{음의 엔트로피} \leftrightarrow \text{KL 정칙화}}
\end{aligned}
}
$$

| 측면 | VAE | VaTD |
|---|---|---|
| 목표 | 데이터 분포 $p_{\text{data}}(x)$ 근사 | Boltzmann 분포 $p_\beta(\mathbf{x})$ 근사 |
| 변분 분포 | 인코더 $q_\phi(z\|x)$ (잠재 공간) | 자기회귀 모델 $q_\theta(\mathbf{x})$ (배위 공간) |
| 부등식 | $\log p(x) \geq \text{ELBO}$ | $F_q \geq F_{\text{exact}}$ |
| 등호 조건 | 완벽한 인코더/디코더 | $q_\theta = p_\beta$ |
| 훈련 데이터 | **필요** ($x \sim p_{\text{data}}$) | **불필요** (에너지 함수만 있으면 됨) |

### 10.3 훈련 목적함수

$$
\mathcal{L}(\theta; \beta) = \mathbb{E}_{\mathbf{x} \sim q_\theta}\!\left[\log q_\theta(\mathbf{x}) + \beta E(\mathbf{x})\right]
$$

이 손실은 $D_{\text{KL}}(q_\theta \| p_\beta)$와 상수($\log Z$)만큼 차이납니다. 최소화하면:

- $\beta E(\mathbf{x})$ 항: 저에너지 배위에 높은 확률 부여
- $\log q_\theta(\mathbf{x})$ 항: 분포의 엔트로피 최대화 (모드 붕괴 방지)

### 10.4 분배함수 추정

학습 후, 변분 자유에너지는 참 자유에너지의 **상계(upper bound)**를 제공합니다:

$$
Z_q(\beta) = e^{-\beta F_q(\beta)} \leq Z(\beta)
$$

`vatd_exact_partition.py`의 정확한 $Z$와 비교하여 모델의 학습 품질을 평가할 수 있습니다.

---

## 11. REINFORCE와 RLOO 기울기 추정

### 11.1 이산 샘플링의 문제

> **ML 전문가를 위한 핵심**: 이진 스핀 $s_i \in \{-1, +1\}$은 **이산적**이므로, 재매개변수화 트릭(reparameterization trick)을 사용할 수 없습니다.

변분 손실의 기울기를 구하려면:

$$
\nabla_\theta \mathcal{L} = \nabla_\theta \mathbb{E}_{q_\theta}\![f(\mathbf{x}, \theta)]
$$

여기서 $f(\mathbf{x}, \theta) = \log q_\theta(\mathbf{x}) + \beta E(\mathbf{x})$입니다. 기댓값이 $\theta$에 의존하는 분포 $q_\theta$ 위에서 취해지므로, 기울기가 분포 자체를 통과해야 합니다.

### 11.2 REINFORCE (로그 미분 트릭)

Williams (1992)의 정책 기울기(policy gradient) 추정기:

$$
\nabla_\theta \mathbb{E}_{q_\theta}[f(\mathbf{x})] = \mathbb{E}_{q_\theta}\!\left[f(\mathbf{x}) \cdot \nabla_\theta \log q_\theta(\mathbf{x})\right]
$$

$\nabla_\theta \log q_\theta$는 **스코어 함수(score function)**입니다. $f(\mathbf{x})$는 "보상 신호"로 작용하여 스코어 함수를 가중합니다. 이 추정기는 **비편향(unbiased)**이지만 **높은 분산**을 가집니다.

### 11.3 기준선(Baseline)을 통한 분산 감소

핵심 성질: $\mathbb{E}_{q_\theta}[\nabla_\theta \log q_\theta(\mathbf{x})] = 0$. 따라서 상수 $b$를 빼도 기울기는 비편향:

$$
\nabla_\theta \mathcal{L} = \mathbb{E}_{q_\theta}\!\left[(f(\mathbf{x}) - b) \cdot \nabla_\theta \log q_\theta(\mathbf{x})\right]
$$

$b \approx \mathbb{E}[f]$를 선택하면 분산이 극적으로 감소합니다. 이것은 RL에서 actor-critic의 가치 함수 기준선과 동일한 역할입니다.

### 11.4 RLOO (Leave-One-Out) 기준선

배치 내 $B$개 샘플에서, 샘플 $b$의 기준선을 **나머지 모든 샘플의 평균**으로 설정:

$$
b^{(b)} = \frac{1}{B-1}\sum_{j \neq b} f(\mathbf{x}^{(j)})
$$

어드밴티지(advantage):

$$
A^{(b)} = f(\mathbf{x}^{(b)}) - \frac{1}{B-1}\sum_{j \neq b} f(\mathbf{x}^{(j)})
$$

**효율적 계산**: 전체 합 $S = \sum_b f^{(b)}$를 미리 구하면 $b^{(b)} = (S - f^{(b)}) / (B-1)$로 $O(B)$ 계산.

**배치 평균 기준선과의 차이**: 배치 평균은 $b = S/B$로, 샘플 $b$ 자신이 자기 기준선에 포함되어 미묘한 상관을 만듭니다. RLOO는 자기 자신을 제외하므로 **정확히 비편향**입니다.

### 11.5 온도별 RLOO

다중 온도 훈련에서, 다른 온도의 샘플은 매우 다른 보상 스케일을 가집니다 (높은 $\beta$는 큰 $|\beta E|$). 전체 기준선은 고온 샘플에 대한 분산 감소가 불충분합니다. 해법: **온도 그룹별로 독립적인 RLOO 기준선** 계산.

$$
A^{(b)}_k = f(\mathbf{x}^{(b)}, \beta_k) - \frac{1}{B_k - 1}\sum_{\substack{j \neq b \\ \beta_j = \beta_k}} f(\mathbf{x}^{(j)}, \beta_k)
$$

### 11.6 RL과의 대응

| 강화학습 개념 | 통계역학 대응 |
|---|---|
| 정책 $\pi_\theta(a\|s)$ | 자기회귀 모델 $q_\theta(\mathbf{x}\|\beta)$ |
| 행동 시퀀스 | 스핀 배위 $\mathbf{x}$ |
| 보상 $R$ | 음의 손실: $-(\log q_\theta + \beta E)$ |
| 기준선 $b$ | RLOO 기준선 |
| REINFORCE 기울기 | 스코어 함수 $\times$ 어드밴티지 |

### 11.7 현대 REINFORCE 변종들과의 비교

VaTD의 온도별 RLOO는 최근 LLM 정렬(alignment) 연구에서 등장한 REINFORCE 변종들과 구조적으로 관련됩니다:

| 방법 | 기준선 구성 | 정규화 | 핵심 아이디어 |
|------|-----------|--------|------------|
| **RLOO (VaTD)** | 온도 그룹 내 LOO 평균 | 온도별 독립 | 같은 물리 조건의 샘플끼리 비교 |
| **GRPO** (DeepSeek, 2024) | 그룹 내 평균 + 표준편차 정규화 | $A_i = (r_i - \mu_G) / \sigma_G$ | 그룹별 z-score 정규화 |
| **REINFORCE++** (Hu et al., 2025) | 글로벌 이동 평균 | 토큰 수준 KL 페널티 | 실시간 기준선 + KL 정칙화 |
| **James-Stein baseline** (2025) | Shrinkage: $b^* = \lambda \bar{f}_{\text{group}} + (1-\lambda) \bar{f}_{\text{global}}$ | 최적 보간 | 그룹 vs 글로벌 기준선의 최적 혼합 |

**VaTD의 온도별 RLOO는 어디에 위치하는가?**

VaTD의 접근은 GRPO와 가장 유사합니다: 둘 다 **그룹별 독립적인 기준선**을 사용합니다.

- GRPO에서 "그룹" = 같은 프롬프트에 대한 여러 응답
- VaTD에서 "그룹" = 같은 온도 $\beta_k$에 대한 여러 배위

차이점: GRPO는 표준편차로 정규화하지만, VaTD는 정규화하지 않습니다. 물리적으로 이것은 정당합니다: 같은 온도의 에너지 스케일은 이미 $\beta$에 의해 결정되므로, 추가 정규화가 물리적 신호를 왜곡할 수 있습니다.

James-Stein shrinkage baseline은 향후 개선 방향을 제시합니다: 온도별 기준선(그룹)과 전체 기준선(글로벌)을 최적 비율로 혼합하면, 그룹 크기가 작을 때 (= 온도당 샘플 수가 적을 때) 분산을 더 줄일 수 있습니다.

---

## 12. 핵심 연결: 왜 임계점에서 Low-Rank인가?

이 섹션은 `LOW_RANK.md`의 결과를 이해하기 위한 모든 배경지식을 통합합니다.

### 12.1 세 가지 온도 레짐에서의 신경망 표현

Boltzmann 분포 $p(\mathbf{x} \mid T)$를 학습하는 신경망의 내부 표현을 생각합니다:

**고온 ($T \gg T_c$):**
- 물리: 스핀이 열적으로 비상관 ($\xi \ll 1$). 각 스핀은 독립적 노이즈.
- 신경망: 구조 없는 고엔트로피 배위를 표현하려면 많은 독립 특성 채널이 필요 → **높은 랭크**
- 비유: 고차원 공간을 균일하게 채우는 데이터

**임계 온도 ($T \approx T_c$):**
- 물리: 상관길이 발산 ($\xi \sim L$). 전체 격자가 소수의 스케일링 장(자화 밀도 $\phi$, 에너지 밀도 $\epsilon$)에 의해 지배. CFT의 관련 연산자 **2개** (+ 항등원).
- 신경망: 특성 맵이 **고도로 동선(collinear)**해짐 — 대부분의 채널이 같은 전역 모드에 대한 중복 정보를 인코딩 → **낮은 랭크**
- 비유: 저차원 다양체(manifold) 위에 놓인 데이터

**저온 ($T \ll T_c$):**
- 물리: 대칭 깨진 바닥 상태 ($+M$ 또는 $-M$) 중 하나에 동결. 단순하지만, 배치 내에 두 섹터가 공존.
- 신경망: 부분적 랭크 회복 ($\sim 8$–$12$), 도메인 벽 여기도 기여
- 비유: 두 개의 뚜렷한 클러스터를 가진 데이터

### 12.2 압축 역설 (Compression Paradox)

`LOW_RANK.md`의 가장 인상적인 결과:

> **활성화 랭크는 $T_c$에서 최소**이지만, **가중치 압축 민감도는 $T_c$에서 최대**입니다.

이 역설은 **데이터 다양체(data manifold)**와 **변환 민감도(transformation sensitivity)**의 구별로 해소됩니다:

- **낮은 활성화 랭크**: 데이터가 저차원 다양체 위에 놓여 있음 → 연필이 세로로 서 있는 상태 (3D 공간에서의 1D 상태)
- **높은 압축 민감도**: 데이터를 그 다양체 위에 유지하려면 변환(가중치)이 극도로 정밀해야 함 → 연필의 균형을 유지하기 위한 미세한 운동 제어

**물리적 대응**: 이것은 임계점에서의 **감수율 발산**과 직접 연결됩니다:

$$
\chi = \frac{\partial \langle m \rangle}{\partial h}\bigg|_{h=0} \sim |T - T_c|^{-7/4}
$$

가중치 절단은 모델의 유효 해밀토니안에 대한 "섭동"을 도입하고, 그 응답(열화)은 $T_c$에서 $\chi$와 함께 발산합니다.

#### Information Bottleneck Theory 관점

이 압축 역설은 **Information Bottleneck (IB) Theory** (Tishby et al., 2000; Shwartz-Ziv & Tishby, 2017)의 관점에서 더 깊이 이해됩니다:

IB 원리는 표현 $T$가 입력 $X$에 대한 압축 $I(X; T)$을 최소화하면서, 출력 $Y$에 대한 예측 $I(T; Y)$를 최대화하는 최적 트레이드오프를 찾습니다:

$$
\min_{p(t|x)} \; I(X; T) - \beta_{\text{IB}} \cdot I(T; Y)
$$

임계점은 이 트레이드오프의 **최적 지점**에 해당합니다:
- **최대 압축** (낮은 eRank): 관련 없는 자유도가 모두 제거됨 → $I(X; T)$ 최소
- **최대 예측력** (높은 압축 민감도): 남은 자유도가 물리의 핵심을 포착 → $I(T; Y)$ 최대
- 더 이상 압축하면 핵심 정보가 손실 → 민감도 발산

이것은 **적대적 강건성(adversarial robustness)**과도 연결됩니다: 임계점 표현은 가장 효율적이지만, 가중치에 대한 작은 섭동(적대적 공격)에 가장 취약합니다. 마치 연필을 세우는 것처럼, 최적 균형점이 곧 최대 불안정점입니다.

### 12.3 $d(\text{eRank})/dT$와 $C_v$의 관계

비열은 에너지 요동의 **크기**를 측정합니다:

$$
C_v(T) \sim \text{요동의 크기 (magnitude)}
$$

유효 랭크는 요동의 **차원수**를 측정합니다:

$$
\text{eRank}(T) \sim \text{요동의 차원수 (dimensionality)}
$$

임계점에서 요동은 **거대하지만 모두 같은 방향**입니다: $C_v \uparrow$ (큰 요동) 이면서 $\text{eRank} \downarrow$ (저차원 요동). 이것이 반상관(anti-correlation)의 물리적 기원입니다.

### 12.4 RG와 층별 랭크 구조

eRank 히트맵(`LOW_RANK.md` Figure 1d)에서 **초기 층에서 랭크 하락이 가장 깊은** 패턴은 RG 흐름을 반영합니다:

- **Block 0-1**: UV 자유도를 적분해 내는 조대화 수행. 임계점에서 무관한(irrelevant) 연산자가 소멸하므로 유효 차원이 급격히 감소.
- **Block 2-3**: 이미 재규격화된 IR 변수를 다룸. 관련(relevant) 연산자만 남은 상태.

2D Ising 임계 고정점에서 관련 연산자는 정확히 **2개** ($\epsilon$과 $\sigma$)입니다. eRank가 $T_c$에서 $\sim 3$–$5$로 떨어지는 관측은, 네트워크가 이 소수의 관련 방향과 약간의 보정을 발견하는 것과 일치합니다.

#### Rank Diminishing Theorem과의 연결

Huang et al. (2022)은 심층 신경망에서 **층을 지나며 활성화 랭크가 단조 감소하는 보편적 성질**을 증명했습니다:

$$
\text{rank}(f_{l+1}(X)) \leq \text{rank}(f_l(X))
$$

여기서 $f_l(X)$는 $l$번째 층의 활성화입니다. 이 정리는 ReLU와 같은 비선형 활성화 함수가 행렬의 랭크를 보존하거나 감소시키는 성질에 기반합니다.

이것은 **RG 흐름의 신경망 내부 반영**입니다:
- RG: 각 조대화 스텝에서 무관 연산자가 소멸 → 유효 자유도 감소
- 신경망: 각 층에서 활성화 랭크가 감소 → 독립 특성 채널 감소
- 두 경우 모두 깊이(또는 스케일)가 증가할수록 **관련 정보만 남는** 정보 압축 과정

본 프로젝트의 eRank 히트맵에서 관찰되는 "초기 층에서 급격한 랭크 하락, 후기 층에서 안정화" 패턴은 이 정리의 직접적 반영입니다.

### 12.5 유한 크기 효과

$L = 16$에서의 주요 효과:

| 효과 | 설명 | 정량적 크기 |
|------|------|------------|
| 유사임계 이동 | $T_c(L) = T_c + a \cdot L^{-1/\nu}$ | $\sim 6\%$ ($L^{-1} = 1/16$) |
| 비열 둥글림 | $C_v^{\max} \sim \ln(L) \approx 2.77$ | 완만한 피크 |
| 상관길이 포화 | $\xi \sim L = 16$ | 격자 너머 상관 불가 |
| 수용 영역 조건 | RF $= 37 \times 37 > 16 \times 16$ | 충족됨 |

$L = 16$은 다음과 같은 이유로 선택되었습니다:
1. Onsager의 정확한 $Z$가 유한 $L$에 대해 계산 가능
2. $N = 256$ 스핀의 순차 샘플링이 관리 가능
3. 수용 영역이 격자를 완전히 포괄
4. 계산물리학 벤치마크로 널리 사용됨

### 12.6 실용적 함의: 신경망 설계에 대한 교훈

본 프로젝트의 결과에서 도출되는, 물리-정보 기반 신경망 설계의 실용적 지침을 정리합니다:

#### 임계점 근처 학습이 어려운 이유: Critical Slowing Down

물리학에서 **critical slowing down**은 임계점 근방에서 시스템의 이완 시간(relaxation time)이 발산하는 현상입니다:

$$
\tau \sim \xi^z \sim |T - T_c|^{-z\nu}
$$

여기서 $z$는 동적 임계지수입니다. 신경망 학습에서 이것은 **Hessian의 조건수(condition number) 악화**로 나타납니다:

- 임계점에서 손실 지형(loss landscape)의 곡률이 극도로 비등방적
- 가장 급한 방향(관련 연산자)과 가장 완만한 방향(무관 연산자)의 곡률 비율이 발산
- 이는 경사 하강법의 수렴 속도를 극적으로 저하시킴

**실용적 대응**: 본 프로젝트에서 사용하는 2-phase curriculum (`curriculum_enabled: true`)은 이 문제의 물리적으로 동기부여된 해결책입니다.

#### Curriculum Learning의 물리적 정당화

2-phase curriculum (Phase 1: 고온만 → Phase 2: 점진적으로 $\beta_{\max}$ 확장)의 물리적 근거:

1. **고온 = 쉬운 학습**: $T \gg T_c$에서 스핀이 거의 독립이므로, 독립 Bernoulli에 가까운 단순한 분포 → 빠른 초기 수렴
2. **점진적 저온 확장**: 고온에서 학습한 가중치가 저온 학습의 좋은 초기화 제공 — RG 관점에서, 고온 고정점의 근방에서 학습을 시작하여 임계 고정점으로 이동
3. **임계점 회피**: 임계점을 직접 학습하지 않고, 양쪽(고온/저온)에서 점진적으로 접근하여 critical slowing down을 완화

이것은 시뮬레이티드 어닐링(simulated annealing)의 역방향 버전입니다: 어닐링은 고온에서 저온으로 냉각하여 최적해를 찾지만, curriculum learning은 쉬운 문제에서 어려운 문제로 확장하여 학습을 안정화합니다.

#### 확장 합성곱의 필요성: $\xi$ 발산 → RF $\geq L$

상관길이의 온도 의존성으로부터 수용 영역 요구사항이 직접 도출됩니다:

$$
\xi(T) \sim |T - T_c|^{-\nu} \xrightarrow{T \to T_c} L \quad \Longrightarrow \quad \text{RF} \geq L
$$

| 아키텍처 | RF ($3 \times 3$ 커널, 8층) | $L = 16$ 커버 | $L = 32$ 커버 |
|---|---|---|---|
| 일반 CNN (dilation = 1) | $17 \times 17$ | 겨우 충족 | 불충분 |
| 확장 CNN (`[1,2,4,8]`) | $61 \times 61$ | 여유 있음 | 충족 |
| 확장 CNN (`[1,2,4,8,16]`) | $125 \times 125$ | — | 여유 있음 |

이것은 **격자 크기를 키울 때 자동으로 확장 패턴을 조정해야 하는** 설계 원칙을 제공합니다. 물리적으로 정당화된 아키텍처 선택입니다.

---

## 13. 용어 대조표

### 물리학 ↔ 머신러닝 대응

| 통계역학 / HEP | 머신러닝 | 본 프로젝트 |
|---|---|---|
| 해밀토니안 $H(\mathbf{x})$ | 에너지 함수 / 음의 로그 우도 | `energy(samples, beta)` |
| 역온도 $\beta = 1/T$ | (softmax 온도의 역수) | 훈련 파라미터 |
| Boltzmann 분포 $p \propto e^{-\beta H}$ | 에너지 기반 모델 (EBM) | 목표 분포 |
| 분배함수 $Z$ | 정규화 상수 | `vatd_exact_partition.py` |
| 자유에너지 $F = -T \ln Z$ | 음의 ELBO | 변분 자유에너지 |
| 질서변수 (자화 $M$) | 특성 (평균 픽셀 강도) | 상전이 지표 |
| 비열 $C_v$ | (해당 없음) | 에너지 분산의 스케일링 |
| 감수율 $\chi$ | 섭동에 대한 민감도 | 가중치 압축 열화 |
| 상관길이 $\xi$ | 특성의 유효 범위 | 수용 영역의 하한 |
| RG 흐름 | CNN 풀링 / 깊은 층 | 층별 랭크 구조 |
| 유클리드 작용 $S_E$ | 손실함수 (변분 손실) | $\log q + \beta E$ |
| 경로적분 $\int \mathcal{D}\phi$ | 배위 합 (Monte Carlo) | REINFORCE 샘플링 |
| 관련 연산자 (2개) | 주요 특성/주성분 | eRank $\approx 3$–$5$ at $T_c$ |
| 무관 연산자 | 노이즈/중복 특성 | SVD에서 작은 특이값 |
| Critical slowing down | Poor Hessian conditioning | Curriculum learning 필요성 |
| 보편성류 | Transfer learning | 도메인 무관 특성 |

### ML 전용 용어 해설

| 용어 | 정의 | 본 문서에서의 맥락 |
|---|---|---|
| **ELBO** (Evidence Lower Bound) | VAE의 손실함수. $\log p(x) \geq \text{ELBO}$. 재구성 오차 + KL 정칙화 | VaTD의 변분 자유에너지와 구조적 동치 (Section 1.2, 10.2) |
| **EBM** (Energy-Based Model) | 에너지 함수로 확률 정의: $p(x) \propto e^{-E(x)}$ | Boltzmann 분포와 동일 형태 (Section 1.3) |
| **LoRA** (Low-Rank Adaptation) | 가중치 업데이트를 저랭크 $\Delta W = BA$로 제한하는 미세조정 | Low-rank hypothesis의 직접 응용 (Section 7.7) |
| **Neural Collapse** | 분류기의 마지막 층에서 클래스 특성이 정규 심플렉스로 수렴하는 현상 | Rank collapse의 한 형태 (Section 5.4) |
| **Grokking** | 암기 후 갑자기 일반화가 발생하는 현상. Effective rank 급락과 동반 | 상전이와 유사한 급격한 전환 (Section 5.4) |
| **Information Bottleneck** | 압축과 예측의 최적 트레이드오프를 찾는 정보이론 원리 | 임계점 = 최적 압축-예측 균형 (Section 12.2) |
| **Implicit Rank Regularization** | GD가 명시적 정칙화 없이 저랭크 해를 선호하는 현상 | Low-rank bias의 최적화 기원 (Section 7.7) |
| **GRPO** (Group Relative Policy Optimization) | 그룹별 z-score 정규화를 사용하는 REINFORCE 변종 | VaTD의 온도별 RLOO와 구조적 유사 (Section 11.7) |

### HEP ↔ 응집물질 대응

| 고에너지 물리학 | 응집물질 / 통계역학 |
|---|---|
| 유클리드 작용 $S_E[\phi]$ | $\beta H(\mathbf{s})$ |
| 전역 $Z_2$ ($\phi \to -\phi$) | 스핀 뒤집기 ($s_i \to -s_i$) |
| Higgs VEV $\langle \phi \rangle = v$ | 자발 자화 $M$ |
| 질량 갭 $m$ | 역 상관길이 $1/\xi$ |
| 질량 없는 이론 ($m = 0$) | 임계점 ($\xi \to \infty$) |
| 달리는 결합상수 $g(\mu)$ | 블록 스핀 RG |
| 관련 연산자 | 임계 고정점의 불안정 방향 |
| 무관 연산자 | 보편성 (미시적 세부사항 무시) |
| 이상 차원 $\gamma_\phi$ | 임계지수 $\eta$ |
| OPE | 융합 규칙 |
| 등각 부트스트랩 | 임계지수 결정 |
| 중심 전하 $c$ | 자유도의 수 ($c = 1/2$: Majorana 페르미온 1개) |
| $\mathcal{N}=4$ SYM S-쌍대성 ($g \to 1/g$) | Kramers-Wannier 쌍대성 (고온 ↔ 저온) |

---

## 14. 참고문헌

### 원 논문

1. **Low-rank hypothesis**: Thibeault, V., Allard, A., & Desrosiers, P., "The low-rank hypothesis of complex systems," *Nature Physics* **20**, 294–302 (2024). [DOI](https://doi.org/10.1038/s41567-023-02303-0)

2. **Effective rank**: Roy, O. & Vetterli, M., "The effective rank: A measure of effective dimensionality," *15th European Signal Processing Conference (EUSIPCO)*, 606–610 (2007).

3. **Onsager 엄밀해**: Onsager, L., "Crystal statistics. I. A two-dimensional model with an order-disorder transition," *Physical Review* **65**, 117 (1944).

4. **VaTD (변분 자기회귀 네트워크)**: Wu, D., Wang, L., & Zhang, P., "Solving statistical mechanics using variational autoregressive networks," *Physical Review Letters* **122**, 080602 (2019).

5. **Kramers-Wannier 쌍대성**: Kramers, H.A. & Wannier, G.H., "Statistics of the Two-Dimensional Ferromagnet. Part I," *Physical Review* **60**, 252 (1941).

6. **자발 자화 증명**: Yang, C.N., "The spontaneous magnetization of a two-dimensional Ising model," *Physical Review* **85**, 808 (1952).

### 배경 교재 및 리뷰

7. **2D Ising CFT**: Di Francesco, P., Mathieu, P., & Sénéchal, D., *Conformal Field Theory*, Springer (1997). — Chapter 7: 최소 모델과 Ising 모델

8. **등각 부트스트랩**: El-Showk, S. et al., "Solving the 3D Ising model with the conformal bootstrap," *Physical Review D* **86**, 025022 (2012).

9. **REINFORCE**: Williams, R.J., "Simple statistical gradient-following algorithms for connectionist reinforcement learning," *Machine Learning* **8**, 229 (1992).

10. **RLOO baseline**: Kool, W., van Hoof, H., & Welling, M., "Buy 4 REINFORCE samples, get a baseline for free!" *ICLR Workshop* (2019).

11. **PixelCNN**: van den Oord, A. et al., "Conditional image generation with PixelCNN decoders," *NeurIPS* (2016).

12. **차원 축소와 네트워크 역학**: Thibeault, V. et al., "Threefold way to the dimension reduction of dynamics on networks," *Physical Review Research* **2**, 043215 (2020).

### 딥러닝-물리 연결

13. **CNN과 RG의 정확한 매핑**: Mehta, P. & Schwab, D.J., "An exact mapping between the variational renormalization group and deep learning," *arXiv:1410.3831* (2014).

14. **Implicit rank regularization**: Arora, S. et al., "Implicit regularization in deep matrix factorization," *NeurIPS* (2019).

15. **LoRA**: Hu, E.J. et al., "LoRA: Low-rank adaptation of large language models," *ICLR* (2022).

16. **Neural collapse**: Papyan, V. et al., "Prevalence of neural collapse during the terminal phase of deep learning training," *PNAS* **117**(40), 24652–24663 (2020).

17. **Grokking**: Power, A. et al., "Grokking: Generalization beyond overfitting on small algorithmic datasets," *arXiv:2201.02177* (2022).

18. **Information Bottleneck**: Shwartz-Ziv, R. & Tishby, N., "Opening the black box of deep neural networks via information," *arXiv:1703.00810* (2017).

19. **Rank diminishing theorem**: Huang, W. et al., "On the rank diminishing phenomenon in deep neural networks," *NeurIPS* (2022).

20. **AlphaPruning**: Lu, H. et al., "AlphaPruning: Using heavy-tailed self regularization theory for improved layer-wise pruning of large language models," *NeurIPS* (2024).

21. **Edge of chaos**: Poole, B. et al., "Exponential expressivity in deep neural networks through transient chaos," *NeurIPS* (2016).

22. **GRPO**: Shao, Z. et al., "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models," *arXiv:2402.03300* (2024).

23. **REINFORCE++**: Hu, J. et al., "REINFORCE++: A simple and efficient approach for aligning large language models," *arXiv:2501.03262* (2025).

24. **Roberts et al. (criticality in neural networks)**: Roberts, D.A. et al., "The principles of deep learning theory," *Cambridge University Press* (2022); *Phys. Rev. Research* (2025).
