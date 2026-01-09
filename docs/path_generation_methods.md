# PixelCNN Path Generation Methods for 2D Ising Model

이 문서는 VaTD 프로젝트에서 사용되는 autoregressive 샘플링 경로 생성 방식들을 정리합니다.

## Overview

PixelCNN은 autoregressive 모델로, 각 픽셀을 순차적으로 생성합니다. 생성 순서(경로)에 따라 샘플링 속도와 학습 품질이 달라질 수 있습니다.

| Path Type | Forward Passes | Speedup | Autoregressive Valid | 2D Locality |
|-----------|---------------|---------|---------------------|-------------|
| Raster | 256 | 1x (baseline) | Yes | Poor |
| **Diagonal** | **31** | **~8x** | **Yes** | Good |
| Hilbert | 256 | 1x | Experimental | **Best** |
| ~~Checkerboard~~ | 2K | ~100x | **No (broken)** | N/A |

---

## 1. Raster Scan (Default)

### 개념

표준 좌→우, 상→하 순서로 픽셀을 생성합니다.

```
Generation order for 4x4:
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

### 마스킹

`MaskedConv2D`는 현재 위치 (i, j)에서 다음만 볼 수 있습니다:
- 이전 행 전체: (0, *), (1, *), ..., (i-1, *)
- 현재 행의 왼쪽: (i, 0), (i, 1), ..., (i, j-1)

```python
# Spatial mask (3x3 kernel, center at (1,1))
# 1 = can see, 0 = masked
[[1, 1, 1],
 [1, 0, 0],  # Type A: center masked
 [0, 0, 0]]
```

### 장단점

**장점:**
- 구현 간단
- 완전한 autoregressive 일관성
- 모든 이전 정보 활용

**단점:**
- 256번의 순차적 forward pass 필요 (16x16 grid)
- 인접 픽셀 간 거리 불균일 (예: (0,15)와 (1,0)은 순서상 인접하지만 공간적으로 멀리 떨어짐)

### 사용법

```yaml
net_config:
  path_type: raster  # 기본값, 생략 가능
```

---

## 2. Diagonal Scan (Recommended)

### 개념

Anti-diagonal 순서로 픽셀을 생성합니다. 같은 대각선 내의 모든 픽셀은 **단일 forward pass**로 동시에 샘플링 가능합니다.

```
Generation order for 4x4:
Step 0: (0,0)                    → 1 pixel
Step 1: (0,1), (1,0)             → 2 pixels (parallel)
Step 2: (0,2), (1,1), (2,0)      → 3 pixels (parallel)
Step 3: (0,3), (1,2), (2,1), (3,0) → 4 pixels (parallel)
Step 4: (1,3), (2,2), (3,1)      → 3 pixels (parallel)
Step 5: (2,3), (3,2)             → 2 pixels (parallel)
Step 6: (3,3)                    → 1 pixel

Total: 7 steps (vs 16 for raster)
```

16x16 grid의 경우: **31 steps** (vs 256 for raster) → **~8x speedup**

### 마스킹

`DiagonalMaskedConv2D`는 대각선 인덱스 `d = i + j`를 기준으로 인과성을 정의합니다:
- 위치 (i, j)는 `i' + j' < i + j`인 모든 위치를 볼 수 있음
- Type B는 같은 대각선 (`i' + j' == i + j`)도 볼 수 있음

```python
# Diagonal mask (3x3 kernel, center at (1,1))
# diag_diff = (ky - 1) + (kx - 1)
[[1, 1, 0],   # diag_diff: -2, -1, 0
 [1, 0, 0],   # diag_diff: -1, 0, 1
 [0, 0, 0]]   # diag_diff: 0, 1, 2
# Type A: diag_diff == 0도 마스킹
```

### 장단점

**장점:**
- **~8x 속도 향상** (31 vs 256 forward passes)
- 완전한 autoregressive 일관성 유지
- 좋은 2D 국소성 (대각선 내 픽셀들은 공간적으로 인접)

**단점:**
- 첫 번째/마지막 대각선은 컨텍스트가 제한적
- DiagonalMaskedConv2D 구현 필요

### 구현

```python
class DiagonalMaskedConv2D(nn.Conv2d):
    """대각선 인과성을 가진 마스킹 컨볼루션."""

    def __init__(self, ..., mask_type):
        # Mask: diag_diff > 0이면 0, Type A는 diag_diff == 0도 0
        for ky in range(height):
            for kx in range(width):
                diag_diff = (ky - y_center) + (kx - x_center)
                if diag_diff > 0 or (diag_diff == 0 and mask_type == "A"):
                    mask[:, :, ky, kx] = 0
```

### 사용법

```yaml
net_config:
  path_type: diagonal
```

```bash
python main.py --run_config configs/v0.16/ising_pixelcnn_diagonal.yaml
```

---

## 3. Hilbert Curve Scan (Experimental)

### 개념

Hilbert 공간 채우기 곡선을 따라 픽셀을 생성합니다. 이 곡선은 **2D 국소성을 최적으로 보존**합니다 - 순서상 인접한 픽셀들이 공간적으로도 이웃입니다.

```
Hilbert curve for 4x4:
 0  1 14 15
 3  2 13 12
 4  7  8 11
 5  6  9 10

Path: (0,0)→(0,1)→(1,1)→(1,0)→(2,0)→(2,1)→(3,1)→(3,0)→...
```

### 마스킹 (실험적)

현재 구현에서는 **raster masking**을 그대로 사용하고, **샘플링 순서만** Hilbert 곡선을 따릅니다.

**한계점:**
- 네트워크의 receptive field는 raster 순서에 최적화되어 있음
- 샘플링 순서와 마스킹 순서의 불일치로 완전한 autoregressive 일관성이 없음
- 완전한 Hilbert masking은 256개의 다른 마스크가 필요 (복잡도 높음)

### 장단점

**장점:**
- 최고의 2D 국소성 (인접 샘플이 공간적 이웃)
- Ising 모델의 nearest-neighbor 상호작용 구조와 잘 맞음
- 임계 온도에서의 장거리 상관관계 학습에 유리할 수 있음

**단점:**
- 속도 향상 없음 (여전히 256 forward passes)
- 실험적 구현 (raster masking 사용)
- 완전한 autoregressive 일관성 없음

### 구현

```python
def hilbert_curve_order(n: int) -> List[Tuple[int, int]]:
    """Generate Hilbert curve indices for n x n grid."""
    def hilbert_d2xy(n, d):
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (d // 2)
            ry = 1 & (d ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            d //= 4
            s *= 2
        return x, y

    return [(y, x) for d in range(n * n) for x, y in [hilbert_d2xy(n, d)]]
```

### 사용법

```yaml
net_config:
  path_type: hilbert
```

```bash
python main.py --run_config configs/v0.16/ising_pixelcnn_hilbert.yaml
```

---

## 4. Checkerboard (Deprecated - 실패)

### 개념

2D Ising 모델의 bipartite 구조를 활용하여 병렬 샘플링을 시도했습니다.

```
Checkerboard pattern:
B W B W
W B W B
B W B W
W B W B

Sampling:
1. Black cells ~ p(black | white)  [parallel]
2. White cells ~ p(white | black)  [parallel]
3. Repeat K times (Gibbs refinement)
```

### 실패 원인

**1. Sampling-Log_prob 불일치 (치명적)**

```python
# Sampling (num_iterations > 1):
for k in range(K):
    x_black^{(k)} ~ p(x_black | x_white^{(k-1)})
    x_white^{(k)} ~ p(x_white | x_black^{(k)})

# Log_prob 계산:
log_prob = log p(x_black | x_white) + log p(x_white | x_black)
# → Pseudo-likelihood, NOT true generative probability!
```

이는 **다른 분포**를 모델링합니다:
- Sampling: K번 반복 후의 Gibbs equilibrium을 향해 수렴
- Log_prob: 단일 흑/백 조건부의 합 (pseudo-likelihood)

REINFORCE는 `log_prob`을 사용하여 그래디언트를 계산하므로, **편향된 그래디언트**가 발생합니다.

**2. Phase Conditioning 약함**

```python
# Checkerboard: phase 채널로 흑/백 구분
inp = torch.cat([x, T, phase], dim=1)  # phase: 0=black, 1=white

# 문제: 네트워크가 phase를 무시하고 학습할 수 있음
# Causal masking처럼 아키텍처 수준의 강한 제약이 아님
```

**3. Curriculum Learning 불일치**

Curriculum은 sequential PixelCNN용으로 설계되었으며, checkerboard의 반복 기반 refinement와 호환되지 않습니다.

### 교훈

- **Autoregressive 일관성**이 핵심: 샘플링과 log_prob 계산이 동일한 분포를 모델링해야 함
- **아키텍처 수준 제약**이 채널 기반 조건화보다 강력함
- 속도 향상을 위해 수학적 정확성을 희생하면 안 됨

---

## 성능 비교

### 샘플링 속도 (16x16 grid, CPU, batch_size=16)

| Path Type | Forward Passes | Time | Speedup |
|-----------|---------------|------|---------|
| Raster | 256 | 3.35s | 1.0x |
| **Diagonal** | **31** | **0.43s** | **7.8x** |
| Hilbert | 256 | 3.36s | 1.0x |

### 메모리 사용량

모든 방식이 동일한 메모리를 사용합니다 (동일한 네트워크 아키텍처).

### 학습 품질 (예상)

- **Raster**: 베이스라인
- **Diagonal**: 동등하거나 더 나음 (autoregressive 유효성 유지)
- **Hilbert**: 임계 온도에서 잠재적 개선 (실험 필요)

---

## 권장 사용 시나리오

### 빠른 실험/하이퍼파라미터 튜닝
→ **Diagonal Scan** 사용 (8x 속도 향상)

### 임계 온도 학습 연구
→ **Hilbert Curve** 실험 (국소성 효과 테스트)

### 안정적인 베이스라인
→ **Raster Scan** 사용 (검증된 방식)

---

## 설정 파일 위치

```
configs/v0.16/
├── ising_pixelcnn_diagonal.yaml   # Diagonal scan (8x faster)
├── ising_pixelcnn_hilbert.yaml    # Hilbert curve (best locality)
├── ising_pixelcnn_muon.yaml       # Raster scan + Muon optimizer
├── ising_pixelcnn_compact.yaml    # Raster scan + compact architecture
└── ising_pixelcnn_mhc.yaml        # Raster scan + MHC fusion
```

---

## 코드 위치

| Component | File | Line |
|-----------|------|------|
| `MaskedConv2D` | `model.py` | 100-185 |
| `DiagonalMaskedConv2D` | `model.py` | 188-291 |
| `hilbert_curve_order()` | `model.py` | 294-328 |
| `DiscretePixelCNN` | `model.py` | 549+ |
| `_sample_raster()` | `model.py` | 721-748 |
| `_sample_diagonal()` | `model.py` | 751-788 |
| `_sample_hilbert()` | `model.py` | 790-826 |

---

## 향후 개선 방향

1. **Hilbert Masking 완전 구현**: 각 위치에 대해 Hilbert 순서 기반 마스크 생성
2. **적응형 경로**: 온도에 따라 경로 선택 (고온: random, 저온: structured)
3. **MADE 스타일 임의 순서**: 배치마다 다른 순서로 학습하여 일반화 향상
4. **Z-Order (Morton) Curve**: Hilbert보다 간단하면서 유사한 국소성

---

## References

- [PixelCNN (van den Oord et al., 2016)](https://arxiv.org/abs/1601.06759)
- [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509)
- [Hilbert Space-Filling Curve](https://en.wikipedia.org/wiki/Hilbert_curve)
- [Neural Space-Filling Curves](https://arxiv.org/abs/2204.08453)
