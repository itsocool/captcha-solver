# CRNN + CTC Loss 아키텍처

> CAPTCHA 인식 기반 모델: Convolutional Recurrent Neural Network + Connectionist Temporal Classification

---

## 아키텍처 요약

```
입력 이미지 (1CH x H x W)
         │
    ┌─────────────┐
    │   CNN       │  특징 추출기
    │ (Feature    │  - Residual Connection
    │  Extractor) │  - GELU 활성화
    └──────┬──────┘  - BatchNorm
           │
    ┌─────────────┐
    │   LSTM      │  BiDirectional 2-Layer
    │ (Sequence   │  - LayerNorm
    │  Model)     │  - Dropout
    └──────┬──────┘
           │
    ┌─────────────┐
    │  Output     │  log probabilities (T, N, C)
    │  Projection │
    └──────┬──────┘
           │
    ┌─────────────┐
    │   CTC Loss  │  Focal / Standard
    └─────────────┘
```

---

## 1. CRNN 모델

CRNN는 **CNN**로 이미지를 특징 벡터로 변환하고, **RNN(BiLSTM)**로 시퀀스를 모델링한後, **CTC**로 텍스트를 출력하는 아키텍처입니다.

### 1.1 CNN Feature Extractor

3단계 블록 구조로, 각 블록에서 차원을 점진적으로 감소시킵니다.

```
Input: (N, 1, H, W) → Output: (N, 256, H/8, W/4)

Block 1: 1CH → 64CH  (stride=1) → Pool → H/2, W/2
Block 2: 64CH → 128CH (stride=1) → Pool → H/4, W/4
Block 3: 128CH → 256CH (stride=(2,1)) → Pool → H/8, W/4
```

블록당 구성:
```
Conv2d(3x3, padding=1) → BatchNorm2d → GELU → Conv2d(3x3, padding=1) → BatchNorm2d → GELU → MaxPool2d → Dropout2d
```

| 구성 요소 | 값 |
|-----------|-----|
| 인접 커널 | `3x3`, stride=1, padding=1 |
| 풀링 | `MaxPool2d(kernel_size=2)` 또는 `(2,1)` |
| 활성화 | `GELU` (ReLU 대비 부드럽고 안정적인 학습) |
| 정규화 | `BatchNorm2d` (차원별) |
| 드롭아웃 | `Dropout2d(dropout=0.1)` |
| 최종 채널 | 256 |

차원 변화 예시 (대법원 캡챠 120x40 기준):

```
Input:       (N, 1, 40, 120) → H/8=5, W/4=30 → (N, 256, 5, 30)
Feature dim: C x H = 256 x 5 = 1280 (LSTM 입력 차원)
Time steps:  W = 30 (CTC 레이어 수)
```

**CTC 요구사항**: Time steps (T) >= Label length. 입력 크기가 너무 작으면 학습 시 에러 발생.

### 1.2 Feature Projection

```
(N, W, 256*H) → Linear(256) → LayerNorm → GELU → Dropout
```

- CNN 출력을 시계열 형태로 재구성: `(N, C, H, W) → (N, W, C*H)`
- 차원 축소: `C*H = 1280` → `256`
- LayerNorm 적용 (LSTM 안정성 향상)

### 1.3 Bidirectional LSTM

```
Hidden Size: 128
Layers: 2 (bidirectional)
Input dim: 256
Output dim: 256 (128 x 2)
Batch first: True (cuDNN 최적화)
```

```python
self.rnn = nn.LSTM(
    input_size=256,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=dropout
)
```

---

## 2. CTC Loss

CTC (Connectionist Temporal Sequence) Loss는 이미지와 텍스트 레이블 간 **정렬 없이** 학습할 수 있는 손실 함수입니다.

### 2.1 표준 CTC Loss

```python
criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
```

| 파라미터 | 설명 |
|----------|-----|
| `blank=0` | CTC blank 토큰 인덱스 (0) |
| `reduction='mean'` | 배치 평균 손실 |
| `zero_infinity=True` | 무한대/NaN 방지 |

**CTC 동작 원리**:
- 이미지 프레임(시간 단계)마다 각 문자/blank의 확률을 예측
- blank는 연속된 동일 문자를 분리하거나 반복 문자를 처리
- `log_probs.log_softmax(2) → CTC Loss` 구조

### 2.2 Focal CTC Loss

어려운 샘플에 더 높은 가중치를 부여하는 손실 함수입니다.

```python
class FocalCTCLoss(nn.Module):
    def __init__(self, blank=0, gamma=2.0, alpha=0.25, reduction='mean'):
        ...
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        p = torch.exp(-ctc_loss)
        focal_weight = alpha * (1 - p) ** gamma
        return (focal_weight * ctc_loss).mean()
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|-----|
| `gamma` | 2.0 | 쉬운 샘플 가중치 감소 정도 |
| `alpha` | 0.25 | 클래스 불균형 보정 |

### 2.3 Loss 함수 비교

| Loss | 과적합 방지 | 어려운 샘플 강조 | 권장 시나리오 |
|------|------------|-----------------|--------------|
| `ctc` (기본) | - | - | 기본, 소규모 데이터 |
| `focal` | - | O | 클래스 불균형, 학습 어려움 |

---

## 3. PyTorchModel

PyTorch 기반 CAPTCHA 인식 모델 클래스.

### 3.1 초기화 및 설정

```python
model = PyTorchModel(
    captcha_type=captcha_type,
    verbose=1,
    device=None,              # None=auto (CUDA 우선)
    use_compile=False,        # torch.compile 사용 여부
    use_amp=True,             # Mixed Precision Training (AMP)
    loss_type='focal'         # 'ctc', 'focal'
)
```

**PyTorch 최적화 설정** (`core.py`):
```python
torch.backends.cudnn.benchmark = True  # 고정 입력 크기 기반 자동 튜닝
torch.backends.cuda.matmul.allow_tf32 = True  # Ampere GPU TF32 가속
torch.backends.cudnn.allow_tf32 = True
```

### 3.2 모델 빌드

```python
model.build_model(dropout=0.1)
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|-----|
| `dropout` | 0.1 | 드롭아웃 비율 |

torch.compile 사용 가능한 경우 모델 최적화 적용:
```python
if use_compile:
    model = torch.compile(model, mode='reduce-overhead')
```

### 3.3 학습 파이프라인

```python
model.train_model(
    train_loader=train_loader,
    val_loader=val_loader,           # 검증용 (선택)
    epochs=50,
    lr=1e-4,
    warmup_epochs=5,
    early_stopping_patience=0,       # 0=비활성화
    weight_decay=1e-4,
    grad_clip=5.0,
    loss_type='focal',
    dropout=0.1
)
```

**학습 특징**:
- **옵티마이저**: AdamW, fused=True (CUDA 시), weight_decay
- **학습률 스케줄러**: Warmup → ReduceLROnPlateau (factor=0.5, patience=3)
- **Mixed Precision**: GradScaler + autocast (CUDA 시)
- **Gradient Clipping**: max_norm=5.0
- **Early Stopping**: 검증 손실 기반 (사용 시 선택)
- **모델 저장**: 최종 모델 + TorchScript (.jit) + ONNX

### 3.4 추론 파이프라인

**단일 이미지**:
```python
pred_text, confidence = model.predict(image_path, beam_width=10, length_bonus=0.5)
```

**배치 추론**:
```python
results = model.predict_batch(image_paths, batch_size=32)
# [(pred_text, confidence), ...]
```

**추론 특징**:
- 고정 길이 Beam Search CTC 디코딩
- AMP (torch.inference_mode, autocast)
- 검증에 적합한 길이 기반 보너스 시스템

---

## 4. CTC Beam Search 디코딩

CTC의 빈번한 중복 문자 문제를 해결하고, 고정 길이를 위한 보너스 시스템을 제공합니다.

```python
def ctc_beam_decode_fixed_length(
    log_probs: np.ndarray,       # (T, num_classes)
    mapping_inv: Dict[int, str], # index -> character
    expected_length: int,        # 예상 레이블 길이
    beam_width: int = 10,
    unk_token: str = "[UNK]",
    top_k: int = 0,
) -> Tuple[str, float]:
```

**디코딩 흐름**:
1. 타임스텝 순으로 beam 확장 (blank, 동일문자, 신규문자)
2. 기대 길이를 초과하거나 남은 프레임으로 채울 수 없는 prefix 가지치기
3. prefix 점수는 blank 종료 경로와 문자 종료 경로를 log-sum-exp 로 합산한 값
   — 즉 단일 정렬 경로가 아니라 P(문자열 | 이미지)
4. 최종 결과에서 예상 길이에 정확히 매칭되는 beam 선택

**신뢰도 계산**: 길이 제약으로 조건부화한 사후확률
```python
confidence = exp(best_score - length_logprob(log_probs, expected_length))
#           = P(예측 문자열 | 이미지, 길이 = expected_length)
```

분모 `length_logprob()` 는 길이가 정확히 L 인 **모든** 문자열의 확률 합을 O(T·L·C) DP 로
정확히 구한다. 살아남은 beam 집합으로 정규화하면 1위와 2위의 비율만 재게 되어,
`beam_width` 를 바꿀 때마다 값이 흔들리고 학습 분포 밖 입력에도 0.5 언저리를 돌려준다.

---

## 5. 데이터 흐름

### 5.1 전처리 (dataclass.py:174-224)

```
원본 이미지 (RGB/RGBA) → 그레이스케일 → 임계값 처리 → 리사이즈 (WxH)
```

- **기본 전처리**: 투명 → 흰색 배경, grayscale, thresholding, 리사이즈
- **대법원 캡차**: crop + padding + grayscale + resize (120x40 고정)

### 5.2 학습 Transform

```python
get_train_transform(train_data)
# T.Compose([
#     Lambda(image_pre_process),
#     RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.97, 1.03), shear=2),
#     RandomApply(GaussianBlur),     p=0.3
#     RandomApply(ColorJitter),      p=0.2
#     RandomErasing,                 p=0.1
# ])
```

### 5.3 데이터셋

```python
dataset = CaptchaDataset(df, path, mapping, transform)
# __getitem__: 이미지 로드 → transform → 레이블 텐서
# 레이블 = 파일명(확장자 제외)을 char_to_idx로 매핑한 토큰 시퀀스
```

### 5.4 데이터 분할

```python
train_loader, val_loader = model.split_dataset(
    batch_size=16,
    train_size=0.8,
    num_workers=0,
    pin_memory=True,  # CUDA 시
)
# train: 80% / val: 20% (shuffle=True)
```

---

## 6. 파일 저장 형식

| 형식 | 파일명 | 생성 |
|------|--------|------|
| PyTorch StateDict | `model.pth` | 학습 종료 시 (`save_model`) |
| torch.export 아카이브 | `model.pt2` | 학습 종료 시 (`export_pt2`) |
| ONNX | `model.onnx` | 학습 종료 시 (`export_onnx`) |

TorchScript/ONNX: inference 전용 forward(wrapper)로 export.

---

## 7. 사용 예시

```python
from engine import get_captcha_model, train_model, predict

# 1. 모델 생성
model = get_captcha_model(captcha_id="supreme_court")

# 2. 학습 (Focal Loss)
train_model(
    model=model,
    epochs=60,
    batch_size=32,
    learning_rate=1e-4,
    loss_type='focal',      # 'ctc', 'focal'
    use_amp=True,
)

# 3. 추론
pred_text, confidence = predict(
    model=model,
    image_path="captcha_data/supreme_court/1/images/pred/sample.png"
)
print(f"예측: {pred_text} (신뢰도: {confidence:.4f})")
```
