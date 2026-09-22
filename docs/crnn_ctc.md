# CRNN + CTC Loss 아키텍처

> CAPTCHA 인식 기반 모델: Convolutional Recurrent Neural Network + Connectionist Temporal Classification
>
> 기준 구현: [`aso_ai/core.py`](../packages/python_3.13/src/aso_ai/core.py).
> 호출·데이터 계약: [`web/core/engine.py`](../apps/web/core/engine.py), [`web/core/dataclass.py`](../apps/web/core/dataclass.py).
> 이 문서는 함수 본문과 분기 조건을 기준으로 작성했다. 웹 서비스와 모델 구현의 연결은
> [web-architecture.md](./web-architecture.md), 데이터 구조는 [web-domain-model.md](./web-domain-model.md)를 참고한다.

모델·손실·디코더는 `aso_ai.core`, 캡차 레지스트리와 실행 진입점은 `web.core.engine`,
전처리·경로·감지 정보는 `web.core.dataclass`가 소유한다. `aso_ai.core`의 데이터 타입 import는
`TYPE_CHECKING` 안에 있고, 실행 중에는 전달받은 데이터 객체를 사용한다.
sidecar 캐시 갱신도 `TrainData.apply_meta()`에 위임한다. 웹 서비스는 `web.core.engine`을 함수 안에서 지연 import한다.

---

## 아키텍처 요약

```
입력 이미지 (1CH x H x W)  ← image_pre_process() 를 거친 그레이스케일, 0~1 float
         │
    ┌─────────────┐
    │   CNN       │  특징 추출기 (3 블록)
    │ (Feature    │  - Conv3x3 x2 + BatchNorm + GELU
    │  Extractor) │  - MaxPool (2,2) (2,2) (2,1)
    └──────┬──────┘  - Dropout2d (블록 1, 2)
           │  (N, 256, H/8, W/4) → (N, T=W/4, 256·H/8)
    ┌─────────────┐
    │  Feature    │  Linear(C·H → 256) → LayerNorm → GELU → Dropout
    │  Projection │
    └──────┬──────┘
           │  (N, T, 256)  ── 학습 시 SpecAugment (time/freq 마스킹)
    ┌─────────────┐
    │   LSTM      │  BiDirectional 2-Layer, hidden 128
    │ (Sequence   │  batch_first=True
    │  Model)     │
    └──────┬──────┘
           │  (N, T, 256)
    ┌─────────────┐
    │  Output     │  Linear(256→128) → GELU → Dropout → Linear(128→C+1)
    │  Projection │  → permute → (T, N, C+1)  (+1 = blank, index 0)
    └──────┬──────┘
           │
    ┌─────────────┐
    │ Focal CTC   │  학습 시 log_softmax 후 FocalCTCLoss (표준 CTC 는 제거됨)
    └─────────────┘
```

---

## 1. CRNN 모델 (`core.py` `class CRNN`)

CRNN은 **CNN**으로 이미지를 특징 시퀀스로 바꾸고, **BiLSTM**으로 시퀀스를 모델링한 뒤, **CTC**로 텍스트를 출력하는 아키텍처다.
입력 크기는 고정이며(전처리에서 리사이즈/크롭으로 보장), 생성자에서 더미 입력을 한 번 흘려 `feature_dim`과 `time_steps`를 계산한다.
아래 `H/8`, `W/4` 표기는 풀링 결과를 줄여 쓴 것이다. 실제 크기는 각각 `H // 8`, `W // 4`다.

### 1.1 CNN Feature Extractor

```
Input: (N, 1, H, W) → Output: (N, 256, H/8, W/4)

Block 1: 1CH  → 64CH  (Conv3x3 x2) → MaxPool(2,2) → Dropout2d   # H/2, W/2
Block 2: 64CH → 128CH (Conv3x3 x2) → MaxPool(2,2) → Dropout2d   # H/4, W/4
Block 3: 128CH→ 256CH (Conv3x3 x2) → MaxPool(2,1)               # H/8, W/4  (Dropout2d 없음)
```

블록당 구성:
```
Conv2d(3x3, stride=1, padding=1) → BatchNorm2d → GELU → Conv2d(3x3, stride=1, padding=1) → BatchNorm2d → GELU → MaxPool2d [→ Dropout2d]
```

| 구성 요소 | 값 |
|-----------|-----|
| 합성곱 커널 | `3x3`, stride=1, padding=1 (모든 Conv 동일; 다운샘플은 풀링만 담당) |
| 풀링 | `MaxPool2d(2, 2)` x2, `MaxPool2d((2,1), (2,1))` x1 |
| 활성화 | `GELU` |
| 정규화 | `BatchNorm2d` |
| 드롭아웃 | `Dropout2d(dropout=0.1)` — 블록 1, 2 뒤에만 |
| 최종 채널 | 256 |
| 가중치 초기화 | 이름에 `rnn`/`lstm`이 있는 2차원 이상 weight는 `orthogonal_`, `proj`/`linear`는 `xavier_uniform_`, 모든 bias는 0 |

> 코드 docstring 에 "Residual Connection" 이 언급돼 있으나 실제 `nn.Sequential` 에는 잔차 연결이 없다.

`_init_weights()`는 모듈 타입 대신 **파라미터 이름**으로 분기한다. Conv weight는 `cnn.0.weight` 같은
이름이라 `'conv'` 조건에 걸리지 않으며, `Conv2d` 생성 시 초기화된 값을 유지한다.
`kaiming_normal_(fan_out)` 호출 분기는 있지만 현재 CNN에는 적용되지 않는다.

차원 변화 예시 (대법원 캡차 120x40 기준):

```
Input:       (N, 1, 40, 120) → H/8=5, W/4=30 → (N, 256, 5, 30)
Feature dim: C x H = 256 x 5 = 1280 (Feature Projection 입력 차원)
Time steps:  T = W/4 = 30 (CTC 프레임 수)
```

**생성자 검사**: `label_length`가 주어지면 `time_steps >= label_length`를 확인하고 위반 시 `ValueError`를 낸다.
이 검사는 시퀀스 길이 하한만 확인하며 모든 타깃의 CTC 정렬 가능성을 보장하지는 않는다.

### 1.2 Feature Projection

```
(N, C, H, W) → permute/view → (N, W, C·H) → Linear(C·H → 256) → LayerNorm(256) → GELU → Dropout
```

- CNN 출력을 시계열로 재구성: 폭(W)이 시간축(T)이 된다
- 차원 축소: 예) `1280 → 256`

### 1.3 SpecAugment (학습 시에만)

Feature Projection 출력 `(N, T, 256)` 에 시간/주파수 마스킹을 건다 (`class SpecAugment`, `model.training` 일 때만 동작).

| 파라미터 | 값 |
|----------|-----|
| `time_mask_max_size` | 15 (단, `T // 2` 이하) |
| `time_mask_count` | 2 |
| `freq_mask_max_size` | 8 (단, `C // 2` 이하) |
| `freq_mask_count` | 2 |

`build_model()` 은 `spec_augment=True` 로 고정 생성한다.
실제 입력은 `(N, T, C)`이며 마스크 위치는 배치 전체가 공유한다. 특징 텐서의 선택 영역을
직접 0으로 바꾼다. `eval()`에서는 마스킹하지 않는다.

### 1.4 Bidirectional LSTM

```python
self.rnn = nn.LSTM(
    input_size=256,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=dropout,   # 0.1
)
# 출력: (N, T, 256)  (128 x 2 방향)
```

### 1.5 Output Projection 과 forward 반환

```python
self.output_proj = nn.Sequential(
    nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
    nn.Linear(128, num_classes + 1),   # +1 = blank
)
```

`forward(X, y=None, criterion=None)` 는 `(T, N, C+1)` 로짓과, `y`/`criterion` 이 주어졌을 때의 loss 를 함께 돌려준다.
loss 계산 시 `input_lengths` 는 모든 샘플에 `T`, `target_lengths` 는 모든 샘플에 `label_length`(고정 길이) 로 채워진다.
반환되는 `out` 은 **log_softmax 이전** 로짓이며, 추론 쪽에서 별도로 `log_softmax` 를 취한다.

문자 인덱스 규약: `char_to_idx` 는 1-based, `0 = blank`. `num_classes = len(문자셋)` 이고 출력층 크기는 `num_classes + 1` 이다.

---

## 2. Loss (`core.py` `class FocalCTCLoss`)

CTC (Connectionist Temporal Classification) Loss 는 이미지 프레임과 문자열 사이의 **정렬 없이** 학습할 수 있는 손실 함수다.
학습 진입점은 **Focal CTC만 지원**한다. `loss_type`은 전달값 → 인스턴스 속성 → `'focal'` 순서로
정하고, 그 결과가 `'focal'`이 아니면 `ValueError`를 낸다. Focal 계산 내부에서는 `nn.CTCLoss`를 사용한다.

### 2.1 Focal CTC Loss

```python
class FocalCTCLoss(nn.Module):
    def __init__(self, blank=0, gamma=2.0, alpha=0.25, reduction='mean', zero_infinity=True):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
        self.ctc = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)  # per-sample
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        per_sample_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        p = torch.exp(-per_sample_loss)
        focal_weight = self.alpha * (1 - p) ** self.gamma
        focal_loss = focal_weight * per_sample_loss
        return focal_loss.mean()   # reduction='mean'
```

| 파라미터 | 값 (학습 시) | 설명 |
|----------|--------------|------|
| `blank` | 0 | CTC blank 토큰 인덱스 |
| `gamma` | 2.0 | 쉬운 샘플 가중치 감소 정도. `train_model()` 이 `FocalCTCLoss(gamma=2.0)` 로 생성 |
| `alpha` | 0.25 (기본값) | 모든 샘플에 동일하게 곱하는 스칼라. 클래스별 가중치는 없음 |
| `zero_infinity` | True | 무한대 loss 를 0 으로 (도달 불가 정렬 방어) |
| 내부 CTC reduction | `'none'` | 샘플별 loss 를 받아 focal 가중치를 곱한 뒤 평균 |

위 코드는 기본 `reduction='mean'` 계산을 요약했다. 실제 클래스는 `'sum'`이면 합을,
그 외 값이면 샘플별 텐서를 반환한다. 길이로 나누는 추가 정규화는 없다.

**CTC 동작 원리**:
- 프레임(시간 단계)마다 각 문자/blank 의 확률을 예측
- blank 는 연속된 동일 문자를 분리하거나 반복 문자를 처리
- `out.log_softmax(2) → CTCLoss` 순서 (`CRNN.forward` 내부)

---

## 3. PyTorchModel (`core.py` `class PyTorchModel`)

PyTorch 기반 CAPTCHA 인식 모델 래퍼. `CaptchaType`(→ `TrainData`) 하나를 받아 데이터 로딩·학습·저장·추론을 담당한다.

### 3.1 초기화 및 설정

```python
model = PyTorchModel(
    captcha_type=captcha_type,   # web.core.dataclass.CaptchaType
    verbose=1,
    device=None,                 # None=auto (CUDA 가능하면 cuda, 아니면 cpu)
    use_compile=False,           # torch.compile 사용 여부
    use_amp=True,                # Mixed Precision (CUDA 에서만 실제 적용)
    loss_type='focal',           # 'focal' 만 유효
    model_dir=None,              # None 이면 TrainData.get_model_base_dir()
)
```

생성 시 문자 매핑을 만든다. 문자셋·라벨 길이·이미지 크기는 모두 `TrainData` 의 **감지값(`detected_*`)을 우선**한다 (§5.1).
`device`를 직접 넘길 때는 `torch.device`를 사용한다. 문자열 변환은 `engine.get_captcha_model()`이 담당한다.
`model_dir`는 속성에 저장되지만 `get_model_path()`·`get_model_base_dir()`는 `TrainData`에 위임하므로
이 인자만 바꿔서는 체크포인트나 export 경로가 바뀌지 않는다.

**PyTorch 전역 설정** (`core.py` import 시점):
```python
torch.backends.cudnn.benchmark = True         # 고정 입력 크기 자동 튜닝
torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+ TF32
torch.backends.cudnn.allow_tf32 = True
# CUDA가 있으면 conv2d를 한 번 실행하고, RuntimeError 발생 시 cuDNN을 끈다.
```

### 3.2 모델 빌드

```python
model.build_model(dropout=0.1)
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `dropout` | 0.1 | 드롭아웃 비율 (CNN Dropout2d, projection, LSTM 공통) |

`CRNN(in_channels=1, output=num_classes, img_height, img_width, label_length, dropout, spec_augment=True)` 를 만들어 디바이스로 옮긴다.
`use_compile=True` 면 CUDA 는 `mode='reduce-overhead'`, 그 외는 `'default'` 로 `torch.compile` 을 건다.

### 3.3 데이터 분할 (`split_dataset`)

```python
train_loader, val_loader = model.split_dataset(
    batch_size=16, train_size=0.8, shuffle=True,
    num_workers=0, pin_memory=True,          # pin_memory and torch.cuda.is_available()
    persistent_workers=None, prefetch_factor=2,   # num_workers > 0 일 때만 적용
)
```

- 대상: `images/train/*.png` 중 파일명 길이가 `detected_label_length` 와 같은 파일만 (`TrainData.get_data_files`)
- 라벨 = 파일명(확장자 제외) → `char_to_idx` 로 매핑한 정수 시퀀스 (매핑에 없는 문자는 버림)
- `sklearn.train_test_split(test_size=1-train_size, shuffle=shuffle)` 로 나눔 (고정 seed 없음)
- train 쪽은 증강 transform, val 쪽은 eval transform (§5.3)
- `DataLoader`: train `shuffle=True`, val `shuffle=False`

`engine.train_model()`은 `train_size=0.8, shuffle=True, pin_memory=False`로 호출하며
`num_workers`는 자신의 인자(기본 0)를 전달한다. 직접 `split_dataset()`을 호출할 때 pin-memory 판정은
모델의 선택 디바이스가 아니라 시스템의 CUDA 가용 여부를 따른다.

### 3.4 학습 파이프라인 (`train_model`)

```python
model.train_model(
    train_loader, val_loader=None,
    epochs=50, lr=1e-4,
    save_best=True, model_path=None,      # None → TrainData.get_model_path()
    warmup_epochs=5, early_stopping_patience=0,   # 0=비활성화
    weight_decay=1e-4, grad_clip=5.0,
    loss_type=None,                        # None → self.loss_type → 'focal'
    dropout=0.1,
    on_event=None,                         # 진행 콜백 (§3.5)
)
```

> 위는 `PyTorchModel.train_model` 의 기본값이다. 실제 진입점인 `engine.train_model()` 은 다른 기본값을 넘긴다 (§6.2).

**학습 특징**:

| 항목 | 내용 |
|------|------|
| 옵티마이저 | `AdamW(lr, weight_decay)`. `fused`는 `self.device.type == 'cuda' and hasattr(optim.AdamW, 'fused')`일 때만 True |
| 스케줄러 | **Linear Warmup → Cosine Annealing** (`LambdaLR`, 에폭 단위): `step < warmup` 이면 `(step+1)/warmup`, 이후 `0.5·(1+cos(π·progress))` |
| Mixed Precision | CUDA 이고 `use_amp=True` 일 때만 `GradScaler` + `autocast(float16)`. CPU 는 AMP 없이 fp32 |
| Gradient Clipping | `clip_grad_norm_(max_norm=grad_clip)` (`grad_clip > 0` 일 때, AMP 는 unscale 후) |
| 검증 | 매 에폭 `val_loader`로 loss 계산 (AMP 동일 적용). 개선되고 `save_best=True`이면 체크포인트 `.tmp` 저장 |
| val_loader 없을 때 | `save_best=True`이면 10에폭마다 `.tmp` 저장 |
| Early Stopping | `early_stopping_patience > 0` 이고 무개선이 patience 만큼 이어지면 `stop_reason='early_stopping'` |
| 중단 | `on_event` 가 `False` 를 돌려주면 `cancelled` (best 확정), `'discard'` 면 `cancelled_discarded` (`.tmp` 삭제, 기존 아티팩트 유지) |
| 덮어쓰기 가드 | 종료 후 디스크의 기존 `model.pth` 를 같은 `val_loader` 로 재평가(`_evaluate_checkpoint`). 기존이 더 좋으면(`incumbent <= best`) `.tmp` 를 버리고 `skipped` 이벤트로 끝냄. 비교 불가(구조 변경 등)면 그냥 덮어씀 |
| 확정 | `.tmp` 가 있으면 `os.replace` 로 `model.pth` 승격, 없으면 현재 가중치 저장 → `finalize_artifacts()` (§7) |

`fused` 판정은 생성자 인자 지원 여부가 아닌 **클래스 속성의 존재**를 검사한다. 따라서 CUDA라고 항상
활성화되는 것은 아니다. `_evaluate_checkpoint()`는 같은 검증 데이터를 사용하지만 autocast 없이 평가한다.

`train_hist` 반환값은 에폭 평균이 아닌 **학습 미니배치별 loss** 목록이다. 이벤트의 train/val loss는
각 배치 loss의 단순 평균이다. `epoch.lr`는 에폭 종료 시 `lr_scheduler.step()`을 호출한 **이후** 값이다.
`dropout` 인자는 `self.model is None`이라 모델을 새로 만들 때만 적용된다.
`save_best=False`여도 종료 후 체크포인트 확정·export는 수행한다.

체크포인트 저장은 `*.writing`에 쓴 뒤 `os.replace`로 교체한다. export 파일별 저장 방식은 §7을 참고한다.

### 3.5 진행 이벤트 (`on_event`)

`train_model(on_event=콜백)` 을 주면 dict 하나를 인자로 콜백이 호출된다. 웹앱(`apps/web/services/train.py`)이 이 콜백으로 SSE 를 만든다.
**`epoch` 이벤트**의 반환값이 `False` / `'discard'`일 때 다음 에폭으로 넘어가지 않는다 (§3.4).
미니배치 중간에는 콜백을 호출하지 않으며, `start`·`done`·`skipped`의 반환값은 사용하지 않는다.

| `type` | 시점 | 주요 payload 키 |
|--------|------|-----------------|
| `start` | 학습 시작 시 1회 | `captcha_id, rev, device, epochs, loss_type, batch_size, train_batches, val_batches, image_width, image_height, label_length, characters, lr, warmup_epochs, early_stopping_patience, use_amp, model_path` |
| `epoch` | 매 에폭 종료 후 | `epoch, epochs, train_loss, val_loss, lr, best_val_loss, best_epoch, improved, patience_counter, elapsed_sec` |
| `skipped` | 덮어쓰기 가드가 기존 모델을 유지할 때 (종료) | `reason='incumbent_better', incumbent_val_loss, best_val_loss, epochs_run, epochs, elapsed_sec` |
| `done` | 정상/조기종료/취소 종료 시 | `epochs_run, epochs, stop_reason` (`completed`\|`early_stopping`\|`cancelled`\|`cancelled_discarded`), `best_val_loss, best_epoch, elapsed_sec, artifacts` (경로 dict, discard 시 `{}`) |

`skipped`는 종료 이벤트이며 뒤에 `done`을 추가로 보내지 않는다. 저장·export·검증 예외가 발생하면
예외가 호출자에게 전달되고 `done`은 발생하지 않는다. 웹앱의 `shuffle`·`error` 이벤트는
`apps/web/services/train.py`에서 추가한다.
`skipped`·`cancelled_discarded`는 디스크의 기존 아티팩트를 유지하지만, 메모리의 학습 모델까지
기존 체크포인트로 되돌리지 않는다. 저장된 모델을 이어 추론하려면 `load_prediction_model()`로 다시 읽는다.

### 3.6 모델 로드 (`load_prediction_model`)

```python
model.load_prediction_model(model_path=None)   # None → model.pth
```

1. `_apply_meta()`는 `TrainData.get_meta_path()`의 JSON을 읽고 `characters`, `image_width`, `image_height`, `label_length`가 모두 truthy인지 확인한다. 파일 부재·읽기/JSON 파싱 실패·필수 값 누락이면 적용을 건너뛴다.
2. `TrainData.apply_meta(meta)`에 감지 캐시 갱신을 위임하고 문자 인덱스·클래스 수를 다시 만든다. `characters`는 문자열이며, 크기와 길이는 정수로 변환한다. 타입·변환 오류는 전파된다.
3. `build_model()` → `torch.load(model_path, map_location=self.device, weights_only=False)` → `load_state_dict` → `eval()` 순서로 로드한다.
4. `torch.load` 또는 `load_state_dict` 실패 시 `self.model`을 이전 값으로 되돌린다. 앞에서 적용한 데이터 캐시와 문자 매핑까지 되돌리지는 않는다.

메타데이터의 `threshold`는 감지 캐시에 저장된다. 실제 전처리가 읽는 `TrainData.threshold`,
`preprocess`, `crop`, 크롭 전 기준 크기는 이 과정에서 바뀌지 않는다. 학습 때의 전처리 설정도 함께 맞춰야 한다.
`model_path`를 별도로 넘겨도 sidecar 조회 경로는 `TrainData.get_meta_path()`를 따른다.

### 3.7 추론 파이프라인

**단일 이미지**:
```python
pred_text, confidence = model.predict(image_path, unk_token="[UNK]", beam_width=10)
```

**배치 추론**:
```python
results = model.predict_batch(image_paths, unk_token="[UNK]", beam_width=10, batch_size=32)
# [(pred_text, confidence), ...]
```

**추론 특징**:
- `get_eval_transform` (증강 없음) → `torch.inference_mode()`
- CUDA + `use_amp` 면 `autocast(float16)`; 로짓은 **float32 로 올린 뒤** `log_softmax` (fp16 양자화 잡음이 신뢰도에 섞이고 ONNX 경로와 값이 갈리는 것을 막기 위함)
- `(T, N, C) → (N, T, C)` 로 바꿔 샘플별로 `ctc_beam_decode_fixed_length()` (§4)
- `expected_length = label_length` (고정 길이 디코딩 전용; `None` 이면 `ValueError`)
- `PyTorchModel.predict()`·`predict_batch()`는 모델을 자동 로드하지 않는다. 이미지 읽기·변환 실패는 호출자에게 전달된다.
- `predict_batch()`는 순서를 유지해 최대 `batch_size`장씩 `torch.stack`하고, 결과를 입력 순서대로 반환한다. 개별 파일 실패를 건너뛰는 처리는 이 메서드에 없다.

> `predict()` 시그니처의 `loss_type`, `use_amp` 인자는 받기만 하고 내부에서 쓰지 않는다 (인스턴스 속성 `self.use_amp` 를 본다).

---

## 4. CTC Beam Search 디코딩 (`core.py` `ctc_beam_decode_fixed_length`)

고정 길이 라벨을 위한 CTC **Prefix** Beam Search. 프레임마다 prefix 를 확장하고, 기대 길이로 도달 불가능한 prefix 를 가지치기한다.

```python
def ctc_beam_decode_fixed_length(
    log_probs: np.ndarray,       # (T, num_classes), index 0 = blank
    mapping_inv: Dict[int, str], # index -> character
    expected_length: int,        # 기대 라벨 길이 (하드 제약)
    beam_width: int = 10,        # 유지할 prefix 수 (문자열과 분자 점수에 영향)
    unk_token: str = "[UNK]",
    top_k: int = 0,              # 프레임당 후보 문자 수 (0 → beam_width*2), blank 는 항상 포함
) -> Tuple[str, float]:
    ...
```

**디코딩 흐름**:
1. 프레임별 상위 `top_k` 후보 + blank만 확장한다. `top_k <= 0`이면 `min(num_classes, beam_width*2)`를 사용한다.
2. 각 prefix에 대해 blank 종료 경로(`p_b`)와 문자 종료 경로(`p_nb`)를 log-sum-exp로 **합산**한다. 점수는 탐색에 남은 정렬 경로들의 확률 합이며 가지치기로 제외된 경로는 포함하지 않는다.
3. 같은 문자 반복: blank 없이 이어지면 같은 prefix, blank 를 거쳤으면 새 문자
4. 길이 초과·남은 프레임으로 도달 불가능한 확장을 제외하고 상위 `beam_width`를 유지한다. 남은 길이 필터가 후보를 모두 없애면 필터 전 후보로 폴백한다.
5. 최종: 길이가 정확히 `expected_length` 인 후보 중 최고 점수 (없으면 전체 중 최고)

**신뢰도**: 길이가 L인 전체 확률 질량으로 정규화한 beam 점수
```python
confidence = exp(best_score - length_logprob(log_probs, expected_length))
# best_score: 가지치기 후 선택된 prefix의 누적 log 확률
```

`length_logprob()`는 길이가 정확히 L인 **모든** 문자열의 확률 합을 구하는 DP다. 시간 복잡도는
O(T·L·C), 상태 배열은 O(L·C)이며, 프레임마다 확률을 재스케일링해 언더플로를 줄인다.
분모는 beam과 무관하지만 **분자는 beam/top-k 가지치기의 영향을 받는다**. 따라서 출력은
길이 조건부 사후확률의 근사값이고 `beam_width`에 완전히 불변인 값이나 보정된 정확도로 해석하면 안 된다.

정확한 길이의 후보가 없거나 길이 확률이 `-inf`이면 남은 beam 점수 합으로 정규화한다.
결과는 `[0, 1]`로 제한한다. 디코더는 `T == 0` 또는 `expected_length <= 0`이면 `('', 0.0)`을 반환한다.
`length_logprob()`는 `L <= 0`, `T == 0`, `T < L`이면 `-inf`를 반환한다.

---

## 5. 데이터 흐름

### 5.1 `TrainData` 와 자동 감지 (`web.core.dataclass`)

`TrainData`(pydantic) 는 생성 시 `images/train/*.png` 를 스캔해 학습 정보를 감지·캐시한다 (`_detect_and_cache`).

| 감지 항목 | 방법 |
|-----------|------|
| 이미지 크기 | 정렬된 마지막 파일을 열어 `im.size` (실패 시 PNG 헤더 16 바이트 오프셋에서 직접 읽음) |
| 라벨 길이 | 파일명(확장자 제외) 길이의 최댓값 |
| 문자셋 | 모든 파일명 문자의 정렬된 집합 |
| 모델 입력 크기 | 마지막 파일에 `image_pre_process()` 를 한 번 돌려 **전처리 후** 크기로 확정 (크롭 전처리가 있으면 파일 크기와 다름) |

`detected_image_width/height`, `detected_label_length`, `detected_characters` 프로퍼티는 "감지값 우선, 없으면 생성자 값" 이며 `PyTorchModel` 은 항상 이쪽을 쓴다.
생성자 인자 `image_width/height`는 감지값이 없을 때의 입력 크기이며,
`iptime` 크롭 전처리에서는 **크롭 좌표계의 기준 크기**로도 쓰인다.

경로 규약: `captcha_data/<captcha_id>/<rev>/images/{train,pred}/`, `.../model/`. 파일명 = 정답 라벨. 리비전은 **1부터 시작**한다 (`TrainData.rev` 기본값 1).
`get_data_files()` 는 파일명 길이가 `detected_label_length` 와 같은 PNG 만 돌려준다.

### 5.2 전처리 (`TrainData.image_pre_process`)

`TrainData.preprocess` 값에 따라 분기한다. 모두 결과는 그레이스케일(`L`) PIL 이미지다.

| `preprocess` | 흐름 | 사용 캡차 |
|--------------|------|-----------|
| `default` | RGBA→흰 배경 합성 → `L` → 임계값(`0<threshold<255` 이면 `p>threshold → 255`) → 테두리 2px 제거 → 밝기 >128 → 255 → 감지 크기로 리사이즈 | gov24(threshold=60), wetax |
| `supreme_court` | RGBA면 흰 배경 합성. 그 외 모드에서 폭·높이가 모두 감지 크기보다 크면 감지 W/H 기준 `(3,1,W-1,H-7)` 크롭 후 캔버스 `(1,1)`에 붙임 → `L` → 테두리 제거 → 배경 흰색 → 감지 크기로 리사이즈 | supreme_court |
| `iptime` | RGBA면 흰 배경 합성 → `L` → 필요 시 기준 크기(200x70)로 리사이즈 → `crop=[27,10,195,70]` (168x60). 임계값·테두리 제거·크롭 후 리사이즈 없음 | iptime |

크롭 박스는 PIL `(left, top, right, bottom)` 이며 `model.meta.json` 에 `crop` / `crop_source` 로 실려 다른 언어 클라이언트가 재현한다.

### 5.3 학습 Transform (`core.py` `get_train_transform`)

```python
T.Compose([
    T.Lambda(train_data.image_pre_process),
    T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=[0, 3], fill=255),
    T.RandomPerspective(distortion_scale=0.1, p=0.3, fill=255),
    T.RandomApply([T.RandomGrayscale(p=0.1)], p=0.2),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.3),
    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2)], p=0.3),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.RandomErasing(p=0.15, scale=(0.01, 0.05), ratio=(0.3, 3.0), value=1.0),
])
```

평가/추론용 `get_eval_transform` 은 `Lambda(image_pre_process) → ToImage → ToDtype(float32, scale=True)` 만 적용한다.
(`torchvision.transforms.v2` 사용.)
`RandomGrayscale`은 바깥 확률 0.2와 안쪽 확률 0.1이 함께 적용되지만, 이 시점의 이미지는 이미
전처리에서 그레이스케일로 변환된 상태다. `RandomErasing`의 `value=1.0`은 정규화 후 흰색이다.

### 5.4 데이터셋

```python
dataset = CaptchaDataset(df, path, mapping, transform)
# __getitem__: PIL 로드 → transform → (image_tensor, label_tensor)
# 라벨 = 파일명(확장자 제외)을 char_to_idx 로 매핑한 정수 시퀀스
```

라벨 정수화는 loader를 만드는 쪽에서 수행한다. `CaptchaDataset`은 DataFrame의 `label`을
`torch.long`으로 바꾸며, 저장해 둔 `mapping`으로 다시 변환하지 않는다. transform이 없으면 이미지 반환형은
텐서가 아닌 그레이스케일 PIL 이미지다. `create_prediction_dataset()`은 `images/pred`를 사용하고
매핑에 없는 문자를 0으로 치환한다 (학습 loader에서는 그 문자를 버린다).

### 5.5 train/pred 재분배

- `engine.redistribute_train_pred(image_dir, train_ratio=0.9, extension='png', seed=42)`: `pred` 파일을 `train`으로 옮긴 뒤 셔플해 `train_ratio`만큼 남기고 나머지를 `pred`로 이동하는 함수다. 웹 UI의 `shuffle` 옵션이 호출한다. 현재 중복 파일명 분기는 기존 `train` 파일만 삭제하고 `pred` 원본은 이동하지 않으므로, 충돌이 있으면 전체 파일을 합친 뒤 분할한 결과와 달라진다.
- `TrainData.shuffle_train_data(train_size=0.9)`: 같은 목적의 구버전 (`_temp` 디렉터리 경유, seed 없음).

둘 다 디스크의 파일을 **실제로 옮기는** 파괴적 동작이다.

---

## 6. engine 진입점 (`web.core.engine`)

### 6.1 캡차 레지스트리

`get_captcha_type_list(train_data_base_dir="./captcha_data")` 가 `CaptchaType` 4종을 코드로 등록한다. 등록은 이 함수가 유일한 소스다.

| `captcha_id` | preprocess | 기준 크기 | crop | 기타 |
|--------------|-----------|-----------|------|------|
| `supreme_court` | `supreme_court` | 120x40 | — | |
| `gov24` | `default` | (기본 200x50) | — | `threshold=60` |
| `wetax` | `default` | 높이 60 | — | |
| `iptime` | `iptime` | 200x70 | `[27,10,195,70]` → 168x60 | `label_length=5`, `characters=a-z` (유일한 비숫자) |

모든 캡차의 rev 는 기본값 1 이다 (리비전은 1부터 시작). `with_rev(captcha_type, rev)` 는 rev 만 바꾼 사본을 만든다 — `TrainData` 가 다시 생성되면서 해당 rev 기준으로 재감지된다.

```python
model = engine.get_captcha_model(train_data_base_dir="./captcha_data", captcha_id="supreme_court",
                                 verbose=1, device=None, rev=None)
# device: None → auto, 문자열('cpu'/'cuda') → torch.device 로 변환
# rev: None → 레지스트리 기본 rev
```

> `engine` 자체에는 모델 캐시가 없다. 호출마다 새 `PyTorchModel` 을 만든다. 메모리 캐시(`_MODEL_CACHE`)는 `apps/web/services/captcha.py` 가 관리한다.

### 6.2 학습 (`engine.train_model`)

```python
engine.train_model(
    model,
    epochs=80, batch_size=32,
    earlystopping=True, early_stopping_patience=15,   # earlystopping=False 면 patience=0
    learning_rate=0.001, num_workers=0, warmup_epochs=0,
    loss_type='focal', use_amp=True,
    on_event=None,
)
```

내부: `model.use_amp = use_amp` → `build_model()` → `split_dataset(batch_size, train_size=0.8, shuffle=True, num_workers, pin_memory=False)` → `PyTorchModel.train_model(save_best=True, ...)`.
기본 patience 15의 선택 근거는 초기 CTC 정체 구간에 대한 코드 주석에 기록돼 있다.
기본 `warmup_epochs=0`이므로 이 진입점을 기본 인자로 호출하면 warmup 없이 cosine 스케줄을 쓴다.
항상 `build_model()`부터 호출하므로 기존 가중치를 이어 학습하는 재개 API가 아니다.
`engine.train_model()`은 내부 `train_hist`를 반환하지 않는다 (반환값 `None`).

### 6.3 추론

```python
text, confidence = engine.predict(model, image_path, verbose=1, unk_token="[UNK]", loss_type='focal')
# model.model 이 None 이면 load_prediction_model() 을 먼저 호출한다. use_amp=True 로 강제.
```

### 6.4 일괄 추론 (`iter_batch_predict`)

`images/pred/`(또는 `pred_image_dir`) 의 이미지를 한 장씩 예측하며 이벤트 dict 를 yield 하는 제너레이터. 웹의 SSE 와 CLI 출력(`batch_predict_model`)이 이 하나를 공유한다.

| `type` | 시점 | 키 |
|--------|------|-----|
| `start` | 1회 | `captcha_id, rev, device, loss_type, total, pred_image_dir` |
| `item` | 매 장 | `index, image, expected(파일명), pred, confidence, match` (+ 실패 시 `error`, `pred=''`, `confidence=0.0`) |
| `summary` | 1회 | `loss_type, total, match, mismatch, accuracy(%), elapsed_sec` |

`match` 는 `pred == expected and len(pred) == detected_label_length` 다. 한 장이 실패해도 전체를 멈추지 않고 불일치로 센다.
단, 모델 로드는 `start` 이벤트 이전에 수행하므로 체크포인트 로드 실패는 전체 호출의 예외가 된다.
이 경로는 이미지를 한 장씩 처리하며 `PyTorchModel.predict_batch()`를 사용하지 않는다.

---

## 7. 파일 저장 형식 (`finalize_artifacts`)

학습 종료 시 확정된 `model.pth` **하나를 디스크에서 다시 읽어** 나머지 산출물을 만든다 (메모리 모델에서 export 하면 체크포인트와 에폭이 어긋나는 사고가 있었음).
기본 경로는 `captcha_data/<captcha_id>/<rev>/model/`이다. 사용자 지정 `model_path`를 넘겨도
파생 파일의 경로는 `TrainData`의 `get_export_path()`·`get_onnx_path()`·`get_ort_path()`·`get_meta_path()`를 따른다.

| 파일 | 형식 | 생성 방법 | 용도 |
|------|------|-----------|------|
| `model.pth` | `state_dict` | `torch.save` (`.tmp` → `os.replace`) | 파이썬 추론·재학습 기준 체크포인트 |
| `model.pt2` | `torch.export` 아카이브 | `torch.export.export(wrapper, (dummy,))` → `torch.export.save`, 배치 1 고정 | PyTorch 런타임에서 원래 CRNN 클래스 정의 없이 그래프 로드 |
| `model.onnx` | ONNX | `torch.onnx.export(..., opset_version=17, dynamo=False)`, 입력 `input`/출력 `output`, 배치 1 고정 | Rust CLI / Spring Boot / WinConsoleApp |
| `model.ort` | ORT flatbuffer | `onnxruntime` 세션 옵션 `ORT_ENABLE_EXTENDED` + `save_model_format=ORT` | 로드 빠름, minimal build 런타임용 (`ENABLE_ALL` 은 CPU 명령셋 종속이라 쓰지 않음) |
| `model.meta.json` | JSON | `CaptchaType.build_meta()` | 문자셋·크기·전처리 (§7.1) |

export 는 `_InferenceWrapper` (추론 전용 `forward(x)`) 로 감싸 학습용 `y/criterion` 인자를 감춘다.
입력은 `(1, 1, H, W)`, 출력은 **log_softmax 이전 로짓 `(T, 1, C+1)`**이다.
`finalize_artifacts()`의 순서는 체크포인트 재로드 → `.pt2` → `.onnx` → `.ort` → meta 저장 → 검증이다.

`onnxruntime` 은 export/검증 시점에만 늦게 import 한다 (`_require_onnxruntime`).

### 7.1 `model.meta.json`

```json
{
  "captcha_id": "iptime", "name": "ipTIME", "rev": 1,
  "image_width": 168, "image_height": 60,
  "label_length": 5, "characters": "abcdfhijklmnopqrstuvwxy",
  "threshold": 255, "preprocess": "iptime",
  "crop": [27, 10, 195, 70], "crop_source": [200, 70],
  "blank_index": 0
}
```

위 문자셋은 예시다. 실제 값은 학습 파일명에서 감지한다. `image_width/height`는 크롭 후 모델 입력 크기,
`crop_source`는 크롭 전 기준 크기이며, 크롭이 없으면 `crop`과 `crop_source`는 모두 `null`이다.
`load_prediction_model()`은 이 파일의 입력 크기·문자셋을 적용하지만 전처리 설정 전체를 복원하지는 않는다 (§3.6).

### 7.2 저장·검증 범위

- `save_model()`, `export_pt2()`, `export_ort()`, `save_meta()`는 각각 `.writing`, `.writing.pt2`, `.writing.ort`, `.writing` 스테이징 파일을 만든 뒤 `os.replace`로 교체한다. ONNX는 목적 경로에 직접 export한다.
- 전체 파일 묶음은 트랜잭션이 아니다. 뒤 단계에서 실패하면 앞서 교체한 파일을 자동으로 되돌리지 않는다.
- 기본 `verify=True`일 때 `.onnx`·`.ort` 각각을 `verify_onnx_export(num_samples=8, logit_tol=0.5)`로 확인한다. `images/train/*.png`를 정렬한 앞 최대 8장이 대상이며, PyTorch는 현재 모델 디바이스, ONNX Runtime은 `CPUExecutionProvider`에서 실행한다. 이 비교에는 autocast를 쓰지 않는다.
- 두 출력에 log-softmax와 동일한 고정 길이 디코더를 적용한다. 예측 문자열 불일치가 하나라도 있거나 로짓 최대 절댓값 오차가 0.5보다 크면 `RuntimeError`다.
- 학습 PNG가 없으면 세션 생성만 수행하고 이미지 비교 루프는 0회 실행한다. `.pt2` 재로드·출력 동등성, confidence 일치, 정답 대비 모델 정확도는 이 검증에 포함되지 않는다. `verify=False`면 ONNX/ORT 비교도 건너뛴다.
- `export_onnx(fixed_batch=False)` 분기는 입력·출력의 0번 축을 모두 `batch_size`로 지정한다. 실제 출력 `(T,N,C)`의 배치 축은 1번이므로, 이 옵션의 동적 배치 동작은 현재 축 선언만으로 보장되지 않는다. 기본 export 경로는 배치 1 고정이다.

---

## 8. 사용 예시

### 8.1 라이브러리

```python
from web.core import engine

# 1. 모델 생성 (레지스트리 기본 rev, device auto)
model = engine.get_captcha_model(train_data_base_dir="./captcha_data", captcha_id="supreme_court")

# 2. 학습 (Focal CTC, Warmup→Cosine, 80% train / 20% val)
engine.train_model(
    model=model,
    epochs=80,
    batch_size=32,
    early_stopping_patience=15,
    learning_rate=0.001,
    warmup_epochs=5,
    loss_type='focal',
    use_amp=True,
    on_event=lambda ev: print(ev['type'], ev),   # 선택: 진행 이벤트
)

# 3. 저장된 체크포인트로 단건 추론 (기존 모델 유지로 학습이 종료된 경우도 반영)
model.load_prediction_model()
pred_text, confidence = engine.predict(
    model=model,
    image_path="captcha_data/supreme_court/1/images/pred/sample.png",
)
print(f"예측: {pred_text} (신뢰도: {confidence:.4f})")

# 4. 일괄 추론 (images/pred)
for ev in engine.iter_batch_predict(model=model):
    if ev['type'] == 'summary':
        print(f"accuracy={ev['accuracy']:.2f}%")
```

`import aso_ai` 최상위는 torch 를 로드하지 않는다. `web.core.engine` / `aso_ai.core` 를 import 하는 시점에 `core.py` 의 cuDNN/TF32 설정과 CUDA 프로브가 실행된다.

### 8.2 CLI (`aso_ai.cli`, `apps/web/pyproject.toml`의 `[project.scripts] aso-ai`)

```bash
uv run --project apps/web aso-ai -c supreme_court -i path/to/image.png
uv run --project apps/web aso-ai -c supreme_court -i path/to/image.png -v
uv run --project apps/web python -m aso_ai -c supreme_court -i path/to/image.png
```

CLI 는 **현재 작업 디렉토리** 기준 `./captcha_data` 를 쓴다. 종료 코드: 이미지 없음 2, 모델 생성 실패 3.
기본 출력은 예측 문자열(개행 없음), `-v`는 예측·confidence·소요 시간 JSON이다.
웹 서비스는 설정의 `CAPTCHA_DATA_DIR`을 엔진에 명시적으로 전달한다. 직접 엔진을 호출할 때의 기본
`./captcha_data`와 실행 위치가 다를 수 있으므로 경로를 구분한다.

### 8.3 학습/배치 평가

웹 `/train`에서 학습하고 `/predict`에서 일괄 평가한다. Python에서 직접 실행할 때는
`web.core.engine.get_captcha_model()`로 모델을 만들고 `engine.train_model()` 또는
`engine.batch_predict_model()`을 호출한다. 하드코딩된 수동 실행 스크립트는 제거했다.
