# CRNN + CTC Loss 아키텍처

> CAPTCHA 인식 기반 모델: Convolutional Recurrent Neural Network + Connectionist Temporal Classification
>
> 대상 코드: `packages/python_3.12/hyperCaptcha/src/hypercaptcha/` (`core.py`, `engine.py`, `dataclass.py`, `cli.py`).
> 이 문서는 해당 소스의 **현재 동작**을 기준으로 작성됐다. 웹앱(`apps/web`)이 이 패키지를 어떻게 감싸는지는
> [web-architecture.md](./web-architecture.md), 데이터 구조는 [web-domain-model.md](./web-domain-model.md)를 참고한다.

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
| 가중치 초기화 | Conv: `kaiming_normal_(fan_out)`, LSTM: `orthogonal_`, Linear: `xavier_uniform_`, bias: 0 (`_init_weights`) |

> 코드 docstring 에 "Residual Connection" 이 언급돼 있으나 실제 `nn.Sequential` 에는 잔차 연결이 없다.

차원 변화 예시 (대법원 캡차 120x40 기준):

```
Input:       (N, 1, 40, 120) → H/8=5, W/4=30 → (N, 256, 5, 30)
Feature dim: C x H = 256 x 5 = 1280 (Feature Projection 입력 차원)
Time steps:  T = W/4 = 30 (CTC 프레임 수)
```

**CTC 요구사항**: `time_steps >= label_length`. 위반 시 생성자에서 `ValueError` (`core.py:244`). 입력 폭이 너무 작으면 여기서 죽는다.

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
현재 코드는 **Focal CTC 만 지원**한다 — `train_model()` 은 `loss_type != 'focal'` 이면 `ValueError` 를 올린다 (`core.py:677`).
주석에 따르면 blank 지배·초기 정체 구간에서 focal 이 더 낫고, 실측(iptime 등 99%)도 그 편이라 표준 CTC 는 제거했다.

### 2.1 Focal CTC Loss

```python
class FocalCTCLoss(nn.Module):
    def __init__(self, blank=0, gamma=2.0, alpha=0.25, reduction='mean', zero_infinity=True):
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
| `alpha` | 0.25 (기본값) | 클래스 불균형 보정 |
| `zero_infinity` | True | 무한대 loss 를 0 으로 (도달 불가 정렬 방어) |
| 내부 CTC reduction | `'none'` | 샘플별 loss 를 받아 focal 가중치를 곱한 뒤 평균 |

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
    captcha_type=captcha_type,   # hypercaptcha.dataclass.CaptchaType
    verbose=1,
    device=None,                 # None=auto (CUDA 가능하면 cuda, 아니면 cpu)
    use_compile=False,           # torch.compile 사용 여부
    use_amp=True,                # Mixed Precision (CUDA 에서만 실제 적용)
    loss_type='focal',           # 'focal' 만 유효
    model_dir=None,              # None 이면 TrainData.get_model_base_dir()
)
```

생성 시 문자 매핑을 만든다. 문자셋·라벨 길이·이미지 크기는 모두 `TrainData` 의 **감지값(`detected_*`)을 우선**한다 (§5.1).

**PyTorch 전역 설정** (`core.py` import 시점):
```python
torch.backends.cudnn.benchmark = True         # 고정 입력 크기 자동 튜닝
torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+ TF32
torch.backends.cudnn.allow_tf32 = True
# CUDA 가 있으면 conv2d 를 한 번 찔러 보고 cuDNN 버전 불일치로 실패하면 cudnn 을 끈다.
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
    num_workers=0, pin_memory=True,          # pin_memory 는 CUDA 일 때만 실제 True
    persistent_workers=None, prefetch_factor=2,   # num_workers > 0 일 때만 적용
)
```

- 대상: `images/train/*.png` 중 파일명 길이가 `detected_label_length` 와 같은 파일만 (`TrainData.get_data_files`)
- 라벨 = 파일명(확장자 제외) → `char_to_idx` 로 매핑한 정수 시퀀스 (매핑에 없는 문자는 버림)
- `sklearn.train_test_split(test_size=1-train_size, shuffle=shuffle)` 로 나눔 (고정 seed 없음)
- train 쪽은 증강 transform, val 쪽은 eval transform (§5.2)
- `DataLoader`: train `shuffle=True`, val `shuffle=False`

`engine.train_model()` 은 `train_size=0.8, shuffle=True, num_workers=0, pin_memory=False` 로 호출한다.

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
| 옵티마이저 | `AdamW(lr, weight_decay)`, CUDA 면 `fused=True` |
| 스케줄러 | **Linear Warmup → Cosine Annealing** (`LambdaLR`, 에폭 단위): `step < warmup` 이면 `(step+1)/warmup`, 이후 `0.5·(1+cos(π·progress))` |
| Mixed Precision | CUDA 이고 `use_amp=True` 일 때만 `GradScaler` + `autocast(float16)`. CPU 는 AMP 없이 fp32 |
| Gradient Clipping | `clip_grad_norm_(max_norm=grad_clip)` (`grad_clip > 0` 일 때, AMP 는 unscale 후) |
| 검증 | 매 에폭 `val_loader` 로 loss 계산 (AMP 동일 적용). `val_loss` 가 개선되면 `model.pth.tmp` 로 임시 저장 |
| val_loader 없을 때 | 10 에폭마다 `.tmp` 저장 |
| Early Stopping | `early_stopping_patience > 0` 이고 무개선이 patience 만큼 이어지면 `stop_reason='early_stopping'` |
| 중단 | `on_event` 가 `False` 를 돌려주면 `cancelled` (best 확정), `'discard'` 면 `cancelled_discarded` (`.tmp` 삭제, 기존 아티팩트 유지) |
| 덮어쓰기 가드 | 종료 후 디스크의 기존 `model.pth` 를 같은 `val_loader` 로 재평가(`_evaluate_checkpoint`). 기존이 더 좋으면(`incumbent <= best`) `.tmp` 를 버리고 `skipped` 이벤트로 끝냄. 비교 불가(구조 변경 등)면 그냥 덮어씀 |
| 확정 | `.tmp` 가 있으면 `os.replace` 로 `model.pth` 승격, 없으면 현재 가중치 저장 → `finalize_artifacts()` (§7) |

저장은 항상 `*.writing` 스테이징 파일에 쓴 뒤 `os.replace` 로 교체한다 (중간에 죽어도 직전 파일이 남음).

### 3.5 진행 이벤트 (`on_event`)

`train_model(on_event=콜백)` 을 주면 dict 하나를 인자로 콜백이 호출된다. 웹앱(`apps/web/services/train.py`)이 이 콜백으로 SSE 를 만든다.
콜백 반환값이 `False` / `'discard'` 이면 다음 에폭으로 넘어가지 않는다 (§3.4).

| `type` | 시점 | 주요 payload 키 |
|--------|------|-----------------|
| `start` | 학습 시작 시 1회 | `captcha_id, rev, device, epochs, loss_type, batch_size, train_batches, val_batches, image_width, image_height, label_length, characters, lr, warmup_epochs, early_stopping_patience, use_amp, model_path` |
| `epoch` | 매 에폭 종료 후 | `epoch, epochs, train_loss, val_loss, lr, best_val_loss, best_epoch, improved, patience_counter, elapsed_sec` |
| `skipped` | 덮어쓰기 가드가 기존 모델을 유지할 때 (종료) | `reason='incumbent_better', incumbent_val_loss, best_val_loss, epochs_run, epochs, elapsed_sec` |
| `done` | 정상/조기종료/취소 종료 시 | `epochs_run, epochs, stop_reason` (`completed`\|`early_stopping`\|`cancelled`\|`cancelled_discarded`), `best_val_loss, best_epoch, elapsed_sec, artifacts` (경로 dict, discard 시 `{}`) |

> 웹앱 쪽에서 추가로 `shuffle`(재분배 결과), `error` 이벤트를 같은 스트림에 섞는다. 이는 `hypercaptcha` 가 아니라 `apps/web/services/train.py` 가 만든다.

### 3.6 모델 로드 (`load_prediction_model`)

```python
model.load_prediction_model(model_path=None)   # None → model.pth
```

1. `_apply_meta()`: `model.meta.json` 이 있으면 그 안의 `characters / image_width / image_height / label_length / threshold` 로 `TrainData._train_info` 와 문자 매핑을 **덮어쓴다**. 학습 후 `images/train` 이 바뀌어(라벨 수정, pred 로 이동 등) 감지값이 달라지면 출력층/입력 크기가 어긋나 `load_state_dict` 가 실패하기 때문에, 배포 모델의 진실은 사이드카 meta 로 본다. meta 가 없거나 값이 빠지면 감지값 폴백.
2. `build_model()` → `torch.load(model_path)` → `load_state_dict` → `eval()`.
3. 로드에 실패하면 `self.model` 을 이전 값으로 되돌리고 예외를 올린다 (깨진 체크포인트/LFS 포인터로 무학습 모델이 남는 사고 방지).

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

> `predict()` 시그니처의 `loss_type`, `use_amp` 인자는 받기만 하고 내부에서 쓰지 않는다 (인스턴스 속성 `self.use_amp` 를 본다).

---

## 4. CTC Beam Search 디코딩 (`core.py` `ctc_beam_decode_fixed_length`)

고정 길이 라벨을 위한 CTC **Prefix** Beam Search. 프레임마다 prefix 를 확장하고, 기대 길이로 도달 불가능한 prefix 를 가지치기한다.

```python
def ctc_beam_decode_fixed_length(
    log_probs: np.ndarray,       # (T, num_classes), index 0 = blank
    mapping_inv: Dict[int, str], # index -> character
    expected_length: int,        # 기대 라벨 길이 (하드 제약)
    beam_width: int = 10,        # 유지할 prefix 수 (예측 문자열에만 영향)
    unk_token: str = "[UNK]",
    top_k: int = 0,              # 프레임당 후보 문자 수 (0 → beam_width*2), blank 는 항상 포함
) -> Tuple[str, float]:
```

**디코딩 흐름**:
1. 프레임별 상위 `top_k`(기본 `beam_width*2`) 후보 + blank 만 확장
2. 각 prefix 에 대해 blank 종료 경로(`p_b`)와 문자 종료 경로(`p_nb`)를 log-sum-exp 로 **합산** — beam 점수 = `P(문자열 | 이미지)` (단일 정렬 확률이 아님)
3. 같은 문자 반복: blank 없이 이어지면 같은 prefix, blank 를 거쳤으면 새 문자
4. `len(prefix) > expected_length` 이거나 남은 프레임으로 채울 수 없는 prefix 제거 → 상위 `beam_width` 유지
5. 최종: 길이가 정확히 `expected_length` 인 후보 중 최고 점수 (없으면 전체 중 최고)

**신뢰도**: 길이 제약으로 조건부화한 사후확률
```python
confidence = exp(best_score - length_logprob(log_probs, expected_length))
#           = P(예측 문자열 | 이미지, 길이 = expected_length)
```

분모 `length_logprob()` 는 길이가 정확히 L 인 **모든** 문자열의 확률 합을 O(T·L·C) DP 로 정확히 구한다.
살아남은 beam 집합으로 정규화하면 1위와 2위의 비율만 재게 되어 `beam_width` 에 따라 값이 흔들리고
학습 분포 밖 입력에도 0.5 언저리를 돌려주는 문제가 있어 이렇게 한다.
길이 L 이 도달 불가(`T < L` 등)하면 남은 후보들의 상대 점수로 대체하고, 결과는 `[0, 1]` 로 클램프한다.

---

## 5. 데이터 흐름

### 5.1 `TrainData` 와 자동 감지 (`dataclass.py:63-166`)

`TrainData`(pydantic) 는 생성 시 `images/train/*.png` 를 스캔해 학습 정보를 감지·캐시한다 (`_detect_and_cache`).

| 감지 항목 | 방법 |
|-----------|------|
| 이미지 크기 | 정렬된 마지막 파일을 열어 `im.size` (실패 시 PNG 헤더 16 바이트 오프셋에서 직접 읽음) |
| 라벨 길이 | 파일명(확장자 제외) 길이의 최댓값 |
| 문자셋 | 모든 파일명 문자의 정렬된 집합 |
| 모델 입력 크기 | 마지막 파일에 `image_pre_process()` 를 한 번 돌려 **전처리 후** 크기로 확정 (크롭 전처리가 있으면 파일 크기와 다름) |

`detected_image_width/height`, `detected_label_length`, `detected_characters` 프로퍼티는 "감지값 우선, 없으면 생성자 값" 이며 `PyTorchModel` 은 항상 이쪽을 쓴다.
생성자 인자 `image_width/height` 는 **크롭 좌표계의 기준(크롭 전) 크기** 로만 쓰인다.

경로 규약: `captcha_data/<captcha_id>/<rev>/images/{train,pred}/`, `.../model/`. 파일명 = 정답 라벨. 리비전은 **1부터 시작**한다 (`TrainData.rev` 기본값 1).
`get_data_files()` 는 파일명 길이가 `detected_label_length` 와 같은 PNG 만 돌려준다.

### 5.2 전처리 (`dataclass.py:285-349`)

`TrainData.preprocess` 값에 따라 분기한다. 모두 결과는 그레이스케일(`L`) PIL 이미지다.

| `preprocess` | 흐름 | 사용 캡차 |
|--------------|------|-----------|
| `default` | RGBA→흰 배경 합성 → `L` → 임계값(`0<threshold<255` 이면 `p>threshold → 255`) → 테두리 2px 제거 → 밝기 >128 → 255 → 감지 크기로 리사이즈 | gov24(threshold=60), wetax |
| `supreme_court` | RGBA 면 흰 배경 합성 / 감지 크기보다 크면 `(3,1,W-1,H-7)` 크롭 후 `(1,1)` 에 붙임 → `L` → 테두리 제거 → 배경 흰색 → 리사이즈 (120x40) | supreme_court |
| `iptime` | RGBA→`L` → 기준 크기(200x70)로 리사이즈 → `crop=[27,10,195,70]` (168x60). 임계값·테두리·리사이즈 없음 | iptime |

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

### 5.4 데이터셋

```python
dataset = CaptchaDataset(df, path, mapping, transform)
# __getitem__: PIL 로드 → transform → (image_tensor, label_tensor)
# 라벨 = 파일명(확장자 제외)을 char_to_idx 로 매핑한 정수 시퀀스
```

### 5.5 train/pred 재분배

- `engine.redistribute_train_pred(image_dir, train_ratio=0.9, extension='png', seed=42)`: `pred` 를 전부 `train` 으로 합친 뒤 셔플해 `train_ratio` 만큼만 남기고 나머지를 `pred` 로 이동. 웹 UI 의 `shuffle` 옵션이 호출한다.
- `TrainData.shuffle_train_data(train_size=0.9)`: 같은 목적의 구버전 (`_temp` 디렉터리 경유, seed 없음).

둘 다 디스크의 파일을 **실제로 옮기는** 파괴적 동작이다.

---

## 6. engine 진입점 (`engine.py`)

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
patience 15 는 CTC 초기 정체 구간(무개선 7에폭 연속까지 관측)에서 죽지 않도록 잡은 값이다.

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

---

## 7. 파일 저장 형식 (`finalize_artifacts`)

학습 종료 시 확정된 `model.pth` **하나를 디스크에서 다시 읽어** 나머지 산출물을 만든다 (메모리 모델에서 export 하면 체크포인트와 에폭이 어긋나는 사고가 있었음).
모두 `captcha_data/<captcha_id>/<rev>/model/` 아래.

| 파일 | 형식 | 생성 방법 | 용도 |
|------|------|-----------|------|
| `model.pth` | `state_dict` | `torch.save` (`.tmp` → `os.replace`) | 파이썬 추론·재학습 기준 체크포인트 |
| `model.pt2` | `torch.export` 아카이브 | `torch.export.export(wrapper, (dummy,))` → `torch.export.save`, 배치 1 고정 | 모델 정의 코드 없이 로드 |
| `model.onnx` | ONNX | `torch.onnx.export(..., opset_version=17, dynamo=False)`, 입력 `input`/출력 `output`, 배치 1 고정 | Rust CLI / Spring Boot / WinConsoleApp |
| `model.ort` | ORT flatbuffer | `onnxruntime` 세션 옵션 `ORT_ENABLE_EXTENDED` + `save_model_format=ORT` | 로드 빠름, minimal build 런타임용 (`ENABLE_ALL` 은 CPU 명령셋 종속이라 쓰지 않음) |
| `model.meta.json` | JSON | `CaptchaType.build_meta()` | 문자셋·크기·전처리 (§7.1) |

export 는 `_InferenceWrapper` (추론 전용 `forward(x)`) 로 감싸 학습용 `y/criterion` 인자를 감춘다.
`.onnx` / `.ort` 는 `verify_onnx_export()` 로 `images/train` 앞 8장을 PyTorch 와 비교해 **디코딩된 예측 문자열이 모두 일치**하고 로짓 최대 오차가 0.5 이하일 때만 통과시킨다 (불일치 시 `RuntimeError`).

`onnxruntime` 은 export/검증 시점에만 늦게 import 한다 (`_require_onnxruntime`).

### 7.1 `model.meta.json`

```json
{
  "captcha_id": "iptime", "name": "ipTIME", "rev": 1,
  "image_width": 168, "image_height": 60,      // 크롭 **후** = 모델 입력 크기 (감지값)
  "label_length": 5, "characters": "abcdfhijklmnopqrstuvwxy",   // 파일명에서 감지한 실측 문자셋
  "threshold": 255, "preprocess": "iptime",
  "crop": [27, 10, 195, 70], "crop_source": [200, 70],   // 크롭 없으면 둘 다 null
  "blank_index": 0
}
```

추론 시 `load_prediction_model()` 이 이 파일을 우선 적용한다 (§3.6).

---

## 8. 사용 예시

### 8.1 라이브러리

```python
from hypercaptcha import engine

# 1. 모델 생성 (레지스트리 기본 rev, device auto)
model = engine.get_captcha_model(train_data_base_dir="./captcha_data", captcha_id="supreme_court")

# 2. 학습 (Focal CTC, Warmup→Cosine, 80% train / 20% val)
engine.train_model(
    model=model,
    epochs=80,
    batch_size=32,
    early_stopping_patience=15,
    learning_rate=0.001,
    loss_type='focal',
    use_amp=True,
    on_event=lambda ev: print(ev['type'], ev),   # 선택: 진행 이벤트
)

# 3. 단건 추론
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

`import hypercaptcha` 최상위는 torch 를 로드하지 않는다. `engine` / `core` 를 import 하는 시점에 `core.py` 의 cuDNN/TF32 설정과 CUDA 프로브가 실행된다.

### 8.2 CLI (`hypercaptcha.cli`, `pyproject.toml` `[project.scripts] hypercaptcha`)

```bash
hypercaptcha -c supreme_court -i path/to/image.png        # 예측 문자열만 출력 (개행 없음)
hypercaptcha -c supreme_court -i path/to/image.png -v     # JSON (predicted_text/confidence/execution_time) + ./logs/main.log
python -m hypercaptcha -c supreme_court -i path/to/image.png
```

CLI 는 **현재 작업 디렉토리** 기준 `./captcha_data` 를 쓴다. 종료 코드: 이미지 없음 2, 모델 생성 실패 3.

### 8.3 학습/배치 평가 스크립트

```bash
python -m hypercaptcha.train   # 모듈 상단 상수(captcha_id, epochs=80, batch_size=64, patience=15) 를 고쳐 실행. pth/pt2/onnx/ort/meta 생성
python -m hypercaptcha.pred    # images/pred 일괄 평가 (batch_predict_model)
```
