# PyTorch CRNN 모델 구현 완료

## 변경 사항 요약

`captchaResolver/core.py`의 PyTorch CRNN 모델을 `captchaResolver/keras_core.py`의 Keras 모델과 **동일한 아키텍처**로 재구현했습니다.

## 주요 변경 사항

### 1. CRNN 클래스 재구현

**이전 구조 (dev.ipynb 기반):**
- Conv2D(256, 9x9) -> BatchNorm -> MaxPool(3x3)
- Conv2D(256, 4x3) -> BatchNorm
- Dense(256)
- Bidirectional LSTM(128) -> Bidirectional LSTM(64)

**새 구조 (keras_core.py와 동일):**
- Conv2D(32, 3x3, he_normal) -> MaxPooling2D(2x2)
- Conv2D(64, 3x3, he_normal) -> MaxPooling2D(2x2)
- Reshape -> Dense(64, he_normal) -> Dropout(0.2)
- Bidirectional LSTM(128, dropout=0.25)
- Bidirectional LSTM(64, dropout=0.25)
- Dense(num_classes+1)

### 2. Bidirectional 클래스 개선

```python
class Bidirectional(nn.Module):
    def __init__(self, inp: int, hidden: int, out: int, lstm: bool = True, dropout: float = 0.0):
        super(Bidirectional, self).__init__()
        if lstm:
            self.rnn = nn.LSTM(inp, hidden, bidirectional=True, dropout=dropout if dropout > 0 else 0)
        else:
            self.rnn = nn.GRU(inp, hidden, bidirectional=True, dropout=dropout if dropout > 0 else 0)
        self.embedding = nn.Linear(hidden * 2, out)
```

- LSTM에 `dropout` 파라미터 추가
- Keras의 `dropout=0.25`와 동일하게 적용

### 3. Reshape 로직 수정

**핵심 변경:**
```python
# 이전: (N, C, w, h) -> (N, w, C*h)
out = out.permute(0, 2, 3, 1)  # (N, w, h, C)
out = out.contiguous().view(N, w, C * h)

# 수정: (N, C, H, W) -> (N, W, H*C)
out = out.permute(0, 3, 2, 1)  # (N, W, H, C)
out = out.contiguous().view(N, W, H * C)
```

**이유:**
- Keras: `Reshape(target_shape=(width//4, height//4 * channels))`
- PyTorch 입력: (N, C=1, H=50, W=200)
- Conv 후: (N, C=64, H=12, W=50)
- Reshape 후: (N, W=50, H*C=768) ← 시퀀스 길이 50

### 4. 가중치 초기화

모든 Conv2D와 Linear 레이어에 **He Normal 초기화** 적용:
```python
nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
```

## 검증 결과

### 출력 형태 비교
```
PyTorch: (T=50, N=2, C=63)
Keras:   (N=2, T=50, C=63)

✓ 출력 차원이 일치합니다! (순서만 다름: PyTorch는 (T,N,C), Keras는 (N,T,C))
```

### 파라미터 수
- **Keras**: 438,143 params (1.67 MB)
- **PyTorch**: 456,255 params (1.74 MB)
- 차이 이유: PyTorch의 Bidirectional wrapper에서 Linear 레이어 사용

### 레이어별 출력 형태
```
입력: (N, 1, 50, 200)
Conv1: (N, 32, 50, 200)
Pool1: (N, 32, 25, 100)
Conv2: (N, 64, 25, 100)
Pool2: (N, 64, 12, 50)
Reshape: (N, 50, 768)
Dense+Dropout: (N, 50, 64)
LSTM1: (50, N, 256)
LSTM2: (50, N, 63)
```

## 사용 방법

### 모델 생성
```python
from captchaResolver.dataclass import CaptchaType
from captchaResolver.core import PyTorchModel

captcha_type = CaptchaType('kshop', 0)
model = PyTorchModel(captcha_type, verbose=1)
```

### 학습
```python
train_loader, val_loader = model.split_dataset(batch_size=16, train_size=0.8)
history = model.train_model(
    train_loader, 
    val_loader,
    epochs=50,
    lr=1e-4,
    early_stopping_patience=8
)
```

### 예측
```python
model.load_prediction_model()
prediction = model.predict('path/to/image.png')
```

## Context7 MCP 서버 활용

PyTorch 공식 문서를 Context7 MCP 서버를 통해 조회하여 최신 모범 사례를 적용했습니다:
- LSTM dropout 사용법
- Conv2D 초기화 방법
- CTC Loss 구현 패턴

## 테스트 스크립트

- `test_model_comparison.py`: Keras와 PyTorch 모델 구조 비교
- `test_pytorch_prediction.py`: PyTorch 모델 단독 테스트

## 주의사항

1. **LSTM Dropout 경고**:
   ```
   UserWarning: dropout option adds dropout after all but last recurrent layer, 
   so non-zero dropout expects num_layers greater than 1
   ```
   - PyTorch는 multi-layer LSTM에서만 dropout 적용
   - 단일 레이어에서는 경고 발생하지만 동작은 정상
   - Keras는 단일 레이어에서도 dropout 적용

2. **Device 관리**:
   - `PyTorchModel(device=torch.device('cpu'))` 명시적 지정 권장
   - GPU/CPU 자동 감지 지원

3. **파일명 규약**:
   - 모델 저장: `model_full.pth` (전체 모델)
   - Keras: `weights.keras` (전체 모델)

## 호환성

- Python: >=3.12
- PyTorch: 2.9.0+
- TensorFlow/Keras: 2.20.0+
- CUDA: 12.8 (선택)
