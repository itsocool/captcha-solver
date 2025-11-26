# CRNN 모델 및 학습 파이프라인 개선 보고서

## 개요
`captchaResolver/core.py`의 CRNN 모델 아키텍처와 학습 루틴을 대폭 개선하였습니다. 이번 업데이트는 모델의 특징 추출 능력을 강화하고, 학습 속도와 안정성을 높이며, 데이터 증강을 통해 일반화 성능을 확보하는 데 중점을 두었습니다.

## 1. 모델 아키텍처 개선 (VGG Backbone 도입)

기존의 단순한 2-Layer CNN 구조를 VGG 스타일의 깊은 CNN 백본으로 교체하여 복잡한 캡챠 이미지에서도 더 강력한 특징을 추출할 수 있도록 했습니다.

### 변경 전 (Legacy)
- **구조**: Conv2d(256) -> MaxPool -> Conv2d(256) -> Linear
- **특징**: 얕은 구조로 인해 복잡한 노이즈나 왜곡이 심한 캡챠에 취약할 수 있음.
- **입력 처리**: 고정된 높이/너비에 의존적이며, `view`와 `permute`로 수동 차원 변환.

### 변경 후 (Improved)
- **구조**: VGG-Style Deep CNN (4 Blocks)
  - Block 1: Conv(64) -> MaxPool
  - Block 2: Conv(128) -> MaxPool
  - Block 3: Conv(256) -> BatchNorm -> Conv(256) -> MaxPool(Height only)
  - Block 4: Conv(512) -> BatchNorm -> Conv(512) -> MaxPool(Height only)
  - Feature Map: Conv(512)
- **Adaptive Pooling**: `nn.AdaptiveAvgPool2d((1, None))`을 사용하여 입력 이미지의 높이가 달라져도 강건하게 시퀀스(Width) 차원으로 압축합니다.
- **RNN**: CNN 출력 채널(512)을 입력으로 받는 Bidirectional LSTM 연결.

### 기대 효과
- **정확도 향상**: 더 깊은 레이어와 Batch Normalization을 통해 학습 안정성과 표현력이 향상되었습니다.
- **강건함**: 다양한 스타일의 폰트와 노이즈에 대해 더 잘 견딥니다.

## 2. 학습 파이프라인 최적화

최신 PyTorch 기능을 적극 도입하여 학습 효율성을 극대화했습니다.

### 주요 변경 사항
1.  **Mixed Precision (AMP) 적용**:
    - `torch.cuda.amp.autocast`와 `GradScaler`를 도입했습니다.
    - **효과**: GPU 메모리 사용량을 줄이고(약 30~50% 절감), 최신 GPU(Tensor Core)에서 연산 속도가 빨라집니다.
2.  **Torch Compile (PyTorch 2.0+)**:
    - `torch.compile(model)`을 추가했습니다.
    - **효과**: 모델 그래프를 최적화하여 추론 및 학습 속도가 향상됩니다.
3.  **Scheduler 개선**:
    - 기존: 단순 Warmup 후 고정 LR.
    - 변경: `ReduceLROnPlateau` 적용. Validation Loss가 정체될 때 학습률을 동적으로 감소(Factor 0.5)시켜 더 낮은 Loss로 수렴하도록 유도합니다.

## 3. 데이터 증강 (Data Augmentation)

학습 데이터셋에 실시간 증강을 적용하여 과적합(Overfitting)을 방지하고 일반화 성능을 높였습니다.

### 적용된 기법
- **ColorJitter**: 밝기(Brightness)와 대비(Contrast)를 무작위로 조절 (±20%).
- **RandomAffine**: 미세한 회전, 이동, 스케일 조절.
  - Scale: 0.9 ~ 1.1 (이미지 크기 변화 대응)
  - Translate: 5% (위치 변화 대응)

## 4. 마이그레이션 가이드

**주의**: 모델 구조가 변경되었으므로, **기존에 학습된 가중치(`weights.pth`, `model_full.pth`)와는 호환되지 않습니다.**

1.  **재학습 필요**: `train.py`를 실행하여 새로운 모델 구조로 처음부터 다시 학습해야 합니다.
2.  **하이퍼파라미터**: 모델이 깊어졌으므로 `batch_size`를 조절해야 할 수 있으나, AMP 적용으로 인해 기존과 비슷하거나 더 큰 배치를 사용할 수 있습니다.

---
**작성일**: 2025년 11월 20일
**작성자**: GitHub Copilot
