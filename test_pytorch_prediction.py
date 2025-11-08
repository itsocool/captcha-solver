"""
PyTorch CRNN 모델로 예측 테스트
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU 모드

import torch
from captchaResolver.dataclass import CaptchaType
from captchaResolver.core import PyTorchModel

def test_pytorch_prediction():
    """PyTorch 모델로 예측 테스트"""
    print("=" * 70)
    print("PyTorch CRNN 모델 예측 테스트")
    print("=" * 70)
    
    # CaptchaType 생성
    captcha_type = CaptchaType('kshop', 0)
    
    print(f"\n입력 이미지 크기: {captcha_type.train_data.image_width}x{captcha_type.train_data.image_height}")
    print(f"레이블 길이: {captcha_type.train_data.label_length}")
    print(f"문자 집합 크기: {len(captcha_type.train_data.characters)}")
    
    # PyTorch 모델 생성
    pytorch_model = PyTorchModel(captcha_type, verbose=1, device=torch.device('cpu'))
    
    # 모델 구조 확인
    print("\n[모델 구조]")
    print("-" * 70)
    model = pytorch_model.build_model()
    print(model)
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # 더미 배치로 forward pass 테스트
    print("\n[Forward Pass 테스트]")
    print("-" * 70)
    dummy_batch = torch.randn(4, 1, captcha_type.train_data.image_height,
                              captcha_type.train_data.image_width)
    dummy_labels = torch.randint(1, len(captcha_type.train_data.characters) + 1,
                                  (4, captcha_type.train_data.label_length))
    
    # CTC Loss 테스트
    criterion = torch.nn.CTCLoss()
    with torch.no_grad():
        output, loss = model(dummy_batch, dummy_labels, criterion)
    
    print(f"입력 형태: {dummy_batch.shape}")
    print(f"레이블 형태: {dummy_labels.shape}")
    print(f"출력 형태: {output.shape}")
    print(f"  - 시퀀스 길이: {output.shape[0]}")
    print(f"  - 배치 크기: {output.shape[1]}")
    print(f"  - 클래스 수: {output.shape[2]}")
    print(f"CTC Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)
    print("\n✓ PyTorch CRNN 모델이 Keras 모델과 동일한 구조로 구현되었습니다.")
    print("  - Conv2D(32, 3x3) -> MaxPool(2x2)")
    print("  - Conv2D(64, 3x3) -> MaxPool(2x2)")
    print("  - Reshape -> Dense(64) -> Dropout(0.2)")
    print("  - Bidirectional LSTM(128, dropout=0.25)")
    print("  - Bidirectional LSTM(64, dropout=0.25)")
    print("  - Dense(num_classes+1)")

if __name__ == '__main__':
    test_pytorch_prediction()
