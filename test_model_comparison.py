"""
Keras와 PyTorch CRNN 모델 구조 비교 테스트
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # TensorFlow GPU 비활성화

import torch
import numpy as np
from captchaResolver.dataclass import CaptchaType
from captchaResolver.core import PyTorchModel
from captchaResolver.keras_core import KerasModel

def test_model_architecture():
    """모델 아키텍처 비교"""
    print("=" * 70)
    print("모델 아키텍처 비교 테스트")
    print("=" * 70)
    
    # CaptchaType 생성 (kshop 예시)
    captcha_type = CaptchaType('kshop', 0)
    
    # 이미지 크기 출력
    print(f"\n입력 이미지 크기: {captcha_type.train_data.image_width}x{captcha_type.train_data.image_height}")
    print(f"레이블 길이: {captcha_type.train_data.label_length}")
    print(f"문자 집합 크기: {len(captcha_type.train_data.characters)}")
    
    # PyTorch 모델
    print("\n[PyTorch CRNN 모델 구조]")
    print("-" * 70)
    # CPU에서 테스트
    pytorch_model = PyTorchModel(captcha_type, verbose=0, device=torch.device('cpu'))
    model_pytorch = pytorch_model.build_model()
    model_pytorch.to('cpu')  # 명시적으로 CPU로 이동
    print(model_pytorch)
    
    # 더미 입력으로 forward pass 테스트
    dummy_input = torch.randn(2, 1, captcha_type.train_data.image_height, 
                              captcha_type.train_data.image_width)
    with torch.no_grad():
        output_pytorch, _ = model_pytorch(dummy_input)
    
    print(f"\nPyTorch 출력 형태: {output_pytorch.shape}")
    print(f"  - 시퀀스 길이 (T): {output_pytorch.shape[0]}")
    print(f"  - 배치 크기 (N): {output_pytorch.shape[1]}")
    print(f"  - 클래스 수 (+1 for blank): {output_pytorch.shape[2]}")
    
    # Keras 모델
    print("\n[Keras CRNN 모델 구조]")
    print("-" * 70)
    import tensorflow as tf
    # TensorFlow를 CPU로 강제
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')
    
    keras_model_wrapper = KerasModel(captcha_type, verbose=0)
    model_keras = keras_model_wrapper.build_model(prediction_only=True)
    model_keras.summary()
    
    # 더미 입력으로 forward pass 테스트
    # Keras 모델 입력 형식: (batch, width, height, channels)
    dummy_input_keras = tf.random.normal((2, captcha_type.train_data.image_width,
                                          captcha_type.train_data.image_height, 1))
    output_keras = model_keras.predict(dummy_input_keras, verbose=0)
    
    print(f"\nKeras 출력 형태: {output_keras.shape}")
    print(f"  - 배치 크기 (N): {output_keras.shape[0]}")
    print(f"  - 시퀀스 길이 (T): {output_keras.shape[1]}")
    print(f"  - 클래스 수 (+1 for blank): {output_keras.shape[2]}")
    
    # 출력 형태 비교
    print("\n[출력 형태 비교]")
    print("-" * 70)
    pytorch_T, pytorch_N, pytorch_C = output_pytorch.shape
    keras_N, keras_T, keras_C = output_keras.shape
    
    print(f"PyTorch: (T={pytorch_T}, N={pytorch_N}, C={pytorch_C})")
    print(f"Keras:   (N={keras_N}, T={keras_T}, C={keras_C})")
    
    if pytorch_T == keras_T and pytorch_N == keras_N and pytorch_C == keras_C:
        print("\n✓ 출력 차원이 일치합니다! (순서만 다름: PyTorch는 (T,N,C), Keras는 (N,T,C))")
    else:
        print("\n✗ 출력 차원이 일치하지 않습니다.")
        if pytorch_T != keras_T:
            print(f"  - 시퀀스 길이 불일치: PyTorch={pytorch_T}, Keras={keras_T}")
        if pytorch_C != keras_C:
            print(f"  - 클래스 수 불일치: PyTorch={pytorch_C}, Keras={keras_C}")
    
    # 예상 시퀀스 길이 계산 (Keras 방식)
    expected_T = captcha_type.train_data.image_width // 4
    print(f"\n예상 시퀀스 길이: {expected_T} (image_width // 4)")
    
    print("\n" + "=" * 70)
    print("테스트 완료")
    print("=" * 70)

if __name__ == '__main__':
    test_model_architecture()
