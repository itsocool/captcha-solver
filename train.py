import os
# TensorFlow 관련 환경 변수 설정 (로깅 억제)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 로깅 완전 억제
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN 최적화 비활성화 (경고 제거)
os.environ['GLOG_minloglevel'] = '2'  # Google 로깅

from captchaResolver.core import PyTorchModel
from captchaResolver.engine import get_captcha_type_list, train_model

if __name__ == '__main__':
    # 학습 설정 (dev.ipynb 스타일)
    captcha_id = 'kshop'
    backend = 'pytorch'
    epochs = 60  # dev.ipynb 기본값
    batch_size = 32  # dev.ipynb 기본값
    early_stopping_patience = 8
    learning_rate = 1e-4  # dev.ipynb 기본값
    num_workers = 0  # 단순화 (필요시 증가)
    warmup_epochs = 0  # dev.ipynb는 warmup 미사용

    # CAPTCHA 타입 및 학습 데이터 설정
    captcha_type_list = get_captcha_type_list()
    train_data = captcha_type_list[captcha_id].train_data
    train_data.backend = backend
    train_data.threshold = 60

    # PyTorch 모델 초기화
    model = PyTorchModel(
        train_data=train_data,
        verbose=1
    )

    print("="*60)
    print(f"Training Configuration (dev.ipynb based):")
    print(f"  CAPTCHA ID: {captcha_id}")
    print(f"  Image size: {train_data.image_width}x{train_data.image_height}")
    print(f"  Label length: {train_data.label_length}")
    print(f"  Characters: {train_data.characters}")
    print(f"  Threshold: {train_data.threshold}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Model architecture:")
    print(f"    - CRNN (dev.ipynb based)")
    print(f"    - CNN: Conv2d(256) + MaxPool + Conv2d(256)")
    print(f"    - RNN: Bidirectional LSTM (1024 hidden)")
    print(f"    - Dynamic Linear layer (initialized on first forward)")
    print(f"    - CTC Loss with log_softmax")
    print("="*60)

    # 모델 학습
    model_base_dir = train_model(
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        earlystopping=True,
        early_stopping_patience=early_stopping_patience,
        learning_rate=learning_rate,
        num_workers=num_workers,
        warmup_epochs=warmup_epochs
    )

    # 학습 완료 메시지
    model_path = train_data.get_model_path()
    print("\n" + "="*60)
    print(f"Training completed successfully!")
    print(f"Model saved at: {model_path}")
    print(f"Model directory: {model_base_dir}")
    print("="*60)
    print("\nYou can now use this model for prediction.")
    print(f"Run: python pred_torch.py")
