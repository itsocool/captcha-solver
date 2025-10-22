import os
# TensorFlow 관련 환경 변수 설정 (로깅 억제)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 로깅 완전 억제
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN 최적화 비활성화 (경고 제거)
os.environ['GLOG_minloglevel'] = '2'  # Google 로깅

from captchaResolver.torchcore import PyTorchModel
from captchaResolver.torchengine import train_model
from captchaResolver.keras_core import get_captcha_type_list

if __name__ == '__main__':
    # 학습 설정
    captcha_id = 'kshop'
    epochs = 100
    batch_size = 32
    early_stopping_patience = 16
    learning_rate = 0.001
    num_workers = 4  # 데이터 로더 워커 수 (CPU 코어 수에 맞게 조정)

    # CAPTCHA 타입 및 학습 데이터 설정
    captcha_type_list = get_captcha_type_list()
    train_data = captcha_type_list[captcha_id].train_data
    train_data.threshold = 60

    # PyTorch 모델 초기화
    model = PyTorchModel(
        train_data=train_data,
        verbose=1,  # 로깅 레벨 (0: 없음, 1: 기본, 2: 상세)
        use_compile=False,  # torch.compile 사용 여부 (PyTorch 2.0+)
        use_amp=True  # Mixed precision training 사용 여부
    )

    print("="*60)
    print(f"Training Configuration:")
    print(f"  CAPTCHA ID: {captcha_id}")
    print(f"  Image size: {train_data.image_width}x{train_data.image_height}")
    print(f"  Label length: {train_data.label_length}")
    print(f"  Characters: {train_data.characters}")
    print(f"  Threshold: {train_data.threshold}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Number of workers: {num_workers}")
    print("="*60)

    # 모델 학습
    model_base_dir = train_model(
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        earlystopping=True,
        early_stopping_patience=early_stopping_patience,
        learning_rate=learning_rate,
        num_workers=num_workers
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
