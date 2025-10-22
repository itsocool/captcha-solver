import os

# 환경 변수 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'

from captchaResolver.engine import get_captcha_type_list
from captchaResolver.core import PyTorchModel

def main():
    """Main prediction function with optimized settings."""
    # 예측 설정
    captcha_id = 'kshop'
    batch_size = 32
    
    # Windows에서는 num_workers=0 권장 (multiprocessing 이슈 방지)
    num_workers = 0 if os.name == 'nt' else 4
    
    # CAPTCHA 타입 및 학습 데이터 설정
    captcha_type_list = get_captcha_type_list()
    train_data = captcha_type_list[captcha_id].train_data
    train_data.threshold = 60
    
    # PyTorch 모델 초기화
    model = PyTorchModel(
        train_data=train_data,
        verbose=1,
        use_compile=False,  # 추론 시에는 compile 비활성화 (더 빠른 시작)
        use_amp=False  # 추론 시에는 AMP 비활성화 (더 정확한 결과)
    )
    
    print("=" * 70)
    print(f"Prediction Configuration:")
    print(f"  CAPTCHA ID: {captcha_id}")
    print(f"  Image size: {train_data.image_width}x{train_data.image_height}")
    print(f"  Label length: {train_data.label_length}")
    print(f"  Characters: {train_data.characters}")
    print(f"  Threshold: {train_data.threshold}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of workers: {num_workers}")
    print("=" * 70)
    
    # 모델 로드
    print("\nLoading model...")
    model.load_prediction_model()
    
    # 검증 데이터셋으로 평가
    print("\nPreparing validation dataset...")
    _, val_loader = model.split_dataset(
        batch_size=batch_size,
        train_size=0.9,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False  # 추론 시에는 pin_memory 비활성화
    )
    
    print("\nValidating model...")
    val_loss, accuracy = model.validate_model(val_loader)
    
    print("\n" + "=" * 70)
    print(f"Validation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print("=" * 70)
    
    # 단일 이미지 예측 예제 (선택적)
    """
    print("\nSingle image prediction example:")
    image_path = "captcha_data/kshop/0/images/pred/example.png"
    if os.path.exists(image_path):
        pred_text = model.predict(image_path)
        print(f"  Image: {image_path}")
        print(f"  Predicted: {pred_text}")
    else:
        print(f"  Image not found: {image_path}")
    """
    
    print("\nPrediction completed!")


if __name__ == '__main__':
    main()
