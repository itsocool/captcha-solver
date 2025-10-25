import os

from captchaResolver import engine

# 환경 변수 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'

from captchaResolver.engine import batch_predict_model, get_captcha_type_list, get_model

def main():
    """PyTorch 모델을 사용한 배치 추론 및 결과 표시"""
    # 예측 설정
    captcha_id = 'kshop'
    backend = 'pytorch'
    batch_size = 32
    # num_workers = 0 if os.name == 'nt' else 4
    
    # CAPTCHA 타입 및 학습 데이터 설정
    captcha_type_list = get_captcha_type_list(backend=backend)
    train_data = captcha_type_list[captcha_id].train_data
    train_data.backend = backend
    train_data.threshold = 60
    
    model = get_model(train_data=train_data)
    
    print("=" * 70)
    print(f"Prediction Configuration:")
    print(f"  CAPTCHA ID: {captcha_id}")
    print(f"  Backend: {backend}")
    print(f"  Image size: {train_data.image_width}x{train_data.image_height}")
    print(f"  Label length: {train_data.label_length}")
    print(f"  Characters: {train_data.characters}")
    print(f"  Threshold: {train_data.threshold}")
    print(f"  Batch size: {batch_size}")
    print("=" * 70)
    
    batch_predict_model(
        model=model,
        batch_size=batch_size,
    )

if __name__ == '__main__':
    main()