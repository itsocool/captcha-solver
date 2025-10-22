import os
os.environ['GLOG_minloglevel'] = '2'  # Google 로깅

from captchaResolver.torchcore import PyTorchModel
from captchaResolver.torchengine import batch_predict_model, predict
from captchaResolver.keras_core import get_captcha_type_list

# 예측 설정
captcha_id = 'kshop'
batch_size = 32
num_workers = 4

# CAPTCHA 타입 및 학습 데이터 설정
captcha_type_list = get_captcha_type_list()
train_data = captcha_type_list[captcha_id].train_data
train_data.threshold = 60

# PyTorch 모델 초기화
model = PyTorchModel(
    train_data=train_data,
    verbose=1,
    use_compile=False,
    use_amp=False  # 추론 시에는 AMP 비활성화 (더 정확한 결과)
)

print("="*60)
print(f"Prediction Configuration:")
print(f"  CAPTCHA ID: {captcha_id}")
print(f"  Image size: {train_data.image_width}x{train_data.image_height}")
print(f"  Label length: {train_data.label_length}")
print(f"  Characters: {train_data.characters}")
print(f"  Threshold: {train_data.threshold}")
print(f"  Batch size: {batch_size}")
print(f"  Number of workers: {num_workers}")
print("="*60)

# 배치 예측 수행
results = batch_predict_model(
    model=model,
    batch_size=batch_size,
    num_workers=num_workers
)

# 단일 이미지 예측 예제 (선택적)
# 특정 이미지 파일에 대해 예측하고 싶다면 아래 코드 사용
"""
image_path = "captcha_data/kshop/0/images/pred/example.png"
if os.path.exists(image_path):
    pred_text, confidence = predict(model, image_path)
    print(f"\nSingle Image Prediction:")
    print(f"  Image: {image_path}")
    print(f"  Predicted: {pred_text}")
    print(f"  Confidence: {confidence:.4f}")
"""

print("\nPrediction completed!")
