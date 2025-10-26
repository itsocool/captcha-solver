import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 로깅 완전 억제
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN 최적화 비활성화 (경고 제거)
os.environ['GLOG_minloglevel'] = '2'  # Google 로깅

import captchaResolver.engine as engine
from captchaResolver.dataclass import TrainData
from captchaResolver.core import PyTorchModel

if __name__ == '__main__':
    # 학습 설정 (dev.ipynb 스타일)
    captcha_id = 'default'
    backend = 'pytorch'
    epochs = 100  # dev.ipynb 기본값
    batch_size = 32  # dev.ipynb 기본값
    early_stopping_patience = 10
    learning_rate = 1e-4  # dev.ipynb 기본값
    num_workers = 0  # 단순화 (필요시 증가)
    warmup_epochs = 0  # dev.ipynb는 warmup 미사용
    save_model = False
    
    model: PyTorchModel = engine.get_captcha_model(captcha_id=captcha_id, backend=backend)
    train_data: TrainData = model.train_data
    
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

    engine.batch_predict_model(
        model=model,
        batch_size=batch_size,
    )
