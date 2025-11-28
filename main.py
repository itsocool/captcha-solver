import os
import captchaResolver.engine as engine
from captchaResolver.dataclass import TrainData
from captchaResolver.backend.pytorch.core import PyTorchModel

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
    model_base_dir = engine.train_model(
        model,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        learning_rate=learning_rate,
        num_workers=num_workers,
        warmup_epochs=warmup_epochs,
    )
    print(f"Model trained and saved at: {model_base_dir}{os.path.sep}weights.keras")
    print("Done!")

