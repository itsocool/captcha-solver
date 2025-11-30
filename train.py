import os
from pathlib import Path
import captchaResolver.engine as engine
from captchaResolver.dataclass import TrainData
from captchaResolver.backend.pytorch.core import PyTorchModel

captcha_id = 'dev'
rev = 0
train_ratio = 0.6
shuffle = False
epochs = 120
batch_size = 64
early_stopping_patience = 12
model: PyTorchModel = engine.get_captcha_model(captcha_id=captcha_id)
train_data: TrainData = model.train_data
model.train_data.rev = rev

if shuffle:
    image_dir = Path(train_data.get_image_dir()).parent.as_posix()
    engine.redistribute_train_pred(image_dir=image_dir, train_ratio=train_ratio)

model_base_dir = engine.train_model(
    model=model,
    epochs=epochs,
    batch_size=batch_size,
    early_stopping_patience=early_stopping_patience,
    loss_type="label_smoothing",
)
print(f"Model trained and saved at: {model_base_dir}{os.path.sep}model_full.pth")
print("Done!")
