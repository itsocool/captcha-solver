import os
from pathlib import Path
import captchaResolver.engine as engine
from captchaResolver.dataclass import TrainData
from captchaResolver.core import PyTorchModel

captcha_id = 'supreme_court'
backend = 'pytorch'
rev = 0
# image_width = 120
# image_height = 40
train_ratio = 0.9
shuffle = False
epochs = 60
batch_size = 32
early_stopping_patience = 6

model: PyTorchModel = engine.get_captcha_model(captcha_id=captcha_id, backend=backend)
train_data: TrainData = model.train_data
model.train_data.rev = rev
# model.train_data.image_width = image_width
# model.train_data.image_height = image_height

if shuffle:
    image_dir = Path(train_data.get_image_dir()).parent.as_posix()
    engine.redistribute_train_pred(image_dir=image_dir, train_ratio=train_ratio)

model_base_dir = engine.train_model(
    model=model,
    epochs=epochs,
    batch_size=batch_size,
    early_stopping_patience=early_stopping_patience,
)
print(f"Model trained and saved at: {model_base_dir}{os.path.sep}model_full.pth")
print("Done!")
