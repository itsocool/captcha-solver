import os
import captchaResolver.engine as engine
from captchaResolver.backend.pytorch.core import PyTorchModel

captcha_id = 'supreme_court'
loss_type = 'focal'
use_amp = True
model: PyTorchModel = engine.get_captcha_model(captcha_id=captcha_id)
train_ratio = 0.6
shuffle = False
batch_size = 64
epochs = 60
early_stopping_patience = 12

if shuffle:
    image_dir = os.path.dirname(model.train_data.get_image_dir())
    engine.redistribute_train_pred(image_dir=image_dir, train_ratio=train_ratio)

model_base_dir = engine.train_model(
    model=model,
    epochs=epochs,
    batch_size=batch_size,
    early_stopping_patience=early_stopping_patience,
    loss_type=loss_type,
    use_amp=use_amp,
)
print(f"Model trained and saved at: {model.train_data.get_model_path()}")
print("Done!")
