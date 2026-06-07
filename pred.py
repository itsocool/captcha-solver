import os
import engine
from core import PyTorchModel

captcha_id = 'supreme_court'
loss_type = 'focal'
use_amp = True
model: PyTorchModel = engine.get_captcha_model(captcha_id=captcha_id)
train_ratio = 0.9

engine.batch_predict_model(model=model,loss_type=loss_type)
# pred = engine.predict(model=model, image_path="captcha_data/gov24/1/images/draft/스크린샷 1.jpg")
