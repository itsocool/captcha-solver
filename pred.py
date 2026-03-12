import os
import captchaResolver.engine as engine
from captchaResolver.backend.pytorch.core import PyTorchModel

captcha_id = 'supreme_court'
loss_type = 'focal'
use_amp = True
model_type = 'default'
model: PyTorchModel = engine.get_captcha_model(captcha_id=captcha_id)
train_ratio = 0.9
shuffle = False

if shuffle:
    image_dir = os.path.dirname(model.train_data.get_image_dir())
    engine.redistribute_train_pred(image_dir=image_dir, train_ratio=train_ratio)

engine.batch_predict_model(model=model,loss_type=loss_type,model_type=model_type)
# engine.batch_predict_model(model=model, pred_image_dir="captcha_data/gov24/1/images/draft", loss_type="label_smoothing")
# pred = engine.predict(model=model, image_path="captcha_data/gov24/1/images/draft/스크린샷 1.jpg")
# pred = engine.onnx_predict(model, image_path="captcha_data/gov24/1/images/draft/스크린샷 1.JPG")