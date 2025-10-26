import os

from captchaResolver.dataclass import TrainData
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'

import captchaResolver.engine as engine
from captchaResolver.core import PyTorchModel

captcha_id = 'default'
backend = 'pytorch'

model: PyTorchModel = engine.get_captcha_model(captcha_id=captcha_id, backend=backend)
train_data: TrainData = model.train_data
engine.batch_predict_model(model=model)
model_path = train_data.get_model_path()
image_path = train_data.choice_pred_image()
pred, confidence = engine.predict(model=model, image_path=image_path)
print("image_path : ", image_path)
print("pred : ", pred)
print("confidence : ", f'{confidence:.4f}')
print("Done!")
