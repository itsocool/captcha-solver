import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'

import logging, warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from captchaResolver.dataclass import TrainData
import captchaResolver.engine as engine
from captchaResolver.keras_core import KerasModel

captcha_id = 'gov24'
backend = 'keras'
rev = 1
image_width = 200
image_height = 50

model: KerasModel = engine.get_captcha_model(captcha_id=captcha_id, backend=backend)
train_data: TrainData = model.train_data
model.train_data.rev = rev
model.train_data.image_width = image_width
model.train_data.image_height = image_height
engine.batch_predict_model(model=model)
# model_path = train_data.get_model_path()
# image_path = train_data.choice_pred_image()
# pred, confidence = engine.predict(model=model, image_path=image_path)
# print("image_path : ", image_path)
# print("pred : ", pred)
# print("confidence : ", f'{confidence:.4f}')
# print("Done!")
