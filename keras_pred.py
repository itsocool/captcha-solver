import os

from captchaResolver.dataclass import TrainData
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow C++ 로그 레벨 (0=ALL,1=INFO,2=WARNING,3=ERROR)
os.environ['GLOG_minloglevel'] = '2'  # Google 로깅

import logging, warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import captchaResolver.engine as engine
from captchaResolver.keras_core import KerasModel

captcha_id = 'default'
backend = 'keras'

model: KerasModel = engine.get_captcha_model(captcha_id=captcha_id, backend=backend)
train_data: TrainData = model.train_data
engine.batch_predict_model(model=model)
model_path = train_data.get_model_path()
image_path = train_data.choice_pred_image()
pred, confidence = engine.predict(model=model, image_path=image_path)
print("image_path : ", image_path)
print("pred : ", pred)
print("confidence : ", f'{confidence:.4f}')
print("Done!")
