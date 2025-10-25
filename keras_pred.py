from calendar import c
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow C++ 로그 레벨 (0=ALL,1=INFO,2=WARNING,3=ERROR)
os.environ['GLOG_minloglevel'] = '2'  # Google 로깅

import logging, warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from captchaResolver.engine import get_captcha_type_list, batch_predict_model, predict
from captchaResolver.keras_core import KerasModel

captcha_id = 'default'
backend = 'keras'

captcha_type = get_captcha_type_list(backend=backend)[captcha_id]
train_data = captcha_type.train_data
train_data.threshold = 60
model = KerasModel(captcha_type=captcha_type)
batch_predict_model(model=model)
model_path = train_data.get_model_path()
image_path = train_data.choice_pred_image()
pred, confidence = predict(model=model, image_path=image_path, model_path=model_path)
print("image_path : ", image_path)
print("pred : ", pred)
print("confidence : ", f'{confidence:.4f}')
print("Done!")
