import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow C++ 로그 레벨 (0=ALL,1=INFO,2=WARNING,3=ERROR)
os.environ['GLOG_minloglevel'] = '2'  # Google 로깅

import logging
import warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from captchaResolver.keras_core import KerasModel
from captchaResolver.engine import get_captcha_type_list
from captchaResolver.keras_engine import predict, batch_predict_model

captcha_id = 'kshop'
captcha_type_list = get_captcha_type_list()
train_data = captcha_type_list[captcha_id].train_data
train_data.backend = 'keras'
train_data.threshold = 60
model = KerasModel(train_data=train_data)
batch_predict_model(model=model)
model_path = train_data.get_model_path()
image_path = train_data.get_pred_image_path()
pred, confidence = predict(model=model, image_path=image_path, model_path=model_path)
print("image_path : ", image_path)
print("pred : ", pred)
print("confidence : ", confidence)
print("Done!")
