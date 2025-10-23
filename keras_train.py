import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow C++ 로그 레벨 (0=ALL,1=INFO,2=WARNING,3=ERROR)
os.environ['GLOG_minloglevel'] = '2'  # Google 로깅

import logging, warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from captchaResolver.engine import get_captcha_type_list
from captchaResolver.keras_core import KerasModel
from captchaResolver.keras_engine import train_model

captcha_id = 'default'
captcha_type_list = get_captcha_type_list()
train_data = captcha_type_list[captcha_id].train_data
train_data.backend = 'keras'
train_data.threshold = 60
epochs = 120
batch_size = 32
early_stopping_patience = 16

model = KerasModel(train_data=train_data)
model_base_dir = train_model(
    model,
    epochs=epochs,
    batch_size=batch_size,
    early_stopping_patience=early_stopping_patience)
print(f"Model trained and saved at: {model_base_dir}{os.path.sep}weights.keras")
print("Done!")
