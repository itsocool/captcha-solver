import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 사용 시
os.environ['GLOG_minloglevel'] = '2'  # Google 로깅

from captchaResolver.core import KerasModel, get_captcha_type_list
from captchaResolver.engine import train_model

captcha_id = 'kshop'
captcha_type_list = get_captcha_type_list()
train_data = captcha_type_list[captcha_id].train_data
train_data.threshold = 60
epochs = 120
batch_size = 32
early_stopping_patience = 16

model = KerasModel(train_data=train_data)
train_model(
    model,
    epochs=epochs,
    batch_size=batch_size,
    early_stopping_patience=early_stopping_patience)

print("Done!")
