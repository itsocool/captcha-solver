import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

from captchaResolver.core import Model, get_captcha_type_list

captcha_id = 'kshop'
captcha_type_list = get_captcha_type_list()
train_data = captcha_type_list[captcha_id].train_data
train_data.threshold = 60
epochs = 120
batch_size = 32
earlystopping = True
early_stopping_patience = 16
save_weights = True
save_model = True

model = Model(train_data=train_data, hard_mode=True) 
model.train_model(
    epochs=epochs,
    batch_size=batch_size,
    earlystopping=earlystopping,
    early_stopping_patience=early_stopping_patience)

print("Done!")
