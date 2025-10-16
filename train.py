import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

model = Model(train_data=train_data, save_model=save_model, save_weights=save_weights) 
model.train_model(
    epochs=epochs,
    batch_size=batch_size,
    earlystopping=earlystopping,
    early_stopping_patience=early_stopping_patience,
    save_weights=save_weights,
    save_model=save_model)

print("Done!")
