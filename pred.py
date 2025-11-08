import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['GLOG_minloglevel'] = '2'

# import logging, warnings
# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=DeprecationWarning)

import captchaResolver.engine as engine
from pathlib import Path
from captchaResolver.dataclass import TrainData
from captchaResolver.core import PyTorchModel

captcha_id = 'gov24'
backend = 'pytorch'
rev = 1
image_width = 200
image_height = 50
train_ratio = 0.9
shuffle = False

model: PyTorchModel = engine.get_captcha_model(captcha_id=captcha_id, backend=backend)
train_data: TrainData = model.train_data
model.train_data.rev = rev
model.train_data.image_width = image_width
model.train_data.image_height = image_height

if shuffle:
    image_dir = Path(train_data.get_image_dir()).parent.as_posix()
    engine.redistribute_train_pred(image_dir=image_dir, train_ratio=train_ratio)

engine.batch_predict_model(model=model)

engine.predict(model=model, image_path='captcha_data/gov24/1/images/pred/076074.png')
