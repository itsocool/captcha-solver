import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow C++ 로그 레벨 (0=ALL,1=INFO,2=WARNING,3=ERROR)
os.environ['GLOG_minloglevel'] = '2'  # Google 로깅

import logging, warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import captchaResolver.keras_engine as engine
from pathlib import Path
from captchaResolver.dataclass import TrainData
from captchaResolver.keras_core import KerasModel

captcha_id = 'gov24'
backend = 'keras'
rev = 1
image_width = 200
image_height = 50
epochs = 120
batch_size = 32
early_stopping_patience = 8
train_ratio = 0.9
shuffle = False

model: KerasModel = engine.get_captcha_model(captcha_id=captcha_id, backend=backend)
train_data: TrainData = model.train_data
model.train_data.rev = rev
model.train_data.image_width = image_width
model.train_data.image_height = image_height

if shuffle:
    image_dir = Path(train_data.get_image_dir()).parent.as_posix()
    engine.redistribute_train_pred(image_dir=image_dir, train_ratio=train_ratio)

model_base_dir = engine.train_model(
    model=model,
    epochs=epochs,
    batch_size=batch_size,
    early_stopping_patience=early_stopping_patience,
)
print(f"Model trained and saved at: {model_base_dir}{os.path.sep}weights.keras")
print("Done!")
