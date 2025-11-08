import os
from pathlib import Path
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow C++ 로그 레벨 (0=ALL,1=INFO,2=WARNING,3=ERROR)
# os.environ['GLOG_minloglevel'] = '2'  # Google 로깅

# import logging, warnings
# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=DeprecationWarning)

import captchaResolver.engine as engine
from captchaResolver.dataclass import TrainData
from captchaResolver.core import PyTorchModel

captcha_id = 'gov24'
backend = 'pytorch'
rev = 1
image_width = 200
image_height = 50
epochs = 60
batch_size = 32
early_stopping_patience = 6
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

model_base_dir = engine.train_model(
    model=model,
    epochs=epochs,
    batch_size=batch_size,
    early_stopping_patience=early_stopping_patience,
)
print(f"Model trained and saved at: {model_base_dir}{os.path.sep}model_full.pth")
print("Done!")
