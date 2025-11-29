from captchaResolver.base_core import BaseModel
import captchaResolver.engine as engine
from pathlib import Path
from captchaResolver.dataclass import TrainData

captcha_id = 'dev'
rev = 0
train_ratio = 0.9
shuffle = False
model: BaseModel = engine.get_captcha_model(captcha_id=captcha_id)
train_data: TrainData = model.train_data
model.train_data.rev = rev

if shuffle:
    image_dir = Path(train_data.get_image_dir()).parent.as_posix()
    engine.redistribute_train_pred(image_dir=image_dir, train_ratio=train_ratio)

engine.batch_predict_model(model=model)
# engine.batch_predict_model(model=model, pred_image_dir="captcha_data/supreme_court/0/images/labeled")

# pred = engine.predict(model=model, image_path="captcha_data/supreme_court/0/images/pred/133171.png")
