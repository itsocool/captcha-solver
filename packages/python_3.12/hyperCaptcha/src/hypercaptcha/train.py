import os
from . import engine
from .core import PyTorchModel

captcha_id = 'supreme_court'
loss_type = 'focal'
use_amp = True
model: PyTorchModel = engine.get_captcha_model(captcha_id=captcha_id)
train_ratio = 0.6
shuffle = False
batch_size = 64
# CTC 는 초반에 입력과 무관한 상수 문자열만 뱉는 정체 구간을 거친다. kshop rev0 은
# 그 구간이 epoch 2~28 로 길고, 안에서 무개선이 7에폭 연속까지 난다. 예전 값(40/6)은
# 거기 걸려 epoch 20 에서 죽었다. 근거는 services/train.py 의 PARAM_SPEC 주석 참고.
epochs = 80
early_stopping_patience = 15

if shuffle:
    image_dir = os.path.dirname(model.train_data.get_image_dir())
    engine.redistribute_train_pred(image_dir=image_dir, train_ratio=train_ratio)

model_base_dir = engine.train_model(
    model=model,
    epochs=epochs,
    batch_size=batch_size,
    early_stopping_patience=early_stopping_patience,
    loss_type=loss_type,
    use_amp=use_amp,
)
print(f"Model trained and saved at: {model.train_data.get_model_path()}")
print(f"  torch.export: {model.train_data.get_export_path()}")
print(f"  ONNX:         {model.train_data.get_onnx_path()}")
print(f"  ORT:          {model.train_data.get_ort_path()}")
print(f"  meta:         {model.train_data.get_meta_path()}")
print("Done!")
