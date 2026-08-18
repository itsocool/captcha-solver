import pytest

from hypercaptcha.core import PyTorchModel
from hypercaptcha.dataclass import CaptchaType, TrainData


def _model(base_dir: str) -> PyTorchModel:
	captcha_type = CaptchaType(
		captcha_id="sample",
		name="sample",
		desc="테스트용 캡차",
		train_data=TrainData(captcha_id="sample", train_data_base_dir=base_dir),
	)
	return PyTorchModel(captcha_type=captcha_type, verbose=0)


def test_load_prediction_model_raises_when_checkpoint_is_unreadable(captcha_data_dir):
	model = _model(captcha_data_dir("sample"))

	with pytest.raises(Exception):
		model.load_prediction_model()


def test_failed_load_leaves_no_model_behind(captcha_data_dir):
	"""로드에 실패하면 self.model 은 None 으로 남아야 한다.

	빌드만 된 무학습 모델이 남으면 다음 호출이 그걸 학습된 모델로 착각해 쓴다.
	"""
	model = _model(captcha_data_dir("sample"))

	with pytest.raises(Exception):
		model.load_prediction_model()

	assert model.model is None
