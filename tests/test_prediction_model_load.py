import json
from pathlib import Path

import pytest

from aso_ai.core import PyTorchModel
from web.core.dataclass import CaptchaType, TrainData


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


def test_sidecar_metadata_overrides_detected_training_info(captcha_data_dir):
	model = _model(captcha_data_dir("sample"))
	meta = {
		"image_width": 160,
		"image_height": 48,
		"label_length": 5,
		"characters": "abcde",
		"threshold": 90,
	}
	Path(model.train_data.get_meta_path()).write_text(json.dumps(meta))
	# 손상된 체크포인트 로드는 실패해도 입력 크기·문자셋은 먼저 sidecar를 적용한다.
	with pytest.raises(Exception) as failure:
		model.load_prediction_model()
	assert model.train_data.detected_image_width == 160, str(failure.value)
	assert model.train_data.detected_image_height == 48
	assert model.train_data.detected_label_length == 5
	assert model.train_data.detected_characters == "abcde"
	assert model.train_data.info.threshold == 90
	assert model.char_to_idx == {char: index + 1 for index, char in enumerate("abcde")}
