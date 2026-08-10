import base64
import binascii
import os
import tempfile
from datetime import datetime

from web.core.db import get_service_config


_MODEL_CACHE = {}
_PRELOAD_STATUS: dict[str, str] = {}


class CaptchaPredictionError(Exception):
	pass


def list_captcha_types(serviced_only: bool = True) -> list[tuple[str, str]]:
	"""(captcha_id, 표시명) 목록.

	기본은 DB(service_captchas)에서 서비스 대상으로 지정된 캡차만 반환한다.
	"""
	from hypercaptcha import engine

	registered = engine.get_captcha_type_list()
	if serviced_only:
		ids = [cid for cid in get_service_config()["serviced"] if cid in registered]
	else:
		ids = list(registered)

	return [(cid, registered[cid].name) for cid in ids]


def is_serviced(captcha_id: str) -> bool:
	return captcha_id in get_service_config()["serviced"]


def loaded_model_ids() -> list[str]:
	"""가중치까지 로드된 captcha_id 목록 (engine 임포트 없이 조회)."""
	return sorted(cid for cid, model in _MODEL_CACHE.items() if getattr(model, "model", None) is not None)


def get_model(captcha_id: str):
	if captcha_id in _MODEL_CACHE:
		return _MODEL_CACHE[captcha_id]

	from hypercaptcha import engine

	model = engine.get_captcha_model(captcha_id=captcha_id, verbose=0)
	_MODEL_CACHE[captcha_id] = model
	return model


def preload_models() -> dict[str, str]:
	"""서비스 대상 캡차 모델의 가중치를 미리 로드하고 워밍업한다.

	서버 기동 시 1회 호출. 학습된 모델이 없는 캡차는 건너뛴다(기동 실패 아님).
	반환: captcha_id -> 상태 문자열
	"""
	import torch

	status: dict[str, str] = {}

	for captcha_id, _ in list_captcha_types(serviced_only=True):
		try:
			model = get_model(captcha_id)
			model.load_prediction_model()
		except Exception as e:
			_MODEL_CACHE.pop(captcha_id, None)
			status[captcha_id] = f"skipped ({type(e).__name__}: {e})"
			continue

		# 워밍업: 첫 요청이 CUDA 커널 초기화 비용을 떠안지 않도록 더미 입력을 한 번 통과시킨다
		try:
			train_data = model.train_data
			dummy = torch.zeros(
				1, 1, train_data.detected_image_height, train_data.detected_image_width,
				device=model.device,
			)
			with torch.inference_mode():
				model.model(dummy)
			status[captcha_id] = "loaded"
		except Exception as e:
			status[captcha_id] = f"loaded (warmup skipped: {type(e).__name__})"

	_PRELOAD_STATUS.update(status)
	return status


def model_status() -> list[dict]:
	"""등록된 캡차별 모델 상태. /status 페이지용."""
	from hypercaptcha import engine

	serviced = get_service_config()["serviced"]
	rows = []
	for captcha_id, captcha_type in engine.get_captcha_type_list().items():
		train_data = captcha_type.train_data
		model = _MODEL_CACHE.get(captcha_id)
		loaded = model is not None and model.model is not None
		model_path = train_data.get_model_path()
		file_stat = os.stat(model_path) if os.path.exists(model_path) else None

		rows.append({
			"captcha_id": captcha_id,
			"name": captcha_type.name,
			"desc": captcha_type.desc,
			"loaded": loaded,
			"serviced": captcha_id in serviced,
			"state": _PRELOAD_STATUS.get(captcha_id, "not serviced" if captcha_id not in serviced else "not loaded"),
			"device": str(model.device) if model is not None else "-",
			"rev": train_data.rev,
			"image_size": f"{train_data.detected_image_width}×{train_data.detected_image_height}",
			"label_length": train_data.detected_label_length,
			"char_count": len(train_data.detected_characters),
			"model_path": model_path,
			"model_size_mb": round(file_stat.st_size / 1024 / 1024, 1) if file_stat else None,
			"model_mtime": (
				datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M") if file_stat else None
			),
		})

	return rows


def decode_image_data(image_data: str) -> bytes:
	if not isinstance(image_data, str) or not image_data.strip():
		raise ValueError("image_data is required")

	encoded = image_data.strip()
	if encoded.startswith("data:") and "," in encoded:
		encoded = encoded.split(",", 1)[1]

	try:
		return base64.b64decode(encoded, validate=True)
	except (binascii.Error, ValueError):
		raise ValueError("image_data must be valid base64")


def predict_from_bytes(captcha_id: str, image_bytes: bytes, filename: str = "captcha.png"):
	if not image_bytes:
		raise ValueError("image data is empty")

	if not is_serviced(captcha_id):
		raise ValueError(f"captcha_id '{captcha_id}' is not serviced")

	from hypercaptcha import engine

	safe_filename = os.path.basename(filename) or "captcha.png"
	with tempfile.TemporaryDirectory() as td:
		tmp_path = os.path.join(td, safe_filename)
		try:
			with open(tmp_path, "wb") as out_f:
				out_f.write(image_bytes)
			model = get_model(captcha_id)
			return engine.predict(model=model, image_path=tmp_path, verbose=0)
		except Exception as e:
			raise CaptchaPredictionError(str(e)) from e
