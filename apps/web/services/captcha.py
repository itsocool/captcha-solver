import base64
import binascii
import os
import tempfile
from datetime import datetime

from web.core.db import get_service_config
from web.core.device import resolve as resolve_device


# 모델 인스턴스는 특정 디바이스에 묶인다(가중치가 그 디바이스에 올라가 있다).
# 같은 캡차라도 cpu/cuda 는 별도 인스턴스여야 하므로 (captcha_id, device) 로 키를 잡는다.
_MODEL_CACHE: dict[tuple[str, str], object] = {}
_PRELOAD_STATUS: dict[str, str] = {}


class CaptchaPredictionError(Exception):
	pass


def ordered_captcha_ids(captcha_ids) -> list[str]:
	"""캡차 ID 들을 화면 표시 순서(captcha_types.seq)로 정렬한다.

	모든 셀렉트/목록(Home, Predict, Training, Data Source, Status)이 이 한 곳을 거쳐
	같은 순서를 보인다. DB 에 없는 ID 는 뒤로 가되 원래 순서(레지스트리 순)를 유지한다.
	"""
	from web.core.db import get_captcha_seq

	seq = get_captcha_seq()
	ids = list(captcha_ids)
	return sorted(ids, key=lambda cid: (seq.get(cid, 1 << 30), ids.index(cid)))


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

	return [(cid, registered[cid].name) for cid in ordered_captcha_ids(ids)]


def is_serviced(captcha_id: str) -> bool:
	return captcha_id in get_service_config()["serviced"]


def loaded_model_ids() -> list[str]:
	"""가중치까지 로드된 captcha_id 목록 (engine 임포트 없이 조회).

	같은 캡차가 여러 디바이스에 올라가 있어도 id 는 한 번만 센다.
	"""
	return sorted({
		captcha_id
		for (captcha_id, _), model in _MODEL_CACHE.items()
		if getattr(model, "model", None) is not None
	})


def loaded_devices(captcha_id: str) -> list[str]:
	"""해당 캡차가 실제로 올라가 있는 디바이스 목록."""
	return sorted({
		str(model.device)
		for (cached_id, _), model in _MODEL_CACHE.items()
		if cached_id == captcha_id and getattr(model, "model", None) is not None
	})


def get_model(captcha_id: str, device: str | None = None):
	"""캡차/디바이스 조합의 **가중치까지 올라간** 모델 인스턴스.

	device=None 이면 자동 선택. 쓸 수 없는 디바이스를 요청하면 ValueError 가 올라온다.

	로드에 실패한 인스턴스는 캐시에 남기지 않는다. 예전에는 인스턴스를 먼저 캐시에
	넣고 가중치는 첫 예측 때 지연 로드했는데, 로드가 실패해도 껍데기가 캐시에 남아
	다음 요청이 그걸 재사용했다 (첫 요청만 500, 이후로는 무학습 모델이 그럴듯한 답을
	돌려줌). "캐시에 있으면 곧 쓸 수 있는 모델"이라는 불변식을 여기서 지킨다.
	"""
	device_key = resolve_device(device)
	cache_key = (captcha_id, device_key)

	if cache_key in _MODEL_CACHE:
		return _MODEL_CACHE[cache_key]

	from hypercaptcha import engine

	model = engine.get_captcha_model(captcha_id=captcha_id, verbose=0, device=device_key)
	model.load_prediction_model()
	_MODEL_CACHE[cache_key] = model
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
			# get_model 이 로드까지 끝내고, 실패하면 아무것도 캐시하지 않는다.
			model = get_model(captcha_id)
		except Exception as e:
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
	registered = engine.get_captcha_type_list()
	for captcha_id in ordered_captcha_ids(registered):
		captcha_type = registered[captcha_id]
		train_data = captcha_type.train_data
		# 같은 캡차가 cpu/cuda 양쪽에 올라가 있을 수 있어 디바이스를 모아서 표시한다.
		devices = loaded_devices(captcha_id)
		loaded = bool(devices)
		model_path = train_data.get_model_path()
		file_stat = os.stat(model_path) if os.path.exists(model_path) else None

		rows.append({
			"captcha_id": captcha_id,
			"name": captcha_type.name,
			"desc": captcha_type.desc,
			"loaded": loaded,
			"serviced": captcha_id in serviced,
			"state": _PRELOAD_STATUS.get(captcha_id, "not serviced" if captcha_id not in serviced else "not loaded"),
			"device": ", ".join(devices) if devices else "-",
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


def predict_from_bytes(
	captcha_id: str,
	image_bytes: bytes,
	filename: str = "captcha.png",
	device: str | None = None,
) -> tuple[str, float, str]:
	"""단건 예측. (예측문자열, 신뢰도, 실제 사용한 디바이스) 를 돌려준다.

	device 는 'auto'/'cpu'/'cuda' 또는 None(=auto). 쓸 수 없는 값이면 ValueError.
	"""
	if not image_bytes:
		raise ValueError("image data is empty")

	if not is_serviced(captcha_id):
		raise ValueError(f"captcha_id '{captcha_id}' is not serviced")

	# 모델을 만들기 전에 검증한다. 잘못된 디바이스는 예측 실패(500)가 아니라 요청 오류(400)다.
	device_key = resolve_device(device)

	from hypercaptcha import engine

	safe_filename = os.path.basename(filename) or "captcha.png"
	with tempfile.TemporaryDirectory() as td:
		tmp_path = os.path.join(td, safe_filename)
		try:
			with open(tmp_path, "wb") as out_f:
				out_f.write(image_bytes)
			model = get_model(captcha_id, device_key)
			prediction, confidence = engine.predict(model=model, image_path=tmp_path, verbose=0)
			return prediction, confidence, str(model.device)
		except Exception as e:
			raise CaptchaPredictionError(str(e)) from e
