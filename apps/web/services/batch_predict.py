"""일괄 추론(Predict 페이지) 서비스.

FastAPI 를 모르는 순수 파이썬이다. 평범한 dict 를 yield 하는 제너레이터라
서버 없이도 호출해서 확인할 수 있다.

hypercaptcha 임포트는 함수 안에서 한다 (services/captcha.py 와 같은 이유).
"""

import os
import threading
from pathlib import Path

from web.core.config import BASE_DIR
from web.core.device import resolve as resolve_device


# 검증은 CPU 를 길게 물고 있어서 서빙과 경합한다. 서버 전체에서 한 번에 하나만 돌린다.
_RUN_LOCK = threading.Lock()

CAPTCHA_DATA_DIR = BASE_DIR / "captcha_data"


class BatchPredictBusy(Exception):
	"""이미 다른 일괄 추론이 돌고 있다."""


class BatchPredictError(Exception):
	pass


def _rev_dir(captcha_id: str, rev: int) -> Path:
	return CAPTCHA_DATA_DIR / captcha_id / str(rev)


def list_targets() -> list[dict]:
	"""디스크를 훑어 (캡차, 리비전) 조합을 나열한다.

	DB(train_data_configs)는 캡차당 리비전을 하나만 들고 있어서 목록의 근거로 못 쓴다.
	실제로 학습 결과가 놓이는 곳은 파일시스템이므로 그쪽을 진실로 삼는다.
	모델이 없는 리비전도 이유를 보여주려고 목록에는 포함하되 selectable=False 로 둔다.
	"""
	from hypercaptcha import engine
	from web.services.captcha import ordered_captcha_ids

	registered = engine.get_captcha_type_list()
	targets: list[dict] = []

	for captcha_id in ordered_captcha_ids(registered):
		captcha_dir = CAPTCHA_DATA_DIR / captcha_id
		if not captcha_dir.is_dir():
			continue

		for rev_dir in sorted(captcha_dir.iterdir(), key=lambda p: p.name):
			if not rev_dir.is_dir() or not rev_dir.name.isdigit():
				continue

			rev = int(rev_dir.name)
			model_path = rev_dir / "model" / "model.pth"
			pred_dir = rev_dir / "images" / "pred"
			pred_count = len(list(pred_dir.glob("*.png"))) if pred_dir.is_dir() else 0

			has_model = model_path.is_file()
			if not has_model:
				reason = "학습된 model.pth 가 없습니다"
			elif pred_count == 0:
				reason = "images/pred 에 이미지가 없습니다"
			else:
				reason = ""

			targets.append({
				"captcha_id": captcha_id,
				"name": registered[captcha_id].name,
				"rev": rev,
				"pred_count": pred_count,
				"has_model": has_model,
				"selectable": has_model and pred_count > 0,
				"reason": reason,
			})

	return targets


def find_target(captcha_id: str, rev: int) -> dict:
	for target in list_targets():
		if target["captcha_id"] == captcha_id and target["rev"] == rev:
			return target
	raise ValueError(f"unknown target: captcha_id={captcha_id!r} rev={rev}")


def pred_image_path(captcha_id: str, rev: int, image_name: str) -> Path:
	"""썸네일로 내보낼 pred 이미지의 실제 경로.

	image_name 은 클라이언트가 준 값이라 그대로 믿으면 경로 탈출이 된다.
	basename 만 취하고, 확장자를 png 로 제한하고, 최종 경로가 pred 디렉터리
	안에 있는지 한 번 더 확인한다.
	"""
	safe_name = os.path.basename(image_name)
	if not safe_name or not safe_name.lower().endswith(".png"):
		raise ValueError("png 파일명만 허용합니다")

	pred_dir = (_rev_dir(captcha_id, rev) / "images" / "pred").resolve()
	path = (pred_dir / safe_name).resolve()

	if not path.is_relative_to(pred_dir):
		raise ValueError("허용되지 않은 경로입니다")
	if not path.is_file():
		raise ValueError("이미지를 찾을 수 없습니다")

	return path


def is_running() -> bool:
	return _RUN_LOCK.locked()


def run(captcha_id: str, rev: int, device: str | None = None):
	"""일괄 추론 이벤트 제너레이터.

	engine.iter_batch_predict() 이벤트를 그대로 흘려보낸다. 락은 제너레이터가
	끝나거나 소비자가 중간에 버려도(GeneratorExit) 풀리도록 finally 에서 해제한다.
	"""
	target = find_target(captcha_id, rev)
	if not target["selectable"]:
		raise BatchPredictError(target["reason"] or "실행할 수 없는 대상입니다")

	# 모델을 만들기 전에 검증한다. 잘못된 디바이스는 실행 오류가 아니라 요청 오류다.
	device_key = resolve_device(device)

	if not _RUN_LOCK.acquire(blocking=False):
		raise BatchPredictBusy("이미 다른 일괄 추론이 실행 중입니다")

	try:
		from hypercaptcha import engine

		# 서빙 캐시(_MODEL_CACHE)를 건드리지 않는 별도 인스턴스를 쓴다.
		# iter_batch_predict 가 loss_type/use_amp 를 직접 바꾸기 때문이다.
		model = engine.get_captcha_model(
			captcha_id=captcha_id,
			verbose=0,
			device=device_key,
			rev=rev,
		)
		yield from engine.iter_batch_predict(model=model)
	finally:
		_RUN_LOCK.release()
