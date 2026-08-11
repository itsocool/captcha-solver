"""학습(Training 페이지) 서비스.

services/batch_predict.py 와 같은 규칙이다. FastAPI 를 모르는 순수 파이썬이고,
평범한 dict 를 yield 하는 제너레이터라 서버 없이도 호출해서 확인할 수 있다.
hypercaptcha 임포트는 함수 안에서 한다.

일괄 추론과 다른 점 하나. engine.train_model() 은 제너레이터가 아니라 끝까지
돌아버리는 동기 호출이라 그대로는 진행률을 흘릴 수 없다. 그래서 학습을 워커
스레드에 넣고, core 의 on_event 콜백이 큐에 밀어넣은 이벤트를 이쪽에서 꺼내
yield 한다. 학습 루프 자체를 제너레이터로 뒤집는 것보다 훨씬 작은 변경이다.
"""

import queue
import threading
from pathlib import Path

from web.core.config import BASE_DIR
from web.core.device import resolve as resolve_device


# 학습은 GPU/CPU 를 통째로 붙잡는다. 서버 전체에서 한 번에 하나만 돌린다.
# 일괄 추론과 별개의 락이라 둘이 동시에 돌 수는 있는데, 그건 사용자가 감수할 몫이다.
_RUN_LOCK = threading.Lock()

CAPTCHA_DATA_DIR = BASE_DIR / "captcha_data"

LOSS_TYPES = ("focal", "ctc")

# UI 가 넘길 수 있는 학습 파라미터와 허용 범위. 값은 train.py 의 기본값을 그대로 따른다.
# (min, max, default) — 범위를 벗어나면 400 으로 돌려보낸다.
PARAM_SPEC = {
	"epochs": (1, 500, 40),
	"batch_size": (1, 512, 64),
	"early_stopping_patience": (0, 100, 6),
	"learning_rate": (1e-6, 1.0, 0.001),
	"warmup_epochs": (0, 100, 0),
	"train_ratio": (0.1, 0.95, 0.6),
}


class TrainBusy(Exception):
	"""이미 다른 학습이 돌고 있다."""


class TrainError(Exception):
	pass


def _rev_dir(captcha_id: str, rev: int) -> Path:
	return CAPTCHA_DATA_DIR / captcha_id / str(rev)


def list_targets() -> list[dict]:
	"""학습 가능한 (캡차, 리비전) 조합.

	batch_predict.list_targets() 와 판정 기준이 다르다. 저쪽은 '학습된 모델이 있는가'
	를 보지만 학습은 모델을 만드는 쪽이라 'images/train 에 이미지가 있는가' 를 본다.
	기존 모델이 있으면 has_model 로 알려준다 — 덮어쓰기 확인 모달의 근거다.
	"""
	from hypercaptcha import engine

	registered = engine.get_captcha_type_list()
	targets: list[dict] = []

	for captcha_id in sorted(registered):
		captcha_dir = CAPTCHA_DATA_DIR / captcha_id
		if not captcha_dir.is_dir():
			continue

		for rev_dir in sorted(captcha_dir.iterdir(), key=lambda p: p.name):
			if not rev_dir.is_dir() or not rev_dir.name.isdigit():
				continue

			rev = int(rev_dir.name)
			train_dir = rev_dir / "images" / "train"
			pred_dir = rev_dir / "images" / "pred"
			train_count = len(list(train_dir.glob("*.png"))) if train_dir.is_dir() else 0
			pred_count = len(list(pred_dir.glob("*.png"))) if pred_dir.is_dir() else 0
			has_model = (rev_dir / "model" / "model.pth").is_file()

			targets.append({
				"captcha_id": captcha_id,
				"name": registered[captcha_id].name,
				"rev": rev,
				"train_count": train_count,
				"pred_count": pred_count,
				"has_model": has_model,
				"selectable": train_count > 0,
				"reason": "" if train_count > 0 else "images/train 에 이미지가 없습니다",
			})

	return targets


def find_target(captcha_id: str, rev: int) -> dict:
	for target in list_targets():
		if target["captcha_id"] == captcha_id and target["rev"] == rev:
			return target
	raise ValueError(f"unknown target: captcha_id={captcha_id!r} rev={rev}")


def clean_params(raw: dict) -> dict:
	"""요청으로 들어온 학습 파라미터를 검증해 확정한다.

	쿼리스트링은 전부 문자열로 오고 사용자가 직접 만질 수 있는 값이라
	여기서 형 변환과 범위 확인을 한 번에 끝낸다.
	"""
	params: dict = {}

	for name, (low, high, default) in PARAM_SPEC.items():
		value = raw.get(name)
		if value in (None, ""):
			params[name] = default
			continue
		try:
			value = float(value) if isinstance(default, float) else int(value)
		except (TypeError, ValueError):
			raise ValueError(f"{name}: 숫자가 아닙니다 ({value!r})")
		if not (low <= value <= high):
			raise ValueError(f"{name}: {low} ~ {high} 범위여야 합니다 (받은 값 {value})")
		params[name] = value

	loss_type = (raw.get("loss_type") or "focal").strip().lower()
	if loss_type not in LOSS_TYPES:
		raise ValueError(f"loss_type: {', '.join(LOSS_TYPES)} 중 하나여야 합니다 (받은 값 {loss_type!r})")
	params["loss_type"] = loss_type

	params["use_amp"] = _as_bool(raw.get("use_amp"), True)
	params["shuffle"] = _as_bool(raw.get("shuffle"), False)

	return params


def _as_bool(value, default: bool) -> bool:
	if value in (None, ""):
		return default
	if isinstance(value, bool):
		return value
	return str(value).strip().lower() in ("1", "true", "yes", "on")


def is_running() -> bool:
	return _RUN_LOCK.locked()


def run(captcha_id: str, rev: int, device: str | None = None, params: dict | None = None):
	"""학습 이벤트 제너레이터.

	core.PyTorchModel.train_model() 의 on_event 콜백이 큐로 밀어넣은 이벤트를
	그대로 흘려보낸다. 소비자가 중간에 끊으면(GeneratorExit) 취소 플래그를 세워
	워커가 다음 에폭 경계에서 스스로 빠져나온다. 에폭 도중에는 못 멈춘다 —
	배치 추론과 달리 학습은 중간에 죽이면 체크포인트가 갈리기 때문이다.
	"""
	target = find_target(captcha_id, rev)
	if not target["selectable"]:
		raise TrainError(target["reason"] or "학습할 수 없는 대상입니다")

	# 모델을 만들기 전에 검증한다. 잘못된 디바이스/파라미터는 실행 오류가 아니라 요청 오류다.
	device_key = resolve_device(device)
	params = clean_params(params or {})

	if not _RUN_LOCK.acquire(blocking=False):
		raise TrainBusy("이미 다른 학습이 실행 중입니다")

	events: queue.Queue = queue.Queue()
	cancelled = threading.Event()
	DONE = object()

	def on_event(event: dict) -> bool:
		events.put(event)
		return not cancelled.is_set()

	def worker():
		try:
			from hypercaptcha import engine

			# 서빙 캐시(_MODEL_CACHE)와 섞이지 않는 별도 인스턴스.
			model = engine.get_captcha_model(
				captcha_id=captcha_id, verbose=1, device=device_key, rev=rev,
			)

			if params["shuffle"]:
				# train/pred 이미지를 디스크에서 재분배한다. 되돌릴 수 없다.
				import os

				image_dir = os.path.dirname(model.train_data.get_image_dir())
				moved = engine.redistribute_train_pred(
					image_dir=image_dir, train_ratio=params["train_ratio"], verbose=False,
				)
				events.put({"type": "shuffle", **moved})
				# 재분배 뒤에는 감지 정보(장수)가 바뀌므로 모델을 다시 만든다.
				model = engine.get_captcha_model(
					captcha_id=captcha_id, verbose=1, device=device_key, rev=rev,
				)

			engine.train_model(
				model=model,
				epochs=params["epochs"],
				batch_size=params["batch_size"],
				early_stopping_patience=params["early_stopping_patience"],
				learning_rate=params["learning_rate"],
				warmup_epochs=params["warmup_epochs"],
				loss_type=params["loss_type"],
				use_amp=params["use_amp"],
				on_event=on_event,
			)
		except Exception as e:
			events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
		finally:
			events.put(DONE)

	thread = threading.Thread(target=worker, name="captcha-train", daemon=True)
	thread.start()

	try:
		while True:
			event = events.get()
			if event is DONE:
				break
			yield event
	finally:
		# 소비자가 끊었거나 정상 종료했거나. 어느 쪽이든 워커가 끝나야 락을 놓는다.
		cancelled.set()
		thread.join()
		_RUN_LOCK.release()
