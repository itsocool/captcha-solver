"""학습(Training 페이지) 서비스.

services/batch_predict.py 와 같은 규칙이다. FastAPI 를 모르는 순수 파이썬이고,
평범한 dict 를 yield 하는 제너레이터라 서버 없이도 호출해서 확인할 수 있다.
hypercaptcha 임포트는 함수 안에서 한다.

일괄 추론과 다른 점 하나. engine.train_model() 은 제너레이터가 아니라 끝까지
돌아버리는 동기 호출이라 그대로는 진행률을 흘릴 수 없다. 그래서 학습을 워커
스레드에 넣고, core 의 on_event 콜백이 큐에 밀어넣은 이벤트를 이쪽에서 꺼내
yield 한다. 학습 루프 자체를 제너레이터로 뒤집는 것보다 훨씬 작은 변경이다.
"""

import threading
from pathlib import Path

from web.core.config import BASE_DIR
from web.core.device import resolve as resolve_device


# 학습은 GPU/CPU 를 통째로 붙잡는다. 서버 전체에서 한 번에 하나만 돌린다.
# 일괄 추론과 별개의 락이라 둘이 동시에 돌 수는 있는데, 그건 사용자가 감수할 몫이다.
_RUN_LOCK = threading.Lock()

# 진행 중인(또는 방금 끝난) 학습 세션. 학습이 한 번에 하나뿐이라 전역 하나면 된다.
# 이벤트를 버퍼에 쌓아 두므로, Training 페이지를 벗어났다 다시 열어도 히스토리를
# 재생하고 실시간으로 이어 볼 수 있다. SSE 연결과 학습 수명은 완전히 분리된다.
_SESSION: "_TrainSession | None" = None
_SESSION_GUARD = threading.Lock()

CAPTCHA_DATA_DIR = BASE_DIR / "captcha_data"

LOSS_TYPES = ("focal",)

# UI 가 넘길 수 있는 학습 파라미터와 허용 범위. 값은 train.py 의 기본값을 그대로 따른다.
# (min, max, default) — 범위를 벗어나면 400 으로 돌려보낸다.
#
# epochs 80 / patience 15 의 근거 (한 캡차를 조기종료 끄고 80에폭 측정):
#   epoch 2~28  val 3.89~3.94 정체. 입력과 무관하게 상수 문자열만 출력하는
#               CTC 초기 붕괴 구간이고, 이 안에서 무개선이 최대 7에폭 연속 발생한다.
#   epoch 29~43 탈출 후 0.05 까지 급강하
#   epoch 54    최적 0.0318 → 최종 99.01%
# 이전 기본값(40/6)은 정체 중 진동에 걸려 epoch 20 에서 죽었다. 15 는 관측된
# 최대 무개선 7 의 두 배이고, 수렴 뒤에는 무개선이 26 까지 쌓이므로 과학습
# 방지 기능은 그대로 남는다.
# ponytail: 정체 길이를 실측 한 번으로 잡은 값이다. 다른 캡차에서 더 긴
#           정체가 나오면 patience 를 또 올리는 대신, best 갱신 여부가 아니라
#           "최근 N에폭 개선폭 < 임계치" 로 정체를 판정하도록 고칠 것.
PARAM_SPEC = {
	"epochs": (1, 500, 80),
	"batch_size": (1, 512, 64),
	"early_stopping_patience": (0, 100, 15),
	"learning_rate": (1e-6, 1.0, 0.001),
	"warmup_epochs": (0, 100, 0),
	"train_ratio": (0.1, 0.95, 0.6),
}

# 대상(캡차, 리비전)별로 기억해 두는 파라미터. shuffle 은 디스크 이미지를 되돌릴 수 없게
# 옮기는 1회성 동작이라 뺀다 — 저장했다가 다음 실행에 자동 적용되면 매번 재분배되는 사고가 난다.
PERSIST_PARAMS = (*PARAM_SPEC.keys(), "loss_type", "use_amp")


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


def default_params() -> dict:
	"""저장된 값이 없을 때 쓰는 기본 파라미터 (PERSIST_PARAMS 만)."""
	params = {name: spec[2] for name, spec in PARAM_SPEC.items()}
	params["loss_type"] = LOSS_TYPES[0]
	params["use_amp"] = True
	return params


def load_params(captcha_id: str, rev: int) -> dict:
	"""대상의 저장된 학습 파라미터. 없으면 기본값. 저장 스키마가 바뀌어도 아는 키만 받는다."""
	from web.core.db import get_train_params

	params = default_params()
	stored = get_train_params(captcha_id, rev) or {}
	for key in PERSIST_PARAMS:
		if key in stored:
			params[key] = stored[key]
	return params


def _persist_params(captcha_id: str, rev: int, params: dict) -> None:
	"""확정된 파라미터에서 기억할 것만 골라 DB 에 저장한다."""
	from web.core.db import save_train_params

	save_train_params(captcha_id, rev, {k: params[k] for k in PERSIST_PARAMS if k in params})


def save_params(captcha_id: str, rev: int, raw: dict) -> dict:
	"""폼에서 온 파라미터를 검증·저장한다. 학습을 시작하지 않아도 대상별로 유지된다.

	학습 시작 때만 저장하면 '값만 바꾸고 페이지를 떠나면' 사라진다. 편집이 저장되도록
	이 경로를 따로 둔다. 잘못된 값은 clean_params 가 ValueError 로 튕긴다.
	"""
	params = clean_params(raw)
	_persist_params(captcha_id, rev, params)
	return params


class _TrainSession:
	"""진행 중 학습 하나의 상태. 이벤트를 버퍼에 쌓고 여러 구독자에게 재생/중계한다."""

	def __init__(self, captcha_id: str, rev: int):
		self.captcha_id = captcha_id
		self.rev = rev
		self.events: list[dict] = []
		self.done = threading.Event()
		self.cancel = threading.Event()
		self.discard = threading.Event()
		self._cond = threading.Condition()

	def emit(self, event: dict) -> None:
		with self._cond:
			self.events.append(event)
			self._cond.notify_all()

	def finish(self) -> None:
		self.done.set()
		with self._cond:
			self._cond.notify_all()

	def stream(self):
		"""버퍼에 쌓인 이벤트를 처음부터 재생하고, 끝날 때까지 새 이벤트를 흘린다.

		구독자마다 독립된 인덱스를 쓰므로 몇 개가 동시에 붙어도 되고, 이미 끝난
		세션에 붙으면 히스토리('start'~'done')를 전부 재생하고 바로 끝난다.
		"""
		idx = 0
		while True:
			with self._cond:
				while idx >= len(self.events) and not self.done.is_set():
					self._cond.wait(timeout=1.0)
				batch = self.events[idx:]
				idx = len(self.events)
				finished = self.done.is_set() and idx >= len(self.events)
			for event in batch:
				yield event
			if finished:
				return


def is_running() -> bool:
	return _RUN_LOCK.locked()


def current_session() -> "_TrainSession | None":
	"""진행 중이거나 방금 끝난 학습 세션. 없으면 None."""
	with _SESSION_GUARD:
		return _SESSION


def request_stop(save: bool = True) -> bool:
	"""실행 중인 학습에 중단을 요청한다. save=False 면 best 저장 없이 버린다.

	에폭 경계에서 걸리므로 진행 중인 에폭 하나는 마저 돈다. 돌고 있는 학습이
	없으면 False. (중단 취소는 요청을 안 보내면 되므로 API 가 없다.)
	"""
	session = current_session()
	if session is None or session.done.is_set():
		return False
	if not save:
		session.discard.set()
	session.cancel.set()
	return True


def start(captcha_id: str, rev: int, device: str | None = None, params: dict | None = None) -> None:
	"""백그라운드 학습을 시작한다. 진행 상황은 current_session().stream() 으로 본다.

	학습은 워커 스레드에서 돌고 SSE 연결과 수명이 분리된다 — 페이지를 벗어나도
	계속 돌고, 다시 열면 stream() 이 히스토리를 재생하며 이어 보여준다. 이벤트는
	세션 버퍼에 쌓이고, 중단은 request_stop() 이 세션 플래그로 건다.

	검증 오류/디바이스 오류는 요청 오류(ValueError/TrainError)로 올라오고,
	이미 학습이 돌고 있으면 TrainBusy.
	"""
	target = find_target(captcha_id, rev)
	if not target["selectable"]:
		raise TrainError(target["reason"] or "학습할 수 없는 대상입니다")

	# 모델을 만들기 전에 검증한다. 잘못된 디바이스/파라미터는 실행 오류가 아니라 요청 오류다.
	device_key = resolve_device(device)
	params = clean_params(params or {})

	# 이번에 쓴 값을 대상별로 기억한다. 다음에 같은 대상을 고르면 폼에 다시 채워진다.
	_persist_params(captcha_id, rev, params)

	if not _RUN_LOCK.acquire(blocking=False):
		raise TrainBusy("이미 다른 학습이 실행 중입니다")

	session = _TrainSession(captcha_id, rev)
	global _SESSION
	with _SESSION_GUARD:
		_SESSION = session

	def on_event(event: dict):
		session.emit(event)
		if not session.cancel.is_set():
			return True
		# 'discard' 는 core 가 best(.tmp)를 버리고 멈추라는 신호로 읽는다.
		return "discard" if session.discard.is_set() else False

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
				session.emit({"type": "shuffle", **moved})
				# 재분배 뒤에는 감지 정보(장수)가 바뀌므로 모델을 다시 만든다.
				model = engine.get_captcha_model(
					captcha_id=captcha_id, verbose=1, device=device_key, rev=rev,
				)

			# 학습 시작 시점의 학습 정보를 DB 에 반영한다. 라벨을 고치면 감지 문자셋이
			# 바뀌는데 기록은 낡은 채 남아서, 실제로 학습에 쓰는 값으로 맞춘다.
			# (모델 sidecar meta.json 은 여기서 쓰지 않는다 — 학습이 discard/중단되면
			#  meta 만 새 값이 되고 model.pth 는 옛 것이라 추론이 깨진다. finalize 에서
			#  모델과 함께 쓴다.)
			td = model.train_data
			from web.core.db import save_train_config
			save_train_config(captcha_id, rev, {
				"image_width": td.image_width,
				"image_height": td.image_height,
				"label_length": td.detected_label_length,
				"characters": td.detected_characters,
				"threshold": td.threshold,
				"preprocess": td.preprocess,
				"crop": td.crop,
			})

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
			session.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
		finally:
			# 세션(_SESSION)은 일부러 비우지 않는다 — 끝난 뒤에도 다시 열면 마지막
			# 결과를 재생해 보여줄 수 있게. 락만 풀어 다음 학습을 받을 수 있게 한다.
			session.finish()
			_RUN_LOCK.release()

	try:
		threading.Thread(target=worker, name="captcha-train", daemon=True).start()
	except BaseException:
		# 워커가 뜨지 못하면 아무도 락을 풀어주지 않는다. 세션도 끝으로 표시해
		# 구독자가 무한 대기하지 않게 한다.
		session.emit({"type": "error", "message": "학습 스레드를 시작하지 못했습니다"})
		session.finish()
		_RUN_LOCK.release()
		raise
