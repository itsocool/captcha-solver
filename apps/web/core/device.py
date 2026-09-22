"""추론에 사용할 연산 디바이스 선택.

torch 임포트는 반드시 함수 안에서 한다. 모듈을 임포트하는 것만으로 torch/CUDA
초기화가 딸려오면 서버 기동이 느려진다 (services/captcha.py 와 같은 이유).
"""

from functools import lru_cache

AUTO = "auto"
CPU = "cpu"
CUDA = "cuda"

VALID_DEVICES = (AUTO, CPU, CUDA)


@lru_cache(maxsize=1)
def _cuda_execution_error() -> str | None:
	"""GPU 감지만으로는 커널 호환성을 알 수 없으므로 프로세스당 한 번 실행한다."""
	import torch

	try:
		# item()까지 실행해 비동기 CUDA 오류도 이 경계에서 확인한다.
		torch.ones(1, device=CUDA).add_(1).item()
	except RuntimeError as error:
		return str(error).splitlines()[0]
	return None


def cuda_status() -> dict:
	"""CUDA 가용 여부와 그 사유.

	CPU 전용 빌드, GPU 미감지, 감지됐지만 커널을 실행할 수 없는 경우를 구분한다.
	"""
	import torch

	built = torch.version.cuda
	available = torch.cuda.is_available()

	if available:
		names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
		detail = ", ".join(names)
		if error := _cuda_execution_error():
			available = False
			detail = f"{detail} — 현재 PyTorch에서 CUDA 연산을 실행할 수 없습니다: {error}"
	elif built is None:
		detail = f"torch 가 CPU 전용 빌드입니다 ({torch.__version__})"
	else:
		detail = f"CUDA {built} 빌드이지만 사용 가능한 GPU 가 없습니다"

	return {"available": available, "built": built, "detail": detail}


def device_options() -> list[dict]:
	"""추론 페이지 셀렉터용 선택지."""
	status = cuda_status()
	resolved = CUDA if status["available"] else CPU

	return [
		{"value": AUTO, "label": f"자동 ({resolved})", "available": True, "detail": ""},
		{"value": CPU, "label": "CPU", "available": True, "detail": ""},
		{"value": CUDA, "label": "CUDA", "available": status["available"], "detail": status["detail"]},
	]


def resolve(name: str | None) -> str:
	"""요청된 디바이스 이름을 실제로 사용할 'cpu' 또는 'cuda' 로 확정한다.

	None/빈 값은 auto 로 취급한다 (device 를 넘기지 않던 기존 호출의 동작 유지).
	쓸 수 없는 값이면 ValueError 를 던진다 — 호출부에서 400 으로 바꾼다.
	"""
	key = (name or AUTO).strip().lower()

	if key not in VALID_DEVICES:
		raise ValueError(f"unsupported device: {name!r} (use one of {', '.join(VALID_DEVICES)})")

	if key == CPU:
		return CPU

	status = cuda_status()
	if key == AUTO:
		return CUDA if status["available"] else CPU

	if not status["available"]:
		raise ValueError(f"CUDA 를 사용할 수 없습니다: {status['detail']}")

	return key
