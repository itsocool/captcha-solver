"""추론에 사용할 연산 디바이스 선택.

torch 임포트는 반드시 함수 안에서 한다. 모듈을 임포트하는 것만으로 torch/CUDA
초기화가 딸려오면 서버 기동이 느려진다 (services/captcha.py 와 같은 이유).
"""

AUTO = "auto"
CPU = "cpu"
CUDA = "cuda"

VALID_DEVICES = (AUTO, CPU, CUDA)


def cuda_status() -> dict:
	"""CUDA 가용 여부와 그 사유.

	사용 불가일 때 원인이 두 가지로 갈리기 때문에 detail 을 따로 돌려준다.
	빌드에 CUDA 가 없는 경우와, CUDA 빌드인데 GPU 를 못 찾는 경우는 대처가 다르다.
	"""
	import torch

	built = torch.version.cuda
	available = torch.cuda.is_available()

	if available:
		names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
		detail = ", ".join(names)
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
	import torch

	key = (name or AUTO).strip().lower()

	if key not in VALID_DEVICES:
		raise ValueError(f"unsupported device: {name!r} (use one of {', '.join(VALID_DEVICES)})")

	if key == AUTO:
		return CUDA if torch.cuda.is_available() else CPU

	if key == CUDA and not torch.cuda.is_available():
		raise ValueError(f"CUDA 를 사용할 수 없습니다: {cuda_status()['detail']}")

	return key
