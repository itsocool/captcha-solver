"""hypercaptcha — PyTorch CRNN 기반 캡챠 인식 라이브러리.

루트의 플랫 모듈(`dataclass.py`, `base_core.py`, `core.py`, `engine.py`, `main.py`)을
설치 가능한 패키지로 묶은 것입니다. 공개 API는 루트와 동일합니다.

    from hypercaptcha import engine

    model = engine.get_captcha_model(captcha_id="supreme_court")
    text, confidence = engine.predict(model=model, image_path="sample.png")
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _package_version
from typing import TYPE_CHECKING

try:
    __version__ = _package_version("hypercaptcha")
except PackageNotFoundError:  # 소스 트리에서 직접 import한 경우
    __version__ = "0.0.0.dev0"

if TYPE_CHECKING:  # pragma: no cover - 타입 체커 전용
    from . import engine
    from .base_core import BaseModel
    from .core import PyTorchModel
    from .dataclass import CaptchaType, TrainData

# core.py는 import 시점에 cuDNN benchmark/TF32 설정과 CUDA 프로브를 수행한다.
# 최상위 import만으로 GPU를 건드리지 않도록 실제 접근 시점까지 로딩을 미룬다.
_LAZY_ATTRS = {
    "engine": ("hypercaptcha.engine", None),
    "BaseModel": ("hypercaptcha.base_core", "BaseModel"),
    "CaptchaType": ("hypercaptcha.dataclass", "CaptchaType"),
    "TrainData": ("hypercaptcha.dataclass", "TrainData"),
    "PyTorchModel": ("hypercaptcha.core", "PyTorchModel"),
}

__all__ = ["__version__", *_LAZY_ATTRS]


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
