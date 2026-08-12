"""hypercaptcha — PyTorch CRNN 기반 캡챠 인식 라이브러리.

    from hypercaptcha import engine

    model = engine.get_captcha_model(captcha_id="supreme_court")
    text, confidence = engine.predict(model=model, image_path="sample.png")

`core.py` 는 import 시점에 cuDNN benchmark/TF32 설정과 CUDA 프로브를 수행한다.
여기서 서브모듈을 재노출하지 않는 것이 그 로딩을 실제 사용 시점까지 미루는 방법이다
(`from hypercaptcha import engine` 은 파이썬이 서브모듈 import 로 처리한다).
"""

from importlib.metadata import PackageNotFoundError, version as _package_version

try:
    __version__ = _package_version("hypercaptcha")
except PackageNotFoundError:  # 소스 트리에서 직접 import한 경우
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
