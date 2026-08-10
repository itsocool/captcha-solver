# hypercaptcha

PyTorch CRNN 기반 캡챠 인식 라이브러리 및 CLI. 저장소의 파이썬 라이브러리 코드는
전부 이 패키지에 있습니다 (예전에 루트에 있던 플랫 모듈 사본은 제거됨).

## 설치

```bash
# 워크스페이스 내 개발 설치
uv pip install -e packages/python_3.12/hyperCaptcha

# 또는 순수 pip
pip install -e packages/python_3.12/hyperCaptcha
```

이 패키지는 루트 프로젝트의 uv workspace 멤버이고 루트 `pyproject.toml`이
`hypercaptcha`를 의존성으로 선언하므로 (`[tool.uv.sources] hypercaptcha = { workspace = true }`),
저장소에서는 `uv sync`만 하면 editable로 설치됩니다.

PyTorch CUDA 빌드를 쓰려면 별도 인덱스가 필요합니다. 루트 `pyproject.toml`의
`[[tool.uv.index]]` (`https://download.pytorch.org/whl/cu130`) 설정이 워크스페이스
멤버에도 적용됩니다.

## 라이브러리 사용

```python
from hypercaptcha import engine

model = engine.get_captcha_model(
    train_data_base_dir="./captcha_data",
    captcha_id="supreme_court",
)
text, confidence = engine.predict(model=model, image_path="sample.png")
```

최상위 `import hypercaptcha`는 torch를 로드하지 않습니다. `engine`,
`PyTorchModel` 등 실제 속성에 접근하는 시점에 지연 로딩되며, 이때
`core.py`의 cuDNN benchmark / TF32 설정과 CUDA 프로브가 실행됩니다.

### 공개 심볼

| 심볼 | 원본 모듈 |
|------|-----------|
| `hypercaptcha.engine` | `engine.py` |
| `hypercaptcha.PyTorchModel` | `core.py` |
| `hypercaptcha.BaseModel` | `base_core.py` |
| `hypercaptcha.CaptchaType`, `hypercaptcha.TrainData` | `dataclass.py` |

등록된 캡차 ID는 `supreme_court`, `gov24`, `wetax`, `kshop` 네 가지입니다
(`engine.get_captcha_type_list()`).

## CLI

```bash
hypercaptcha -c supreme_court -i path/to/image.png
hypercaptcha -c supreme_court -i path/to/image.png -v   # JSON 출력 + 로깅
python -m hypercaptcha -c supreme_court -i path/to/image.png
```

기본 출력은 예측 문자열만 개행 없이 출력합니다. `-v`를 주면
`predicted_text` / `confidence` / `execution_time`을 JSON으로 출력하고
`./logs/main.log`에 기록합니다.

## 학습 / 배치 평가 스크립트

`train.py`와 `pred.py`는 인자를 받지 않습니다. 모듈 상단의 `captcha_id`,
`epochs`, `batch_size` 등을 고친 뒤 실행하세요.

```bash
python -m hypercaptcha.train   # 학습, PT/JIT/ONNX 산출물 생성
python -m hypercaptcha.pred    # images/pred 배치 평가
```

## 데이터 경로

CLI와 스크립트는 **현재 작업 디렉토리** 기준으로 `./captcha_data`와 `./logs`를
찾습니다 (설치된 패키지는 site-packages에 놓이므로 모듈 위치를 기준으로 삼을 수
없습니다). 라이브러리로 쓸 때는 `engine.get_captcha_model(train_data_base_dir=...)`로
명시하세요.

데이터 레이아웃은 루트 `AGENTS.md`와 동일합니다:

```
captcha_data/<captcha_id>/<rev>/images/{train,pred}/
```

## 소비자

| 사용처 | 참조 방식 |
|--------|-----------|
| `apps/web/` (FastAPI) | `from hypercaptcha import engine` |
| `apps/cli/tools/*.py` | `from hypercaptcha import engine` |
| `test_ctc_decode.py` | `from hypercaptcha.core import ...` |
