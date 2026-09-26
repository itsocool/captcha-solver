# CAPTCHA FastAPI application

Python 3.13 이상이 필요합니다. 실제 `web` 패키지는 이 디렉터리에 있으며,
`app.py`, `api/`, `core/`, `frontend/`, `schemas/`, `services/`와 템플릿·정적 파일을
함께 빌드합니다. 별도의 `src/web` 디렉터리는 사용하지 않습니다.

## 설치 및 실행

저장소 루트에서:

```bash
uv sync --project apps/web --locked
uv run --project apps/web web
# 개발 시 앱 코드만 감시
uv run --project apps/web uvicorn web.app:app --host 0.0.0.0 --port 5000 --reload --reload-dir apps/web
# 기존 백그라운드 서버 관리
./apps/web/server.sh start
```

이 디렉터리에서는 `uv sync --locked` / `uv run web`을 사용합니다.
FastAPI CLI도 `uv run fastapi run`으로 실행할 수 있습니다 (`web.app:app`).
`WEB_HOST`, `WEB_PORT`, `WEB_DEBUG`는 `web` / `python -m web` 실행에 적용됩니다.
`.venv`와 `uv.lock`은 `apps/web` 안에 생성됩니다.

Linux/Windows에서는 GTX 1050 등 Pascal GPU를 지원하는 PyTorch 2.14.0과
torchvision 0.29.0의 CUDA 12.6 빌드를 설치하도록 인덱스와 lockfile을 고정했습니다.
macOS 등에서는 PyPI 빌드를 사용합니다. GPU를 바꾸거나 PyTorch를 업그레이드할 때는
[PyTorch CUDA 지원 표](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)를 먼저 확인하세요.
의존성을 교체한 뒤에는 `./apps/web/server.sh restart`로 프로세스와 CUDA 상태 캐시를
갱신해야 합니다. `nvidia-smi`의 CUDA 버전은 드라이버가 지원하는 버전이며,
PyTorch가 포함하는 CUDA 런타임 버전과 같을 필요는 없습니다.

연산 디바이스 `auto`는 CUDA 커널 실행까지 확인합니다. GPU가 감지돼도 현재
PyTorch 빌드에서 실행할 수 없으면 CPU를 사용하며, 명시적 `cuda` 요청은 사유와
함께 400으로 응답합니다.

`aso-ai`는 `../../packages/python_3.13`의 editable 의존성입니다.
웹 서비스에서는 `from web.core import engine`을 함수 내부에서 지연 import합니다.
`core/engine.py`는 `aso_ai.core`의 모델을 사용하고, `core/dataclass.py`는 캡차 설정과
전처리를 소유합니다. `aso-ai` CLI 명령도 이 웹 프로젝트가 등록합니다.

## 빌드 및 테스트

저장소 루트에서:

```bash
uv build --project packages/python_3.13
uv build --project apps/web
uv run --project apps/web pytest tests/
```

각 프로젝트의 `dist/`에 wheel과 sdist가 생성됩니다. wheel 배포 시
`aso_ai-*.whl`과 `web-*.whl`을 함께 설치해야 합니다 (로컬 경로 의존성은 uv 개발 설정).

```bash
uv venv --python 3.13 /path/to/venv
uv pip install --python /path/to/venv/bin/python packages/python_3.13/dist/*.whl apps/web/dist/*.whl
WEB_DATA_DIR=/path/to/data /path/to/venv/bin/python -m web
```

## 런타임 데이터

소스/editable 설치는 저장소 루트, wheel 설치는 현재 작업 디렉터리가 기본 데이터
경로입니다. `WEB_DATA_DIR` 환경 변수로 명시할 수도 있습니다 (앱 import 전에 지정).
이 경로에서 `.env`, `db/`, `captcha_data/`를 찾습니다. HTML·JS·favicon은 설치된
패키지에 포함되며 데이터 경로와 무관합니다.

DB 스키마와 모델은 wheel에 포함하지 않습니다. 배포 시 저장소의 `db/schema.sql`,
`db/seed_captcha_types.sql`을 데이터 경로의 `db/`에 복사하고, 사용할 모델·이미지를
`captcha_data/`에 배치하세요. `DB_PATH`, `DB_SCHEMA_PATH`, `DB_SEED_PATH`로
각각 절대경로도 지정할 수 있습니다.

`/ping`, `/version`, `/health`, `/docs`로 기동을 확인합니다. 모델이 없거나 손상되면
서버는 기동하지만 `/health`는 `degraded`를 반환합니다.
