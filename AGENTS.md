# AGENTS.md

## Quick Start

- Python 3.12 (`requires-python = "==3.12.*"`), package manager: **uv** (lockfile: `uv.lock`). 설치: `uv sync`
- Run web API (dev): `./apps/web/server.sh start` (bash) 또는 `.\apps\web\server.ps1 start` (Windows)
  - 둘 다 `start|stop|restart|status|logs` 를 제공하고 상태 파일은 `apps/web/.dev/` 를 공유한다
  - ps1 옵션: `-Port` / `-BindHost` / `-NoReload` / `logs -Follow -Lines N`
  - 포그라운드로 직접 띄우려면 저장소 루트에서
    `uv run uvicorn web.app:app --host 0.0.0.0 --port 5000 --reload --reload-dir apps/web`
    (`--reload-dir` 없이 띄우면 `captcha_data` 수만 장을 감시하느라 리로드가 사실상 멈춘다)
- Run web API (prod): `docker compose up` (호스트 5001 → 컨테이너 8000)
- Run CLI: `uv run hypercaptcha -c supreme_court -i captcha_data/<id>/images/<path>.png`

## Architecture

Single-repo PyTorch captcha solver. 라이브러리 코드는 `packages/python_3.12/hyperCaptcha`
(배포명 `hypercaptcha`, uv workspace 멤버)에만 있다. 루트에 있던 플랫 모듈 사본은 제거됐고,
모든 소비자(`apps/web/`, 루트 스크립트, `apps/cli/tools/`)가 `from hypercaptcha import engine`으로 참조한다.
`uv sync` 하면 editable로 설치된다.

### Core Modules

라이브러리 모듈은 모두 `packages/python_3.12/hyperCaptcha/src/hypercaptcha/` 아래에 있다.

| File | Description |
|------|-------------|
| `hypercaptcha/engine.py` | Entrypoints: `get_captcha_type_list()`, `with_rev()`, `get_captcha_model()`, `train_model()`, `predict()`, `iter_batch_predict()`, `batch_predict_model()`, `redistribute_train_pred()` |
| `hypercaptcha/core.py` | `PyTorchModel` (CRNN build/train/eval/export), `CRNN`, `SpecAugment`, `FocalCTCLoss`, transforms, dataset, beam decoding |
| `hypercaptcha/dataclass.py` | `TrainData`, `CaptchaType` (Pydantic models); paths, char sets, image preprocessing (`default`/`supreme_court`/`iptime`) |
| `hypercaptcha/cli.py` | CLI predictor. `hypercaptcha` 콘솔 스크립트 / `python -m hypercaptcha` |
| `apps/web/app.py` | FastAPI 앱 조립: 프런트 라우터(Jinja2) + `/health`,`/version` + `/api/v1/*` (predict/batch/train/data-source). 모델은 `services/captcha.py` 의 `_MODEL_CACHE` 에 캐시 |
| `hypercaptcha/train.py` / `hypercaptcha/pred.py` | Thin wrappers around `engine` (edit hardcoded vars at top). `python -m hypercaptcha.train` / `.pred` |

상세는 `docs/` 참고 — 문서 맵과 갱신 규칙은 `CLAUDE.md` 에 있다.

### Supported Captcha Types

Hardcoded in `engine.get_captcha_type_list()` (4종):

- `supreme_court` — `preprocess="supreme_court"` (고정 ROI crop → 캔버스 paste), 120×40
- `gov24` — threshold=60
- `wetax` — height=60
- `iptime` — `preprocess="iptime"`, 원본 200×70 을 `crop=[27,10,195,70]` 으로 168×60 으로 자르기만 함. 유일하게 숫자가 아닌 캡차(소문자 5글자, `label_length=5`, `characters=LOWER_CASE`)

`iptime` 의 모델 입력 크기는 crop 결과에서 자동 감지된다 (`image_width`/`image_height` 는 crop 좌표계의 기준인 원본 크기).

### Data Layout

```
captcha_data/<captcha_id>/<rev>/images/{train,pred,draft}/
```

- Image **filename** (no extension) = label (e.g., `abc12.png` → label `abc12`). `draft/` 는 데이터 수집이 쌓은 원본 — 라벨 없으면 `draft-NNNNNN.png`, 라벨 붙으면 `<라벨>.png`(수동 또는 모델 예측으로 개명)
- 리비전은 **1부터 시작**한다 (`TrainData.rev` 기본값 1, DB `rev` DEFAULT 1, `captcha_data/<id>/1/` 이 첫 세대). 시드의 마이그레이션 8 이 옛 rev 0 DB 행을 1 로 옮긴다.
- Models: `captcha_data/<id>/<rev>/model/model.pth` (state dict checkpoint), `model.pt2` (`torch.export` archive), `model.onnx` (ONNX), `model.ort` (ORT format, baked from the ONNX at `ORT_ENABLE_EXTENDED` so it stays CPU-portable), `model.meta.json` (charset/size/preprocess sidecar, built by `CaptchaType.build_meta()`). `finalize_artifacts()` writes all of them from the finalized `.pth` on disk (never the in-memory model) and fails training if the checkpoint and the exported models disagree.

## Image Preprocessing

### Pipeline (`preprocess="default"`)

1. **RGBA handling** — composite onto white background
2. **Grayscale** — `convert("L")`
3. **Threshold** — `p > threshold` 인 픽셀만 255 로 (`0 < threshold < 255` 일 때만; 완전 이진화는 아님)
4. **Border removal** — crop outer 2px margin
5. **Background white** — pixels > 128 become 255
6. **Resize** — to `detected_image_width × detected_image_height`

### Training transforms (`core.get_train_transform`)

| Transform | Parameters |
|-----------|-----------|
| `RandomAffine` | rotation ±5°, translate 5%, scale 95–105%, shear 0–3°, fill=255 |
| `RandomPerspective` | distortion 0.1, p=0.3 |
| `RandomGrayscale` | p=0.1 inside `RandomApply(p=0.2)` |
| `GaussianBlur` | kernel 3, sigma 0.1–0.5, p=0.3 |
| `ColorJitter` | brightness/contrast 0.4, saturation 0.2, p=0.3 |
| `RandomErasing` | p=0.15, scale 1–5%, value=1.0 (흰색) |

추가로 CNN 출력 시퀀스에 `SpecAugment`(time/freq 마스킹)가 학습 시에만 적용된다.

### Captcha-specific preprocess

- `supreme_court`: fixed ROI crop → paste into fixed canvas → grayscale → border remove → background white → resize
- `iptime`: RGBA→흰 배경 → grayscale → 원본 크기로 맞춤 → `crop` 만. 배경이 이미 흰색

## Key Conventions

- Label length and character set are **auto-extracted** from training file names in `TrainData`.
- `hypercaptcha.train` / `hypercaptcha.pred` are **not argument-driven** — edit hardcoded vars at top of the module.
- PyTorch cuDNN benchmarking is enabled globally in `core.py`. Never import `core.py` just to check imports — it triggers GPU setup.
- Model architecture is CRNN only.
- Training loss: **`'focal'` 만 지원** (`FocalCTCLoss`). `train_model()` 은 그 외 `loss_type` 에 `ValueError` 를 던진다 (`core.py`). 표준 `'ctc'` 는 제거됐다.
- LR 스케줄: Linear Warmup → Cosine Annealing (`LambdaLR`). `ReduceLROnPlateau` 는 쓰지 않는다.

## Image Size & CTC Constraints

- CRNN CNN 은 MaxPool 2×2, 2×2, (2,1) 을 거쳐 `H/8 × W/4` 로 줄인다. **Time steps T = W/4** (전처리·crop 후 폭 기준).
- Constraint: `detected_image_width / 4 >= label_length`. 위반 시 `CRNN.__init__` 이 `ValueError`.
- Example: 6-char captcha → 최소 24px 폭. 실제 캡차(120~168px)에서는 여유가 크다.

## Web API

| Env var | Default | Description |
|---------|---------|-------------|
| `WEB_HOST` | `0.0.0.0` | Bind host |
| `WEB_PORT` | `5000` | Bind port |
| `WEB_DEBUG` | `false` | Debug mode |
| `WEB_CONTEXT_PATH` | (비어있음) | 리버스 프록시 하위 경로 접두사 (예: `/captcha`). FastAPI `root_path` + 템플릿/JS 링크에 붙는다 |
| `DB_PATH` / `DB_SCHEMA_PATH` / `DB_SEED_PATH` | `./db/...` | SQLite 경로·스키마·시드 |

기본 캡차는 환경 변수가 아니라 `db/schema.sql` 의 `service_captchas.is_default` 가 정한다.

Endpoints (전체 목록·스키마는 `docs/web-api-reference.md`):

- 시스템: `GET /health`, `GET /version`
- 추론: `POST /api/v1/predictImage` (multipart), `POST /api/v1/predictJson` (base64)
- 일괄추론: `GET /api/v1/batch/{targets,stream,image}`
- 학습: `GET /api/v1/train/{targets,params,stream}`, `POST /api/v1/train/{params,start,stop}`
- 데이터 수집: `GET /api/v1/data-source/{targets,stream,drafts,image}`, `POST /api/v1/data-source/label`
- 프런트(HTML): `/`, `/predict`, `/train`, `/data-source`, `/status`

### 연산 디바이스 선택

`POST /api/v1/predictImage` (form) 와 `POST /api/v1/predictJson` (body) 는 선택적
`device` 필드를 받는다: `auto`(기본) / `cpu` / `cuda`. 생략하면 auto 이고, auto 는
CUDA 가용 시 CUDA, 아니면 CPU 다 (`PyTorchModel` 의 원래 동작). 응답의 `device` 에
실제로 사용된 디바이스가 담긴다. 쓸 수 없는 디바이스를 요청하면 사유와 함께 400 이다.

판정 로직은 `apps/web/core/device.py` 한 곳에 있다. 모델 인스턴스는 특정 디바이스에
묶이므로 `services/captcha.py` 의 `_MODEL_CACHE` 키는 `(captcha_id, device)` 다 —
`captcha_id` 만으로 캐시를 뒤지는 코드를 새로 쓰지 말 것.

## Gotchas

- **테스트**: `tests/` 에 pytest 스위트가 있다 (`uv run pytest tests/`, 웹 서비스 캐시·컨텍스트 경로·모델 로드 위주).
  모델 품질은 테스트로 안 잡히니 `python -m hypercaptcha.train` / `.pred` 를 작은 데이터셋으로 돌리거나
  `apps/cli/tools/compare_with_python.py` 로 Rust CLI 와 대조해 확인한다.
- No lint/typecheck config (ruff, mypy, flake8 모두 없음). 패키징 설정은 `pyproject.toml` 한 곳에 있음.
- `apps/web/services/captcha.py`는 `from hypercaptcha import engine`을 **함수 안에서** 지연 import 한다.
  최상위에서 import 하면 서버 기동 시점에 torch/CUDA 초기화가 딸려온다.
- `web` 파이썬 패키지는 `apps/web/`에 있다 (`pyproject.toml`의 `[tool.setuptools.package-dir] web = "apps/web"`).
  import 이름은 여전히 `web.*` 이고, `fastapi dev apps/web/app.py`가 `apps/`를 sys.path에 넣는다.
