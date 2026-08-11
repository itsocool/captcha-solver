# AGENTS.md

## Quick Start

- Python 3.12 (`requires-python = "==3.12.*"`), package manager: **uv** (lockfile: `uv.lock`). 설치: `uv sync`
- Run web API (dev): `uv run fastapi dev apps/web/app.py --host 0.0.0.0 --port 8000`
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
| `hypercaptcha/engine.py` | Entrypoints: `get_captcha_model()`, `train_model()`, `predict()`, `batch_predict_model()`, `redistribute_train_pred()` |
| `hypercaptcha/core.py` | `PyTorchModel` (CRNN build/train/eval), `FocalCTCLoss`, transforms, dataset |
| `hypercaptcha/dataclass.py` | `TrainData`, `CaptchaType` (Pydantic models); paths, char sets, image preprocessing |
| `hypercaptcha/base_core.py` | `BaseModel` abstract interface |
| `hypercaptcha/cli.py` | CLI predictor. `hypercaptcha` 콘솔 스크립트 / `python -m hypercaptcha` |
| `apps/web/app.py` | FastAPI + Uvicorn (`/health`, `/predict` endpoints), model cached in memory |
| `hypercaptcha/train.py` / `hypercaptcha/pred.py` | Thin wrappers around `engine` (edit hardcoded vars at top). `python -m hypercaptcha.train` / `.pred` |

### Supported Captcha Types

Hardcoded in `engine.get_captcha_type_list()`:

- `supreme_court` — custom crop preprocess, 120×40
- `gov24` — threshold=60, rev=1
- `wetax` — height=60
- `kshop` — 263×54

### Data Layout

```
captcha_data/<captcha_id>/<rev>/images/{train,pred}/
```

- Image **filename** (no extension) = label (e.g., `abc12.png` → label `abc12`)
- Models: `captcha_data/<id>/<rev>/model/model.pth` (state dict checkpoint), `model.pt2` (`torch.export` archive), `model.onnx` (ONNX), `model.ort` (ORT format, baked from the ONNX at `ORT_ENABLE_EXTENDED` so it stays CPU-portable), `model.meta.json` (charset/size/preprocess sidecar, built by `CaptchaType.build_meta()`). `finalize_artifacts()` writes all of them from the finalized `.pth` on disk (never the in-memory model) and fails training if the checkpoint and the exported models disagree.

## Image Preprocessing

### Pipeline

1. **RGBA handling** — composite onto white background
2. **Grayscale** — `convert("L")`
3. **Threshold** — binary if `0 < threshold < 255`
4. **Border removal** — crop outer 2px margin
5. **Background white** — pixels > 128 become 255
6. **Resize** — to captcha-type-specific dimensions

### Training transforms

| Transform | Parameters |
|-----------|-----------|
| `RandomAffine` | rotation ±3°, translate 3%, scale 97–103%, shear 2° |
| `RandomGaussianBlur` | 30% probability |
| `ColorJitter` | brightness/contrast ±0.2, 20% probability |
| `RandomErasing` | 10% probability, scale 1–5% |

### Captcha-specific preprocess

- `supreme_court`: fixed ROI crop → paste into fixed canvas → grayscale → border remove → background white → resize

## Key Conventions

- Label length and character set are **auto-extracted** from training file names in `TrainData`.
- `hypercaptcha.train` / `hypercaptcha.pred` are **not argument-driven** — edit hardcoded vars at top of the module.
- PyTorch cuDNN benchmarking is enabled globally in `core.py`. Never import `core.py` just to check imports — it triggers GPU setup.
- Model architecture is CRNN only.
- Training loss types: `'ctc'`, `'focal'`.
- Loss type defaults to `'focal'` in `hypercaptcha/train.py` / `hypercaptcha/pred.py`.

## Image Size & CTC Constraints

- CRNN CNN outputs `W/16` time steps due to max-pooling (2×2, 2×2, 2×1 = 8×2 = 16 ratio).
- Constraint: `image_width / 16 >= label_length`.
- Example: 6-char captcha → minimum 96px width required.

## Web API

| Env var | Default | Description |
|---------|---------|-------------|
| `WEB_HOST` | `0.0.0.0` | Bind host |
| `WEB_PORT` | `5000` | Bind port |
| `WEB_DEBUG` | `false` | Debug mode |
| `DB_PATH` / `DB_SCHEMA_PATH` / `DB_SEED_PATH` | `./db/...` | SQLite 경로·스키마·시드 |

기본 캡차는 환경 변수가 아니라 `db/schema.sql` 의 `service_captchas.is_default` 가 정한다.

Endpoints: `GET /health`, `POST /predict`

### 연산 디바이스 선택

`POST /api/v1/predictImage` (form) 와 `POST /api/v1/predictJson` (body) 는 선택적
`device` 필드를 받는다: `auto`(기본) / `cpu` / `cuda`. 생략하면 auto 이고, auto 는
CUDA 가용 시 CUDA, 아니면 CPU 다 (`PyTorchModel` 의 원래 동작). 응답의 `device` 에
실제로 사용된 디바이스가 담긴다. 쓸 수 없는 디바이스를 요청하면 사유와 함께 400 이다.

판정 로직은 `apps/web/core/device.py` 한 곳에 있다. 모델 인스턴스는 특정 디바이스에
묶이므로 `services/captcha.py` 의 `_MODEL_CACHE` 키는 `(captcha_id, device)` 다 —
`captcha_id` 만으로 캐시를 뒤지는 코드를 새로 쓰지 말 것.

## Gotchas

- **No test suite** — verify by running `python -m hypercaptcha.train` or `.pred` with small dataset,
  또는 `apps/cli/tools/compare_with_python.py` 로 Rust CLI 와 대조.
- No lint/typecheck config (ruff, mypy, flake8 모두 없음). 패키징 설정은 `pyproject.toml` 한 곳에 있음.
- `apps/web/services/captcha.py`는 `from hypercaptcha import engine`을 **함수 안에서** 지연 import 한다.
  최상위에서 import 하면 서버 기동 시점에 torch/CUDA 초기화가 딸려온다.
- `web` 파이썬 패키지는 `apps/web/`에 있다 (`pyproject.toml`의 `[tool.setuptools.package-dir] web = "apps/web"`).
  import 이름은 여전히 `web.*` 이고, `fastapi dev apps/web/app.py`가 `apps/`를 sys.path에 넣는다.
