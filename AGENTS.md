# AGENTS.md

## Quick Start

- Python 3.12 (`requires-python = "==3.12.*"`), package manager: **uv** (lockfile: `uv.lock`). 설치: `uv sync`
- Run web API (dev): `uv run fastapi dev web/app.py --host 0.0.0.0 --port 8000`
- Run web API (prod): `docker compose up` (호스트 5001 → 컨테이너 8000)
- Run Spring Boot web API (dev): `apps/springBoot/devserver.sh {start|stop|restart|status}` (백그라운드, 포트 5000)
- Run CLI: `uv run python main.py -c supreme_court -i captcha_data/<id>/images/<path>.png`

## Architecture

Single-repo PyTorch captcha solver. No package — all code lives at root as flat `.py` modules.

### Core Modules

| File | Description |
|------|-------------|
| `engine.py` | Entrypoints: `get_captcha_model()`, `train_model()`, `predict()`, `batch_predict_model()`, `redistribute_train_pred()` |
| `core.py` | `PyTorchModel` (CRNN build/train/eval), `FocalCTCLoss`, transforms, dataset |
| `dataclass.py` | `TrainData`, `CaptchaType` (Pydantic models); paths, char sets, image preprocessing |
| `base_core.py` | `BaseModel` abstract interface |
| `web/app.py` | FastAPI + Uvicorn (`/health`, `/predict` endpoints), model cached in memory |
| `main.py` | CLI predictor, default `captcha_id = "supreme_court"` |
| `train.py` / `pred.py` | Thin wrappers around `engine.py` (edit hardcoded vars at top) |

### Supported Captcha Types

Hardcoded in `engine.get_captcha_type_list()`:

- `supreme_court` — custom crop preprocess, 120×40 (default `captcha_id`)
- `gov24` — threshold=60, rev=1
- `wetax` — height=60
- `kshop` — 263×54

### Data Layout

```
captcha_data/<captcha_id>/<rev>/images/{train,pred}/
```

- Image **filename** (no extension) = label (e.g., `abc12.png` → label `abc12`)
- Models: `captcha_data/<id>/<rev>/model/model.pt` (state dict), `model.onnx` (ONNX). `finalize_artifacts()` exports the ONNX from the finalized `.pt` on disk (never the in-memory model) and fails training if their predictions disagree.

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
- `train.py` and `pred.py` are **not argument-driven** — edit hardcoded vars at top.
- PyTorch cuDNN benchmarking is enabled globally in `core.py`. Never import `core.py` just to check imports — it triggers GPU setup.
- Model architecture is CRNN only.
- Training loss types: `'ctc'`, `'focal'`.
- Loss type defaults to `'focal'` in `train.py` / `pred.py`.

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
| `DEFAULT_CAPTCHA_ID` | `supreme_court` | Default captcha type |

Endpoints: `GET /health`, `POST /predict`

## Gotchas

- **No test suite** — verify by running `train.py` or `pred.py` with small dataset.
- No lint/typecheck config (ruff, mypy, flake8 모두 없음). 패키징 설정은 `pyproject.toml` 한 곳에 있음.
- PyInstaller spec exists (`main.spec`) but no build script visible.
- `web/app.py` imports `engine` directly (root-level, not a package).
- `dev.ipynb` exists but no execution commands documented.
