# AGENTS.md

## Quick Start

- Python 3.12, package manager: **uv** (lockfile: `uv.lock`). Fallback: `pip install -r requirements.txt`
- Run web API (dev): `cd web && uvicorn app:app --reload`
- Run web API (prod): `docker compose up` (port 5001) or the Dockerfile targets port 5000
- Run CLI: `python main.py -c supreme_court -i captcha_data/<id>/images/<path>.png`

## Architecture

Single-repo PyTorch captcha solver. No package — all code lives at root as flat `.py` modules.

### Core Modules

| File | Description |
|------|-------------|
| `engine.py` | Entrypoints: `get_captcha_model()`, `train_model()`, `predict()`, `batch_predict_model()`, `redistribute_train_pred()` |
| `core.py` | `PyTorchModel` (CRNN build/train/eval), `FocalCTCLoss`, `LabelSmoothingCTCLoss`, transforms, dataset |
| `dataclass.py` | `TrainData`, `CaptchaType` (Pydantic models); paths, char sets, image preprocessing |
| `base_core.py` | `BaseModel` abstract interface |
| `web/app.py` | FastAPI + Uvicorn (`/health`, `/predict` endpoints), model cached in memory |
| `main.py` | CLI predictor, default `captcha_id = "supreme_court"` |
| `train.py` / `pred.py` | Thin wrappers around `engine.py` (edit hardcoded vars at top) |

### Supported Captcha Types

Hardcoded in `engine.get_captcha_type_list()`:

- `default` — digits only (`2345678bcdefgmnpwxy`), label length 5
- `dev` — mixed alphanumeric, label length 6
- `supreme_court` — custom crop preprocess, 120×40
- `gov24` — threshold=60, rev=1
- `wetax` — height=60
- `kshop` — 263×54

### Data Layout

```
captcha_data/<captcha_id>/<rev>/images/{train,pred}/
```

- Image **filename** (no extension) = label (e.g., `abc12.png` → label `abc12`)
- Models: `captcha_data/<id>/<rev>/model/model_full.pt` (state dict), `model_jit.pt` (TorchScript), `model_full.pt.onnx` (ONNX)

## Image Preprocessing

### Pipeline

1. **RGBA handling** — composite onto white background
2. **Grayscale** — `convert("L")`
3. **Threshold** — binary if `0 < threshold < 255`
4. **Border removal** — crop outer 2px margin
5. **Background white** — pixels > 128 become 255
6. **Resize** — to captcha-type-specific dimensions

### Training transforms (core.py:93-115)

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
- PyTorch cuDNN benchmarking is enabled globally in `core.py:20-23`. Never import `core.py` just to check imports — it triggers GPU setup.
- Loss types: `'ctc'` (default), `'focal'`, `'label_smoothing'`. Pass via `engine.predict()`.
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
- No lint/typecheck config (`setup.cfg` lists `py_modules` but no ruff, mypy, or flake8).
- PyInstaller spec exists (`main.spec`) but no build script visible.
- `web/app.py` imports `engine` directly (root-level, not a package).
- `dev.ipynb` exists but no execution commands documented.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **captcha-solver** (544 symbols, 851 relationships, 28 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/captcha-solver/context` | Codebase overview, check index freshness |
| `gitnexus://repo/captcha-solver/clusters` | All functional areas |
| `gitnexus://repo/captcha-solver/processes` | All execution flows |
| `gitnexus://repo/captcha-solver/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
