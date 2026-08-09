# README Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `README.md` as an accurate, navigable entry point for first-time users, model developers, and deployment operators.

**Architecture:** Keep one concise root README that routes readers by runtime and responsibility. Preserve detailed internals in `docs/crnn_ctc.md` and `docs/codebase-analysis.md`, while the README owns installation, runnable commands, shared contracts, operational warnings, and links.

**Tech Stack:** Markdown, Python 3.12/uv, PyTorch, FastAPI, Rust/Cargo/ONNX Runtime, Java 25/Spring Boot/Maven, SQLite, Docker Compose

## Global Constraints

- Modify only `README.md`; do not change code, APIs, configuration, model assets, or databases.
- Cover first-time users, model developers, and deployment operators with equal priority.
- Treat the root Python implementation as the training and PyTorch inference source of truth.
- Describe Rust and Spring Boot as ONNX consumers requiring a synchronized `.onnx` plus `.meta.json` sidecar.
- State the current CNN time-axis reduction as `W/4`, not `W/16`.
- Link detailed internals to `docs/crnn_ctc.md` and `docs/codebase-analysis.md` instead of duplicating them.
- Preserve existing user changes outside `README.md`.

---

### Task 1: Rebuild the README information architecture

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: commands and contracts from `pyproject.toml`, `engine.py`, `web/`, `apps/cli/README.md`, `apps/springBoot/README.md`, `Dockerfile`, `docker-compose.yml`, and `db/schema.sql`
- Produces: one root onboarding document with stable relative links and copy-paste commands

- [ ] **Step 1: Capture the current README baseline**

Run:

```powershell
rg -n "^#{1,3} " README.md
Get-Content README.md | Measure-Object -Line -Word -Character
```

Expected: the old README has separate installation, architecture, model, API, DB, and Docker sections and is roughly 360 lines.

- [ ] **Step 2: Replace the README with the approved reader-first structure**

Use `apply_patch` to replace `README.md` with these sections in this order:

```text
# Hyper Captcha Solver
## 실행 방식 선택
## 요구사항 및 설치
## 5분 빠른 시작
## 아키텍처
## 데이터와 모델 규칙
## Python 학습 및 평가
## Python CLI
## FastAPI 웹 서비스
## Rust ONNX CLI
## Spring Boot ONNX 서비스
## 공통 REST API
## 서비스 설정과 환경 변수
## Docker 배포
## 테스트와 검증
## 알려진 제약과 주의점
## 상세 문서
```

The content must include:

- A comparison table for Python CLI, FastAPI, Rust CLI, and Spring Boot.
- `uv sync`, Python CLI, FastAPI dev, and Docker quick-start commands.
- `captcha_data/<captcha_id>/<rev>` and portable `ONNX + meta.json` layouts.
- The six registered CAPTCHA IDs and the four default serviced IDs.
- Current `H/8, W/4` CRNN output and fixed-length CTC constraint.
- Training, batch evaluation, model synchronization, Rust, and Spring commands.
- Multipart and JSON API examples plus common response/error semantics.
- DB-over-environment precedence and restart/cache behavior.
- Warnings for PT/JIT/ONNX checkpoint divergence, detected/default dimension mismatch, destructive redistribution, import-time CUDA probing, and process-local model caches.
- Relative links to `docs/crnn_ctc.md`, `docs/codebase-analysis.md`, `apps/cli/README.md`, `apps/springBoot/README.md`, and `AGENTS.md`.

- [ ] **Step 3: Check readability and duplication**

Run:

```powershell
rg -n "^#{1,3} " README.md
Get-Content README.md | Measure-Object -Line -Word -Character
rg -n "W/16|No test suite|평면 모듈입니다.*전부|TODO|TBD|PLACEHOLDER" README.md
```

Expected: headings follow the approved order; obsolete statements and placeholders produce no matches; deep implementation detail is linked rather than repeated.

- [ ] **Step 4: Review the README diff**

Run:

```powershell
git diff -- README.md
git diff --check -- README.md
```

Expected: only intentional README content changes appear and `git diff --check` exits 0.

### Task 2: Verify commands, links, and code facts

**Files:**
- Verify: `README.md`
- Reference: `pyproject.toml`, `engine.py`, `core.py`, `web/api/v1/predict.py`, `web/core/config.py`, `web/core/db.py`, `apps/cli/Cargo.toml`, `apps/springBoot/pom.xml`

**Interfaces:**
- Consumes: completed `README.md` from Task 1
- Produces: verification evidence that documented paths, facts, and the lightweight Python check are valid

- [ ] **Step 1: Verify every documented relative link target**

Run this PowerShell check:

```powershell
$targets = @(
  'docs/crnn_ctc.md',
  'docs/codebase-analysis.md',
  'apps/cli/README.md',
  'apps/springBoot/README.md',
  'AGENTS.md'
)
$missing = $targets | Where-Object { -not (Test-Path $_) }
if ($missing) { throw "Missing README targets: $($missing -join ', ')" }
```

Expected: exit 0 with no missing targets.

- [ ] **Step 2: Verify key facts against source**

Run:

```powershell
rg -n "MaxPool2d\(|get_captcha_type_list|predictImage|predictJson" core.py engine.py web/api/v1/predict.py
rg -n "requires-python|entrypoint" pyproject.toml
rg -n "<java.version>|spring-boot-starter-parent|onnxruntime.version" apps/springBoot/pom.xml
```

Expected: source confirms three pooling stages ending in `(2,1)`, registered IDs, both API paths, Python 3.12, and Spring/ONNX versions.

- [ ] **Step 3: Run the lightweight Python decoder verification**

Run:

```powershell
uv run python test_ctc_decode.py
```

Expected: six `ok` lines and `모든 검사 통과`, exit 0.

- [ ] **Step 4: Perform final Markdown checks**

Run:

```powershell
git diff --check -- README.md
rg -n "TODO|TBD|PLACEHOLDER|W/16" README.md
git status --short
```

Expected: whitespace check exits 0; forbidden text produces no matches; status shows `README.md` plus only pre-existing user changes and previously approved documentation artifacts.

- [ ] **Step 5: Commit the README independently if requested**

Do not include unrelated working-tree changes. If the user requests a commit, run:

```powershell
git add -- README.md
git commit -m "docs: improve project README"
```
