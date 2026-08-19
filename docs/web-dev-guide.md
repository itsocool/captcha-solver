# apps/web 개발 가이드

> 로컬 개발 환경, 설정, 테스트, 확장 절차. 아키텍처는 [web-architecture.md](./web-architecture.md), API 계약은 [web-api-reference.md](./web-api-reference.md), 프런트엔드 구조는 [web-frontend-guide.md](./web-frontend-guide.md) 참고.

## 1. 로컬에서 띄우기

```bash
uv sync                    # 저장소 루트에서 1회, 의존성 설치 (hypercaptcha workspace 포함)
./apps/web/server.sh start # 백그라운드 기동, /health 응답까지 대기
```

`server.sh`는 `start|stop|restart|status|logs [-f]`를 지원하며 상태 파일(`apps/web/.dev/`: PID, 로그)을 관리한다. Windows는 같은 기능의 `server.ps1`(옵션: `-Port`/`-BindHost`/`-NoReload`)을 쓴다.

포그라운드로 직접 띄우려면 (저장소 루트에서):

```bash
uv run uvicorn web.app:app --host 0.0.0.0 --port 5000 --reload --reload-dir apps/web
```

`--reload-dir` 없이 띄우면 uvicorn이 `captcha_data`(이미지 수만 장)까지 감시해 리로드가 사실상 멈춘다.

환경 변수:

| 변수 | 기본값 | 용도 |
|---|---|---|
| `PORT` | 5000 | `server.sh`가 바인드할 포트 |
| `HOST` | 0.0.0.0 | `server.sh`가 바인드할 주소 |
| `RELOAD` | 1 | 1이면 코드 변경 시 자동 리로드 |
| `START_TIMEOUT` | 120 | 기동 대기 한계(초) — lifespan이 ONNX 모델을 preload하므로 여유를 둠 |
| `STOP_TIMEOUT` | 20 | graceful 종료 대기 한계(초) |
| `UVICORN` | `.venv/bin/uvicorn` 우선 | uvicorn 실행 파일 경로 |

## 2. 앱 설정 (`.env`)

`core/config.py`의 `Settings`(pydantic-settings)가 저장소 루트 `.env`를 읽는다. `.env.example`을 복사해 시작한다.

| 필드 | 기본값 | 설명 |
|---|---|---|
| `app_title` | `Captcha Solver` | FastAPI 타이틀 |
| `app_version` | (비어있음) | 표시용 버전. 비면 `pyproject.toml` → 패키지 메타데이터로 폴백. 이미지 재빌드 없이 이 값만 바꿔 배포 가능 |
| `default_captcha_id` | `supreme_court` | `service_captchas` 테이블이 비어있을 때만 쓰는 폴백 |
| `web_context_path` / `WEB_CONTEXT_PATH` | (비어있음) | 리버스 프록시 뒤 하위 경로에 물릴 때의 접두사 (예: `/captcha`). 자세한 내용은 §8 |
| `web_host` / `WEB_HOST` | `0.0.0.0` | 직접 실행 시 바인드 주소 |
| `web_port` / `WEB_PORT` | 5000 | 직접 실행 시 바인드 포트 |
| `web_debug` / `WEB_DEBUG` | `false` | 직접 실행(`python app.py`) 시 reload 여부 |
| `db_path` / `DB_PATH` | `./db/captchaSolver.sqlite3` | SQLite 파일 경로(저장소 루트 기준 상대경로) |
| `db_schema_path` / `DB_SCHEMA_PATH` | `./db/schema.sql` | 기동 시 적용할 스키마 |
| `db_seed_path` / `DB_SEED_PATH` | `./db/seed_captcha_types.sql` | 기동 시 적용할 시드 (없으면 건너뜀) |

**중요**: 기본 서비스 캡차는 환경 변수가 아니라 `db/schema.sql`의 `service_captchas.is_default`가 결정한다. `DB_DRIVER`/`DATABASE_URL` 필드는 존재하지만 현재 DB 계층은 stdlib `sqlite3`와 `db_path`만 쓴다. `.env.example`의 `DB_URL`은 `Settings` 필드명(`database_url`)과 달라 실제로는 아무것도 덮어쓰지 않는다 — `.env`를 새로 작성할 때 이 이름 불일치에 주의한다.

## 3. 데이터베이스

- `init_db()`가 기동할 때마다 `db/schema.sql` + 시드를 재적용한다. 스키마는 `CREATE TABLE IF NOT EXISTS`/`INSERT OR IGNORE`로 작성돼 반복 실행이 안전하다.
- 컬럼을 추가하는 마이그레이션은 `schema.sql` 자체가 아니라 `core/db.py`의 `_add_missing_columns()`에 추가한다 — SQLite에 `ADD COLUMN IF NOT EXISTS`가 없기 때문이다. 새 컬럼이 필요하면 `(table, column, decl)` 튜플을 이 리스트에 추가한다.
- 테이블 구조는 `db/schema.sql` 참고: `captcha_types`, `train_data_configs`, `train_data_characters`, `train_info_cache`, `service_captchas`(서비스 대상/기본 캡차), `character_sets`(이름 붙은 문자 집합 상수), `train_run_params`(UI 폼 값 영속화).
- DB를 초기 상태로 리셋하려면 `db/captchaSolver.sqlite3` 파일을 삭제하고 서버를 재기동한다 (스키마+시드가 자동 재적용됨). **Docker Compose 기본 설정에서는 이 파일이 이미지 안에 있어 컨테이너를 재생성하면 휘발된다** — `service_captchas` 수정을 영속화하려면 `DB_PATH`를 볼륨 마운트 경로로 옮겨야 한다 (`docker-compose.yml` 주석 참고).

## 4. 테스트

```bash
uv run pytest tests/
```

현재 `tests/`에는 웹 서비스 계층에 대한 두 테스트가 있다.

- `tests/test_captcha_service_cache.py` — `services/captcha.py`의 `_MODEL_CACHE` 동작 (device 키, 로드 실패 시 캐시 미보관 등)
- `tests/test_prediction_model_load.py` — 모델 로드 경로

`services/*.py`는 FastAPI에 의존하지 않는 순수 파이썬이라 서버를 띄우지 않고도 함수를 직접 호출해 확인할 수 있다 (`services/batch_predict.py`, `services/train.py`, `services/data_source.py`의 `run()`은 평범한 `dict`-yield 제너레이터). 새 서비스 로직을 추가할 때 이 성질을 유지하면 테스트가 쉬워진다 — `hypercaptcha`/`torch` import를 함수 안으로 지연시키는 관례도 이와 맞물려 있다 (§6 참고).

저장소 전체에는 별도 lint/typecheck 설정(ruff, mypy 등)이 없다.

## 5. 새 API 엔드포인트 추가 절차

1. **`services/`에 순수 함수 작성** — FastAPI를 import하지 않는다. 검증 실패는 `ValueError`(→ 라우터가 400으로 변환)로, 상태 충돌은 전용 예외(`XxxBusy` → 409)로 던진다. 무거운 라이브러리(`hypercaptcha`, `torch`)는 함수 본문 안에서 import한다.
2. **필요하면 `schemas/`에 Pydantic 모델 추가** — JSON 요청/응답 스키마 (`schemas/predict.py` 참고).
3. **`api/v1/<feature>.py`에 라우터 작성** — `services/`를 호출하고 예외를 HTTP 상태로 매핑한다. 장시간 실행 + 진행률이 필요하면 [web-architecture.md §4.2](./web-architecture.md#42-백그라운드-작업--sse-일괄-추론--학습--데이터-수집)의 SSE 패턴을 따른다: 동기 제너레이터 + `StreamingResponse(media_type="text/event-stream")` + 시작 전 검증은 스트림을 열기 전에.
4. **`api/v1/router.py`에 `include_router()` 등록**.
5. **HTML UI가 필요하면** `frontend/router.py`에 라우트를 추가하고 [web-frontend-guide.md](./web-frontend-guide.md) §7을 따른다.
6. Docker 이미지 재빌드 없이 테스트하려면 로컬 `server.sh restart`로 충분하다.

## 6. Gotchas (apps/web 한정)

- **`hypercaptcha`/`torch` import는 함수 안에서** — `services/captcha.py`, `services/batch_predict.py`, `services/train.py`, `core/device.py` 전부 이 규칙을 따른다. 모듈 최상위에서 import하면 그 모듈이 로드되는 순간(예: 다른 코드가 무심코 import만 해도) CUDA/cuDNN 초기화가 딸려 붙는다.
- **모델 캐시는 `(captcha_id, device)` 키** — `captcha_id`만으로 `_MODEL_CACHE`를 조회하는 코드를 새로 쓰지 않는다.
- **모델/DB를 바꿔도 실행 중 서버에 자동 반영되지 않는다** — 재시작 필요. `/status` 페이지가 현재 로드 상태를 보여준다.
- **멀티 워커 미지원** — 모델 캐시·서비스 설정 캐시·학습 세션이 전부 프로세스 로컬이다. `uvicorn --workers N`으로 띄우면 워커마다 상태가 어긋난다.
- **배치/학습/데이터수집은 각각 전역 락 하나씩** — 같은 종류의 작업은 서버 전체에서 동시에 하나만 실행된다(다른 종류끼리는 동시 실행 가능). 이미 실행 중이면 `409`.
- **경로 파라미터로 받은 파일명은 항상 basename + 확장자 제한 + `is_relative_to()` 재검증** — `services/batch_predict.pred_image_path()`, `services/data_source.draft_image_path()` 패턴을 그대로 재사용한다. 새로운 파일 서빙 엔드포인트를 추가할 때 빠뜨리기 쉬운 부분이다.
- **`services/data_source.py`는 SSRF를 의도적으로 막지 않는다** — 임의 URL을 서버가 대신 요청하는 게 기능 자체다(LAN 내부 캡차 소스). 배포 시 반드시 방화벽/리버스 프록시로 접근을 제한해야 한다. 인증 계층이 없다.
- **`shuffle` 학습 파라미터는 저장하지 않는다** — 되돌릴 수 없는 파일 재분배 동작이라, 저장했다가 다음 실행에 자동 적용되면 매번 재분배되는 사고로 이어진다 (`services/train.py` `PERSIST_PARAMS` 주석 참고).
- **테스트가 웹 계층 전체를 커버하지 않는다** — 학습/웹 통합 테스트는 부족한 상태다 (`docs/codebase-analysis.md` §10 우선순위 권고 참고). 새 기능을 추가할 때 최소한 `services/` 레벨 유닛 테스트를 같이 작성하는 것을 권장한다.

## 7. Docker

- `Dockerfile`(GPU, CUDA 빌드) / `Dockerfile.cpu`(CPU 전용) 두 가지가 있다.
- `docker-compose.yml`은 호스트 `30008` → 컨테이너 `8000`으로 노출하고, `deploy.resources.reservations.devices`로 NVIDIA GPU를 예약한다(호스트에 NVIDIA Container Toolkit 필요). GPU가 없는 호스트에서는 `docker-compose-cpu.yml`을 쓴다.
- `captcha_data/`는 읽기-쓰기 볼륨으로 마운트한다 — `/train` UI가 여기 `model/`에 가중치를 쓰므로 `:ro`로 두면 저장이 실패한다.
- `.env`는 `env_file`로 런타임에 주입되며(이미지에는 굽지 않음), 값을 바꾸면 재빌드 없이 재기동만으로 반영된다.
- SQLite DB는 컨테이너 안 경로에 있어 기본 설정으로는 컨테이너 재생성 시 초기화된다 (§3 참고).

```bash
docker compose up --build        # GPU 호스트
docker compose -f docker-compose-cpu.yml up --build  # CPU 전용
```

## 8. 리버스 프록시 하위 경로 배포 (`WEB_CONTEXT_PATH`)

앱을 도메인 루트가 아니라 리버스 프록시의 하위 경로(예: `https://example.com/captcha/`)에 물릴 때 쓴다. `.env`에 `WEB_CONTEXT_PATH=/captcha`(또는 `captcha`, `/captcha/` — `core/config.py`의 `field_validator`가 어느 형태든 `/captcha`로 정규화한다)를 넣으면:

- `core/db.py`가 아니라 `app.py`의 `create_app()`이 `FastAPI(root_path=settings.web_context_path)`로 앱을 만든다 — Swagger(`/docs`)가 접두사 붙은 `openapi.json`을 올바르게 찾고, `openapi.json`의 `servers`에 `{"url": "/captcha"}`가 실린다.
- `PrependRootPath` ASGI 미들웨어(`app.py`)가 함께 동작한다. FastAPI 라우트 매칭 자체는 `root_path`만으로 접두사 없는 요청(`/health`)도 통과시키지만, `StaticFiles` 같은 `Mount`는 하위 `root_path`(`/captcha/static/...`)를 스스로 못 잘라내 404가 난다. 이 미들웨어가 프록시가 접두사를 이미 떼고 넘겼는지 판단해 없으면 앞에 붙여준다 — 그 결과 프록시가 접두사를 떼든 안 떼든(`/health` 요청이든 `/captcha/health` 요청이든) 같은 라우트로 들어온다.
- 템플릿의 링크·정적 파일 경로(`href`, `src`, 폼 `action`)는 절대 경로(`/static/...`)라서 `root_path`가 자동으로 붙지 않는다. `app.py`가 `templates.env.globals["context_path"]`로 전역 변수를 넘기고, 템플릿이 링크에 `{{ context_path }}/...` 식으로 직접 접두사를 붙인다. 정적 자산은 `static_url('js/xxx.js')` 전역 함수가 접두사와 함께 파일 mtime 기반 `?v=` 캐시 무효화 파라미터를 붙여 준다.
- 정적 JS(`train.js`, `predict.js`, `data_source.js`)는 서버 렌더링 문맥이 없으므로 `base.html`이 `<html data-context-path="{{ context_path }}">`로 내려준 값을 `document.documentElement.dataset.contextPath`로 읽어 `fetch`/`EventSource` URL 앞에 붙인다 (`CONTEXT_PATH` 상수, 자세한 패턴은 [web-frontend-guide.md](./web-frontend-guide.md#9-리버스-프록시-하위-경로-context_path) 참고). `app.js`(index.html의 단일 예측 폼)는 예외 — 폼 `action`을 서버가 `frontend/router.py`에서 이미 접두사 붙여 렌더링하므로 JS가 따로 처리할 게 없다.

기본값(빈 문자열)은 루트(`/`)에 뜬 것으로 보고 아무것도 변하지 않는다. 동작을 확인하려면 `tests/test_context_path.py`를 참고 — 정규화 케이스와, 프록시가 접두사를 떼거나 안 떼는 두 시나리오 모두 같은 라우트/정적 파일/Swagger로 연결되는지를 검증한다.

```bash
uv run pytest tests/test_context_path.py -v
```
