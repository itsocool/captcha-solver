# apps/web API 레퍼런스

> `apps/web`이 제공하는 모든 HTTP/SSE 엔드포인트. 아키텍처와 요청 흐름은 [web-architecture.md](./web-architecture.md) 참고.

## 공통 사항

- Base URL: 개발서버 기본값 `http://localhost:5000` (`server.sh`), Docker 기본값은 호스트 `30008` → 컨테이너 `8000` (`docker-compose.yml`).
- 시스템 엔드포인트(`/health`, `/ping`, `/version`)는 루트에 있고, 그 외 API는 전부 `/api/v1` prefix를 쓴다.
- `WEB_CONTEXT_PATH`(예: `/captcha`)가 설정된 배포에서는 이 문서의 모든 경로 앞에 그 접두사가 추가로 붙는다 (예: `GET /captcha/health`, `POST /captcha/api/v1/predictImage`). 프록시가 접두사를 미리 떼고 넘기든 그대로 넘기든 양쪽 다 같은 라우트로 연결된다 — 자세한 동작은 [web-architecture.md §8](./web-architecture.md#8-리버스-프록시-하위-경로-web_context_path) 참고.
- 오류 응답은 FastAPI 기본 형식 `{"detail": "..."}`이다.
- 대부분의 엔드포인트가 받는 `device` 파라미터는 `auto`(기본) / `cpu` / `cuda` 중 하나다. 생략하거나 빈 값이면 `auto`이며, `auto`는 CUDA 가용 시 CUDA, 아니면 CPU로 해석된다. 쓸 수 없는 디바이스를 요청하면 `400`.
- SSE 스트림은 전부 `Cache-Control: no-cache`, `X-Accel-Buffering: no` 헤더를 붙인다 (리버스 프록시 버퍼링으로 진행률이 한꺼번에 도착하는 것을 막기 위함).

## 1. 시스템 (`api/system.py`, prefix 없음)

### `GET /health`

서비스 준비 상태. 서비스 대상 캡차가 전부 로드됐으면 `ok`, 일부만 로드됐으면 `degraded`.

```json
{
  "status": "ok",
  "version": "0.8",
  "default_captcha_id": "supreme_court",
  "serviced_captcha_ids": ["supreme_court", "gov24", "wetax", "iptime"],
  "loaded_captcha_ids": ["gov24", "iptime", "supreme_court", "wetax"],
  "config_source": "db"
}
```

`config_source`는 `service_captchas` 테이블에서 읽었으면 `"db"`, 테이블이 비어 `.env`의 `DEFAULT_CAPTCHA_ID`로 폴백했으면 `"fallback"`이다.

### `GET /ping`

`{"ping": "pong"}`. 로드밸런서/헬스체크용 최소 확인.

### `GET /version`

`{"version": "0.8"}`. `.env`의 `APP_VERSION` → `pyproject.toml` → 설치된 패키지 메타데이터 순으로 폴백한다.

## 2. 추론 (`api/v1/predict.py`, prefix `/api/v1`)

### `POST /api/v1/predictImage`

`multipart/form-data`.

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `image` | file | 예 | 캡차 이미지 |
| `captcha_id` | string | 아니오 | 생략 시 서비스 기본 캡차 사용 |
| `device` | string | 아니오 | `auto`/`cpu`/`cuda` |

```bash
curl -X POST http://localhost:5000/api/v1/predictImage \
  -F "captcha_id=supreme_court" -F "device=auto" -F "image=@captcha.png"
```

### `POST /api/v1/predictJson`

`application/json`.

```json
{
  "captcha_id": "supreme_court",
  "image_data": "data:image/png;base64,iVBORw0KGgo...",
  "device": "auto"
}
```

`image_data`는 순수 base64 또는 `data:image/...;base64,` 접두사가 붙은 형태 둘 다 받는다. `captcha_id`/`device`는 생략 가능.

### 공통 응답 (`PredictResponse`)

```json
{
  "captcha_id": "supreme_court",
  "prediction": "3fk2p",
  "confidence": 0.9931,
  "elapsed_ms": 42,
  "device": "cuda:0"
}
```

`device`는 요청 값이 아니라 **실제로 추론에 쓰인 디바이스**다 — `auto`로 요청했을 때 무엇이 선택됐는지 이 필드로 확인한다.

### 오류

| 상태 | 원인 |
|---|---|
| 400 | 이미지 없음, base64 형식 오류, 비서비스 `captcha_id`, 쓸 수 없는 `device` |
| 500 | 이미지 디코딩/모델 로드/추론 중 예외 (`CaptchaPredictionError`) |

## 3. 일괄 추론 (`api/v1/batch.py`, prefix `/api/v1/batch`)

`captcha_data/<id>/<rev>/images/pred/` 전체를 대상으로 예측하고 파일명(라벨)과 대조한다.

### `GET /api/v1/batch/targets`

`{"targets": [...], "running": false}`. 각 target: `{captcha_id, name, rev, pred_count, has_model, selectable, reason}`. `selectable=false`면 `reason`에 이유(모델 없음/pred 이미지 없음).

### `GET /api/v1/batch/stream?captcha_id=...&rev=1&device=auto`

SSE. 시작 전 검증 실패는 `400`(선택 불가 대상) 또는 `409`(이미 다른 배치가 실행 중)로 응답한다. 정상 시작 후 이벤트:

| 이벤트 | 시점 | 대표 필드 |
|---|---|---|
| `start` | 1회, 시작 시 | `total`, `device`, `loss_type` |
| `item` | 이미지마다 | `index`, `image`, `expected`, `pred`, `confidence`, `match`, `error?` |
| `summary` | 1회, 종료 시 | `accuracy`, `mismatch`, `elapsed_sec` |
| `error` | 실행 중 예외 | `message` |

### `GET /api/v1/batch/image?captcha_id=...&rev=1&name=xxx.png`

`images/pred` 안의 썸네일 이미지를 반환 (`image/png`). `name`은 basename만 허용하고 `pred` 디렉터리 밖으로 벗어나는 경로는 `400`으로 거부한다.

## 4. 학습 (`api/v1/train.py`, prefix `/api/v1/train`)

### `GET /api/v1/train/targets`

`{"targets": [...], "running": bool, "active": {"captcha_id", "rev"} | null}`. target: `{captcha_id, name, rev, train_count, pred_count, has_model, selectable, reason}` — `train_count > 0`이면 선택 가능.

### `GET /api/v1/train/params?captcha_id=...&rev=1`

대상에 마지막으로 저장된 학습 파라미터(없으면 기본값)를 돌려준다: `{"params": {...}}`.

### `POST /api/v1/train/params?captcha_id=...&rev=1&<param>=<value>...`

쿼리스트링으로 폼 파라미터를 저장한다 (학습을 시작하지 않아도 유지됨). 유효성 실패 시 `400`.

### `POST /api/v1/train/start?captcha_id=...&rev=1&device=auto&<params>`

백그라운드 학습을 시작하고 즉시 반환한다. 진행 상황은 `GET /train/stream`으로 별도 확인한다.

파라미터 (쿼리스트링, `PARAM_SPEC` 정의 — `services/train.py`):

| 이름 | 범위 | 기본값 |
|---|---|---|
| `epochs` | 1–500 | 80 |
| `batch_size` | 1–512 | 64 |
| `early_stopping_patience` | 0–100 | 15 |
| `learning_rate` | 1e-6–1.0 | 0.001 |
| `warmup_epochs` | 0–100 | 0 |
| `train_ratio` | 0.1–0.95 | 0.6 |
| `loss_type` | `focal`만 허용 | `focal` |
| `use_amp` | bool (`1/true/yes/on`) | `true` |
| `shuffle` | bool | `false` — train/pred 이미지를 디스크에서 되돌릴 수 없게 재분배하므로 저장하지 않는 1회성 값 |

응답: `{"started": true, "captcha_id": ..., "rev": ...}`.

| 상태 | 원인 |
|---|---|
| 400 | 대상 선택 불가, 파라미터 검증 실패, 디바이스 오류 |
| 409 | 이미 다른 학습 실행 중 (`TrainBusy`) |

### `POST /api/v1/train/stop?save=true`

실행 중인 학습에 중단을 요청한다. 에폭 경계에서 멈추므로 진행 중인 에폭 하나는 마저 돈다. `save=false`면 best 모델을 저장하지 않고 버린다. 돌고 있는 학습이 없으면 `409`.

### `GET /api/v1/train/stream`

진행 중이거나 방금 끝난 학습 세션에 붙는 SSE. 파라미터 없음 — 시작은 `/train/start`가 별도로 한다. 연결할 때마다 세션 버퍼를 처음부터 재생하므로 여러 탭이 동시에 붙어도 각자 히스토리를 받는다.

| 이벤트 | 설명 |
|---|---|
| `idle` | 진행 중인 학습이 없음 |
| `start` | 학습 시작. `epochs`, `batch_size`, `lr`, `device`, `loss_type`, `image_width/height`, `use_amp` 등 |
| `shuffle` | `shuffle=true`일 때 재분배 결과 (`final_train`, `final_pred`) |
| `epoch` | 에폭마다. `epoch`, `train_loss`, `val_loss`, `lr`, `elapsed_sec`, `improved`, `best_val_loss`, `patience_counter` |
| `skipped` | 학습 결과가 기존 모델보다 나빠 아티팩트를 교체하지 않음. `incumbent_val_loss`, `best_val_loss`, `epochs_run` |
| `done` | 정상 종료. `stop_reason`(`completed`/`early_stopping`/`cancelled`/`cancelled_discarded`), `artifacts`(파일 경로 dict), `best_val_loss`, `best_epoch` |
| `error` | 예외 발생. `message` |

## 5. 데이터 수집 (`api/v1/data_source.py`, prefix `/api/v1/data-source`)

지정한 URL에서 캡차 이미지를 반복 요청해 `images/draft/`에 라벨 없이 저장한다.

### `GET /api/v1/data-source/targets`

`{"targets": [...], "running": bool}`. target: `{captcha_id, name, rev, draft_count}`. 학습과 달리 이미지가 아직 없는 캡차도 레지스트리의 rev(1부터 시작, 기본 1)로 포함된다 (모으는 게 목적이므로).

### `GET /api/v1/data-source/params?captcha_id=...&rev=1`

대상에 마지막으로 저장된 수집 입력값(없으면 기본값 `{"url": "", "content_type": "image", "selector": "", "count": 500, "delay_ms": 0}`)을 돌려준다: `{"params": {...}}`.

### `POST /api/v1/data-source/params?captcha_id=...&rev=1&url=&content_type=&selector=&count=&delay_ms=`

쿼리스트링으로 폼 입력값을 저장한다 (수집을 시작하지 않아도 유지됨). `url`은 비워도 되지만 넣으면 `http(s)://`여야 하고, `count`/`delay_ms`는 stream 과 같은 범위. 유효성 실패·미등록 캡차·`rev < 1`은 `400`. 프런트는 입력 `change` 때와 **추가 버튼 클릭 때** 이걸 호출하고, 수집을 실행하면(`/data-source/stream`) 서버도 그때 쓴 값을 같은 저장소에 남긴다.

### `GET /api/v1/data-source/stream?captcha_id=...&rev=1&url=...&content_type=image&selector=&count=N&delay_ms=0`

| 파라미터 | 설명 |
|---|---|
| `rev` | 1 이상 (리비전은 1부터 시작) |
| `url` | `http://`/`https://`로 시작해야 함 |
| `content_type` | `image`(기본) \| `html` \| `json`. URL 응답을 어떻게 읽어 이미지를 얻을지 — `image`: 본문이 이미지 그 자체(셀렉터 미사용). `html`: 본문을 HTML로 파싱해 `selector`(CSS)로 이미지 요소를 찾고 그 주소를 다시 받는다. `json`: 본문을 JSON으로 읽고 `selector`를 키 경로(`data.image`, `items[0].url`)로 써서 값을 꺼낸다 — 값은 이미지 URL(상대 경로 허용) · `data:` URI · 순수 base64 |
| `selector` | `html`: CSS 셀렉터(`src`/`data-src`/`href`/`background:url()` 순으로 탐색, 빈 값이면 `img`). `json`: 키 경로(**필수**). `image`: 무시 |
| `count` | 1–5000 (`MAX_COUNT`) |
| `delay_ms` | 요청 사이 대기(ms). 0–30000 (`MAX_DELAY_MS`), 기본 0. 첫 요청 앞에는 대기하지 않는다 |

요청은 브라우저 유사 헤더(`REQUEST_HEADERS`: Chrome User-Agent 등)로 보낸다 — 대법원 등 공공기관 WAF가 기본 `python-httpx` UA를 차단하기 때문. 받은 이미지는 PNG로 통일해 저장하며 투명 배경(RGBA)은 흰 배경에 합성한다(학습 데이터 규약). 시작 전 검증 실패는 `400`, 이미 실행 중이면 `409`. 응답 크기 상한은 8MB (`MAX_RESPONSE_BYTES`)이며 초과 시 해당 건은 `item` 이벤트의 실패로 기록된다. 이 엔드포인트는 서버가 대신 임의 URL을 요청하므로(SSRF 성격), 사설 대역을 의도적으로 막지 않는다 — LAN 내부 장비에서 캡차를 가져오는 용도이기 때문이다 (자세한 배경은 [web-architecture.md §7](./web-architecture.md#7-동시성과-프로세스-로컬-상태-알아둘-것)).

| 이벤트 | 설명 |
|---|---|
| `start` | 1회. `total`, `content_type`, `delay_ms`, `draft_dir`, `existing`(기존 draft 장수), `start_index` |
| `item` | 요청마다. `index`, `saved`, 성공 시 `name`/`image_url`/`bytes`, 실패 시 `error` |
| `summary` | 1회. `requested`, `saved`, `failed`, `draft_total`, `elapsed_sec` |
| `error` | 검증 통과 후 실행 중 예외 |

### `GET /api/v1/data-source/drafts?captcha_id=...&rev=1&limit=N`

draft 이미지 목록 (수집 순 = mtime 오름차순). `limit` 생략 시 전체.

```json
{"names": ["000001.png", "..."], "total": 42, "draft_dir": "/app/captcha_data/gov24/1/images/draft"}
```

### `POST /api/v1/data-source/label?captcha_id=...&rev=1&name=000001.png&label=3fk2p`

draft 이미지 파일명을 라벨로 바꾼다(=라벨링). 저장소 관례가 "파일명 = 정답"이라 이름 변경이 곧 라벨 지정이다. 이미 같은 이름이 있으면 `400`.

```json
{"name": "3fk2p.png", "renamed": true}
```

### `GET /api/v1/data-source/image?captcha_id=...&rev=1&name=xxx.png`

draft 이미지 썸네일 (`image/png`). 경로 검증은 `/batch/image`와 동일한 규칙.

## 6. 오류 응답 요약

| 상태 코드 | 의미 | 발생 예 |
|---|---|---|
| 400 | 요청 값 검증 실패 | 잘못된 `device`, 비서비스 `captcha_id`, 범위를 벗어난 학습 파라미터, 잘못된 URL |
| 404 | (FastAPI 기본) 경로 없음 | 존재하지 않는 엔드포인트 |
| 409 | 동일 종류 작업 중복 실행 | 배치/학습/수집이 이미 실행 중일 때 |
| 500 | 서버 내부 오류 | 추론 실패 등 예기치 못한 예외 |
