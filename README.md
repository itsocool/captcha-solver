# Hyper Captcha Solver

PyTorch 학습부터 Python·Rust·Spring Boot 추론까지 한 저장소에서 운영하는 CRNN + CTC CAPTCHA 문자 인식 프로젝트입니다. 먼저 사용할 런타임을 고른 뒤 필요한 모델과 명령만 실행하세요.

## 실행 방식 선택

| 방식 | 용도 | 모델/런타임 | 시작점 |
|---|---|---|---|
| Python CLI | 이미지 한 장, 개발·디버깅 | `model_full.pt`, Python/PyTorch | `uv run python main.py ...` |
| FastAPI | 웹 UI와 HTTP API | `model_full.pt`, Python/PyTorch | `uv run fastapi dev web/app.py` |
| Rust CLI | 독립 실행 파일·배포 | portable ONNX + `meta.json` | `apps/cli`의 `captcha-cli` |
| Spring Boot | JVM HTTP 서비스 | portable ONNX + `meta.json` | `apps/springBoot` |

## 요구사항 및 설치

- Python `3.12.x` (고정), [uv](https://docs.astral.sh/uv/)
- Rust CLI: Rust toolchain과 Cargo
- Spring Boot: JDK 25와 Maven
- GPU/CUDA는 선택 사항이며 CPU에서도 실행됩니다.

```bash
uv sync
cp .env.example .env       # 선택: 설정을 덮어쓸 때
```

PyTorch 휠은 `pyproject.toml`의 CUDA 13 인덱스를 사용합니다. CPU 전용 설치가 필요하면 해당 인덱스를 조정한 뒤 `uv lock`을 갱신하세요.

## 5분 빠른 시작

```bash
# Python CLI (샘플 경로는 보유한 이미지로 바꾸세요)
uv run python main.py -c supreme_court -i captcha_data/supreme_court/0/images/pred/091082.png

# FastAPI 개발 서버: http://localhost:8000
uv run fastapi dev web/app.py --host 0.0.0.0 --port 8000

# Docker: http://localhost:5001
docker compose up --build
```

서버가 시작되면 `/health`와 `/docs`를 확인하세요. 운영 실행은 `uv run fastapi run web/app.py --host 0.0.0.0 --port 8000`입니다.

## 아키텍처

`engine.py`가 캡차 설정·학습·예측을 조정하고, `core.py`의 CNN → Feature Projection → 2층 BiLSTM → CTC가 인식합니다. FastAPI는 요청을 검증하고 프로세스 메모리의 모델 캐시를 통해 `engine.predict`를 호출합니다. Rust와 Spring은 동일한 전처리·고정 길이 CTC 디코더를 ONNX Runtime으로 실행합니다.

```text
client → web/api → captcha service/cache → engine → PyTorchModel → CRNN/CTC
                                  └→ ONNX (Rust/Spring portable runtime)
```

자세한 레이어와 디코더 설명은 [CRNN/CTC 문서](docs/crnn_ctc.md)와 [코드베이스 분석](docs/codebase-analysis.md)을 참고하세요.

## 데이터와 모델 규칙

Python 원본은 다음 레이아웃을 사용합니다. 이미지 파일명(확장자 제외)이 정답 레이블입니다.

```text
captcha_data/<captcha_id>/<rev>/
├── images/{train,pred}/
└── model/{model_full.pt,model_jit.pt,model_full.pt.onnx}
```

Rust·Spring은 portable 디렉터리로 ONNX와 메타데이터를 함께 배포합니다.

```text
models/
├── <captcha_id>.onnx
└── <captcha_id>.meta.json  # image_width, image_height, label_length, characters, threshold, preprocess
```

등록된 CAPTCHA ID는 `default`, `dev`, `supreme_court`, `gov24`, `wetax`, `kshop` 여섯 가지입니다. DB의 기본 서비스 대상은 `supreme_court`, `gov24`, `wetax`, `kshop` 네 가지입니다. 학습 데이터가 있으면 이미지 크기·레이블 길이·문자 집합을 파일명에서 감지합니다.

CRNN은 입력에서 특징 맵 높이 `H/8`, 시간축 너비 `W/4`를 출력합니다. 고정 길이 CTC 디코딩은 `W/4 >= label_length`를 요구하므로 모델 입력 폭을 레이블 길이보다 충분히 크게 유지하세요.

## Python 학습 및 평가

`train.py`와 `pred.py` 상단 변수를 수정한 뒤 실행합니다(명령 인자를 받는 스크립트가 아닙니다).

```bash
uv run python train.py              # 학습, PT/JIT/ONNX 산출물 생성
uv run python pred.py               # images/pred 배치 평가
uv run python test_ctc_decode.py    # CTC 디코더 단위 검증
```

엔진 API로도 학습·배치 평가·데이터 재분배를 호출할 수 있습니다.

```python
import engine
engine.train_model(captcha_id="supreme_court")
engine.batch_predict_model(captcha_id="supreme_court")
engine.redistribute_train_pred("captcha_data/supreme_court/0/images", train_ratio=0.9)
```

재분배는 파일을 이동해 `train`/`pred`를 다시 나누는 파괴적 작업이므로 백업 후 실행하세요.

## Python CLI

```bash
uv run python main.py -c <captcha_id> -i <image_path> [-v]
```

기본 출력은 예측 문자열이고, `-v`는 `predicted_text`, `confidence`, `execution_time` JSON을 출력합니다. 이미지가 없으면 종료 코드 2, 모델 생성 실패면 3입니다. 상세 사용법은 [Python CLI 문서](apps/cli/README.md)를 참고하세요.

## FastAPI 웹 서비스

```bash
uv run fastapi dev web/app.py --host 0.0.0.0 --port 8000
```

`/`는 웹 UI, `/status`는 모델 상태, `/health`, `/ping`, `/version`은 상태·버전 엔드포인트입니다. 서버 기동 시 서비스 대상 모델을 preload/warm-up하고 `_MODEL_CACHE`에 보관하므로 모델 파일을 바꾼 뒤에는 프로세스를 재시작해야 합니다.

## Rust ONNX CLI

```bash
uv run python apps/cli/tools/sync_models.py       # 전체 ONNX + meta.json 동기화
cd apps/cli
cargo build --release
cargo test
./target/release/captcha-cli -c supreme_court -i captcha.png --json
```

재학습 후 `sync_models.py`를 다시 실행하세요. `--models-dir`, `--meta`, `--beam-width`, `--threads`, `--list`, stdin(`-i -`) 옵션과 Windows 빌드는 [Rust CLI 문서](apps/cli/README.md)에 있습니다.

## Spring Boot ONNX 서비스

```powershell
cd apps/springBoot
mvn spring-boot:run
# 또는 mvn -DskipTests package; java -jar target\captchaSolver-0.0.1-SNAPSHOT.jar
```

기본 포트는 5000이며 모델·DB 기본 상대 경로는 각각 `../../apps/cli/models`, `../../db/captchaSolver.sqlite3`입니다. 상세 설정과 API 문서는 [Spring Boot 문서](apps/springBoot/README.md)를 참고하세요.

## 공통 REST API

FastAPI와 Spring Boot 모두 다음 경로를 제공합니다.

```bash
curl -X POST http://localhost:8000/api/v1/predictImage \
  -F "captcha_id=supreme_court" \
  -F "image=@captcha_data/supreme_court/0/images/pred/091082.png"

curl -X POST http://localhost:8000/api/v1/predictJson \
  -H 'Content-Type: application/json' \
  -d '{"captcha_id":"supreme_court","image_data":"iVBORw0KGgo..."}'
```

JSON 요청은 순수 base64 또는 `data:image/png;base64,...`를 받습니다. 성공 응답은 `{"captcha_id":"supreme_court","prediction":"091082","confidence":0.99,"elapsed_ms":13}` 형태입니다. `captcha_id` 생략 시 DB 기본값을 사용합니다. 잘못된 이미지·base64·미서비스 ID는 400, 모델 로드/추론 실패는 500이며 오류 본문은 `{"detail":"..."}`입니다.

## 서비스 설정과 환경 변수

`db/schema.sql`의 `service_captchas`가 활성화된 ID와 기본 ID를 결정하며, 유효한 DB 값이 환경 변수보다 우선합니다. DB가 비어 있거나 설정이 없을 때만 `DEFAULT_CAPTCHA_ID`(기본 `supreme_court`)가 폴백으로 사용됩니다. 변경은 기동 시 한 번 읽으므로 DB·`.env`·모델 변경 후 서버를 재시작해야 캐시와 설정에 반영됩니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `DEFAULT_CAPTCHA_ID` | `supreme_court` | DB 설정이 없을 때의 기본 ID |
| `WEB_HOST` / `WEB_PORT` | `0.0.0.0` / `5000` | 직접 실행 시 바인딩 |
| `WEB_DEBUG` | `false` | reload/debug |
| `APP_TITLE`, `APP_VERSION`, `DB_PATH` | 앱 기본값 | UI·버전·SQLite 경로 |

## Docker 배포

```bash
docker compose up --build
docker compose down
```

Compose는 호스트 `5001`을 컨테이너 `8000`에 연결하고 `captcha_data`를 읽기 전용으로 마운트합니다. `Dockerfile`은 `uv.lock`을 고정해 의존성을 설치하고 `/health` 헬스체크를 사용합니다.

## 테스트와 검증

```bash
uv run python test_ctc_decode.py
cd apps/cli && cargo test
uv run python apps/cli/tools/compare_with_python.py --limit 100
cd apps/springBoot && mvn test
```

모델을 동기화한 뒤 Python·Rust·Spring 샘플 결과와 신뢰도를 비교하세요. PT/JIT/ONNX는 학습 시점의 체크포인트가 다르면 서로 다른 예측을 낼 수 있으므로 export 후 동등성을 확인해야 합니다.

## 알려진 제약과 주의점

- ONNX 입력은 고정 차원입니다. 감지된 기본 이미지 크기와 export 차원이 다르면(`dev` 사례) ONNX가 거부합니다.
- 학습 종료 시 `model_full.pt`, JIT, ONNX가 서로 다른 가중치로 남을 수 있습니다. 배포 전 동일 체크포인트에서 재-export하고 `sync_models.py`를 다시 실행하세요.
- `core.py` import 시 CUDA/cuDNN을 탐색하고 전역 최적화를 설정합니다. 단순 import도 GPU probing을 일으킬 수 있습니다.
- FastAPI 모델 캐시는 프로세스 로컬입니다. 여러 worker는 각각 모델을 로드하며 파일 변경은 자동 반영되지 않습니다.
- `redistribute_train_pred`는 이미지를 이동하는 파괴적 작업입니다.
- 모델은 현재 CRNN 단일 아키텍처이며 고정 길이 레이블에 최적화되어 있습니다.

## 상세 문서

- [CRNN/CTC 상세](docs/crnn_ctc.md)
- [코드베이스 분석](docs/codebase-analysis.md)
- [Rust CLI](apps/cli/README.md)
- [Spring Boot](apps/springBoot/README.md)
- [에이전트·저장소 규약](AGENTS.md)
