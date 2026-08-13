# captcha-solver Spring Boot 서비스

저장소 루트의 FastAPI 웹 서비스(`apps/web/`)를 Spring Boot 로 포팅한 것입니다.
API 경로·요청/응답 형태·화면 구성이 모두 동일합니다.

추론만 다릅니다. 파이썬은 PyTorch 로 `model_full.pt` 를 읽지만, JVM 에서는 그 파일을
읽을 수 없어 **`apps/cli` 가 쓰는 것과 같은 ONNX 모델**을 ONNX Runtime 으로 돌립니다.
전처리와 CTC 디코딩은 `apps/cli/src/{preprocess,decode}.rs` 를 그대로 옮겼습니다.

- JDK 25 / Maven / Spring Boot 4.1 / SQLite (sqlite-jdbc)
- 모델·DB 는 복사하지 않고 저장소 자산을 그대로 참조합니다

---

## 실행

```powershell
$env:JAVA_HOME = 'C:\dev\sdk\jdk-25.0.3-full'
mvn spring-boot:run
# 또는
mvn -DskipTests package
java -jar target\captchaSolver-0.0.1-SNAPSHOT.jar
```

기본 포트는 5000 이며 <http://localhost:5000> 에서 예측 화면이 열립니다.

> 상대 경로 기본값(`../../models`)을 쓰므로 **`apps/springBoot` 를 작업
> 디렉터리로 두고** 실행해야 합니다. 다른 곳에서 띄우려면 아래 설정을 절대경로로 덮어쓰세요.

`mvn` 을 돌리면 `generate-resources` 단계에서 `tools/SyncResource.java` 가 저장소 루트
`captcha_data/<id>/<rev>/model/` 의 리비전 중 **번호가 가장 큰 것**을 골라 `model.ort`·
`model.meta.json` 을 `<id>.ort`·`<id>.meta.json` 으로 이름 바꿔 `models/`(pom.xml 과 같은
위치)에, 저장소 루트 `db/` 를 파일명 그대로 `db/`(마찬가지로 pom.xml 과 같은 위치)에
복사합니다 — 둘 다 `target/` 이 아니라서 `mvn clean` 으로도 안 지워집니다
(`models/` 는 `apps/cli/tools/sync_models.py` 의 Java 버전 — Python 없이 동작). 출력
디렉터리가 없으면 만들고, 있으면 파일별로 크기·수정시각이 같은 한 다시 복사하지 않습니다
(수 MB 짜리 모델을 매 빌드마다 반복 복사하지 않기 위해서). `captcha_data`/`db` 가 없으면
(예: Docker 빌드 컨텍스트 안) 그 항목만 조용히 건너뜁니다. **주의**: "가장 큰 리비전 번호"
는 학습만 하고 아직 서비스로 승격 안 한 리비전을 잘못 집어올 수 있습니다 — `apps/web` 쪽
`engine.py` 레지스트리가 실제 서비스 리비전을 정하는 것과는 별개의 휴리스틱입니다.

---

## Docker

```bash
# 1) 배포용 모델(<id>.ort/<id>.meta.json)·DB 를 apps/springBoot/{models,db}/ 에 생성
#    (최초 1회 · 모델/DB 갱신 시)
cd apps/springBoot
mvn generate-resources

# 2) 빌드 + 기동
docker compose up --build -d
```

`http://localhost:8080` 에서 예측 화면이 열립니다 (컨테이너 안에서는 `SERVER_PORT=8080` 으로
로컬 실행 기본값 5000 과 다르게 띄웁니다). 모델·DB 모두 **이미지 빌드 시점에
`apps/springBoot/{models,db}/` 를 그대로 구워 넣습니다** (`Dockerfile` 의
`COPY models ./models`, `COPY db ./db`) — 볼륨 마운트가 아니라서 바꾸려면 위 1번을 다시
돌리고 재빌드해야 반영됩니다. 컨테이너 안에서는
`CAPTCHA_MODELS_DIR`/`CAPTCHA_DB_PATH`/`CAPTCHA_SCHEMA_PATH` 환경변수(Dockerfile 에서
`/app/models`, `/app/db/...` 로 고정)가 `application.yml` 의 상대경로 기본값을 덮어씁니다.
`db/captchaSolver.sqlite3` 는 기동 시 `schema.sql` 을 멱등 적용하느라 컨테이너 안에서
쓰기가 필요한데, 컨테이너의 쓰기 가능 레이어에만 반영되고 이미지·호스트 파일은 안 바뀝니다.

로그만 `./logs`(즉 `apps/springBoot/logs/`)에 볼륨 마운트로 남습니다 — `captchaSolver.log`
는 컨테이너를 지워도 남습니다.

컨테이너는 root 가 아니라 베이스 이미지에 있는 `ubuntu`(uid/gid 1000)로 뜹니다. 리눅스
개발 호스트의 일반 사용자 UID 도 보통 1000이라 `logs` 볼륨 권한 문제가 잘 안 생깁니다.
호스트 UID 가 다르면 `docker-compose.yml` 에 `user: "<uid>:<gid>"` 를 추가하세요.

---

## 설정

`src/main/resources/application.yml`

| 키 | 기본값 | 설명 |
|---|---|---|
| `captcha.models-dir` | `../../models` | `<id>.onnx` 와 `<id>.meta.json` 이 있는 디렉터리 |
| `captcha.db-path` | `../../db/captchaSolver.sqlite3` | 서비스 설정 SQLite 파일 |
| `captcha.schema-path` | `../../db/schema.sql` | 기동 시 멱등 적용 |
| `captcha.beam-width` | `10` | CTC prefix beam search 너비 |
| `captcha.intra-op-threads` | `1` | ONNX Runtime intra-op 스레드 |
| `server.port` | `5000` | |
| `logging.file.name` | `logs/captchaSolver.log` | 콘솔과 별개로 파일에도 남긴다 (롤링 10MB · 7일치, Spring Boot 기본값) |

서비스 대상 캡차와 기본 캡차는 DB 의 `service_captchas` 테이블에서 읽습니다.
테이블이 비어 있으면 `captcha.default-captcha-id` 하나만 서비스합니다.

---

## API

추론 전용 서버이므로 예측 API 와 그것을 테스트하는 화면만 있다.

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/` | 예측 테스트 화면 |
| GET | `/docs` | Swagger UI (`/swagger-ui/index.html` 로 리다이렉트) |
| GET | `/openapi.json` | OpenAPI 3.1 문서 |
| POST | `/api/v1/predictImage` | multipart (`captcha_id`, `image`) |
| POST | `/api/v1/predictJson` | JSON (`captcha_id`, `image_data` = base64 또는 data URL) |

```json
{"captcha_id": "supreme_court", "prediction": "001741", "confidence": 0.9995, "elapsed_ms": 59}
```

오류는 FastAPI 와 같이 `{"detail": "..."}` 로 돌려줍니다 (400 / 500).

### `/docs`

springdoc-openapi 3.x(Spring Boot 4 대응 라인)를 씁니다. 2.x 는 Spring Framework 6 용이라
기동하지 않습니다. 경로는 `application.yml` 의 `springdoc.*` 에서 FastAPI 와 같게 맞춰 두었습니다.

문서를 실제 응답과 맞추기 위해 `config/OpenApiConfig` 에서 두 가지를 손봅니다.

- **스키마 이름** — 런타임 직렬화는 Jackson 3 이 `spring.jackson.property-naming-strategy` 로
  처리하지만 swagger-core 는 자체 Jackson 2 매퍼를 써서 그 설정이 닿지 않습니다. 그대로 두면
  문서만 `captchaId` 로 나옵니다. 그래서 `ModelResolver` 를 snake_case 로 등록합니다.
- **`type: object`** — 위 `ModelResolver` 를 새 `ObjectMapper` 로 만들면 OpenAPI 3.1 설정이
  빠져 스키마에서 `type` 이 사라집니다. springdoc 의 매퍼를 복사해서 써야 합니다.

`predictImage` 의 `captcha_id` 는 springdoc 기본값으로는 query 파라미터로 문서화되는데,
FastAPI 는 `Form(...)` 으로 받으므로 `PredictImageForm` 스키마를 직접 줘서 multipart 필드로
표시합니다(런타임 바인딩은 그대로입니다).

`OpenApiDocsTests` 가 위 네 가지를 모두 검증합니다.

---

## 파이썬 구현과의 일치 검증

서비스 대상 캡차 40장을 Rust CLI 와 대조했습니다.

| | 결과 |
|---|---|
| 정답률 | 40 / 40 |
| Rust CLI 와 예측 일치 | 40 / 40 |
| 신뢰도 최대 오차 | 1.7e-7 |

전처리 텐서도 대부분 비트 단위로 같습니다(1픽셀 ±1/255 반올림 타이 제외).

포팅에서 특히 조심한 곳:

- **그레이스케일 원시값** — `BufferedImage.getRGB()` 는 GRAY 색공간을 sRGB 로 변환해
  값을 바꿉니다(85 → 117). 래스터에서 직접 읽어야 PIL/Rust 와 같아집니다.
- **리샘플링 정밀도** — `image` crate 의 `resize` 는 세로 패스 결과를 f32 로 들고 있다가
  가로 패스에서만 u8 로 반올림합니다. 중간에 u8 로 접으면 픽셀당 최대 20/255 어긋납니다.
- **BICUBIC 커널** — PIL 기본 필터는 Catmull-Rom(a = -0.5)입니다.

---

## 아직 없는 것

- FastAPI 의 `/docs` 는 프런트 라우트(`/`)까지 목록에 넣지만, springdoc 은
  `@RestController` 만 훑어서 API 만 나옵니다.
- 학습 관련 기능, 헬스체크/상태 조회 API. 이 서비스는 추론과 그 테스트 화면만 제공합니다.
