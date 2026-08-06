# captcha-solver Spring Boot 서비스

저장소 루트의 FastAPI 웹 서비스(`web/`)를 Spring Boot 로 포팅한 것입니다.
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

> 상대 경로 기본값(`../../apps/cli/models`)을 쓰므로 **`apps/springBoot` 를 작업
> 디렉터리로 두고** 실행해야 합니다. 다른 곳에서 띄우려면 아래 설정을 절대경로로 덮어쓰세요.

---

## 설정

`src/main/resources/application.yml`

| 키 | 기본값 | 설명 |
|---|---|---|
| `captcha.models-dir` | `../../apps/cli/models` | `<id>.onnx` 와 `<id>.meta.json` 이 있는 디렉터리 |
| `captcha.db-path` | `../../db/captchaSolver.sqlite3` | 서비스 설정 SQLite 파일 |
| `captcha.schema-path` | `../../db/schema.sql` | 기동 시 멱등 적용 |
| `captcha.beam-width` | `10` | CTC prefix beam search 너비 |
| `captcha.intra-op-threads` | `1` | ONNX Runtime intra-op 스레드 |
| `server.port` | `5000` | |

서비스 대상 캡차와 기본 캡차는 DB 의 `service_captchas` 테이블에서 읽습니다.
테이블이 비어 있으면 `captcha.default-captcha-id` 하나만 서비스합니다.

---

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/` | 예측 화면 |
| GET | `/status` | 모델 상태 화면 |
| GET | `/health` | 서비스/로드 상태 |
| GET | `/ping`, `/version` | |
| POST | `/api/v1/predictImage` | multipart (`captcha_id`, `image`) |
| POST | `/api/v1/predictJson` | JSON (`captcha_id`, `image_data` = base64 또는 data URL) |

```json
{"captcha_id": "supreme_court", "prediction": "001741", "confidence": 0.9995, "elapsed_ms": 59}
```

오류는 FastAPI 와 같이 `{"detail": "..."}` 로 돌려줍니다 (400 / 500).

---

## 파이썬 구현과의 일치 검증

`apps/cli/samples` 의 서비스 대상 캡차 40장을 Rust CLI 와 대조했습니다.

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

- `/docs` (FastAPI 의 Swagger UI). springdoc 을 붙이면 되지만 Spring Boot 4 대응 버전
  확인이 필요해 넣지 않았습니다. 사이드바 링크도 빼 두었습니다.
- 학습 관련 기능. 이 서비스는 추론만 합니다.
