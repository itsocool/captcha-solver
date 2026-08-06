# captcha-cli

ONNX 모델로 캡차 이미지를 인식하는 크로스 플랫폼 CLI. Rust + [`ort`](https://github.com/pykeio/ort) + [`image`](https://crates.io/crates/image).

PyTorch·파이썬 런타임 없이 **실행 파일 하나 + 모델 파일**로 동작합니다. 스택 선정 배경은 [dev.md](dev.md) 참고.

---

## 사용법

```bash
captcha-cli -c supreme_court -i captcha.png
# 091082

captcha-cli -c kshop -i captcha.png --json
# {"captcha_id":"kshop","confidence":0.9944,"elapsed_ms":26,"prediction":"050862"}

cat captcha.png | captcha-cli -c gov24 -i -      # stdin 입력
captcha-cli --list                                # 사용 가능한 모델 목록
```

기본 출력은 개행 없이 예측 문자열만 내보내므로 셸 파이프라인에 바로 물릴 수 있습니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-c, --captcha-id` | `supreme_court` | 모델 디렉터리의 `<id>.onnx` 사용 |
| `-i, --image` | (필수) | 이미지 경로. `-`이면 stdin |
| `--models-dir` | 실행 파일 옆 `models/` | 모델 디렉터리 |
| `-m, --model` | — | ONNX 파일 직접 지정 (`--captcha-id` 무시) |
| `--meta` | `<모델>.meta.json` | 메타데이터 경로 |
| `--beam-width` | `10` | Beam Search 너비 |
| `--threads` | `1` | ONNX Runtime intra-op 스레드 수 |
| `--json` | off | JSON 출력 |
| `--list` | off | 모델 목록만 출력 |
| `--dump-input` | — | 전처리 텐서를 f32 LE로 덤프 (파이썬 대조용) |

지원 이미지 포맷: PNG, JPEG, GIF, BMP, WebP.

---

## 모델 준비

CLI는 ONNX만으로는 **문자셋·레이블 길이·전처리 종류**를 알 수 없습니다. 저장소 루트에서 아래를 실행하면 `models/`에 ONNX와 메타데이터가 함께 복사됩니다.

```bash
uv run python apps/cli/tools/sync_models.py            # 전체
uv run python apps/cli/tools/sync_models.py kshop      # 특정 캡차만
```

```
apps/cli/models/
├── supreme_court.onnx        # captcha_data/<id>/<rev>/model/model_full.pt.onnx 복사본
├── supreme_court.meta.json   # {image_width, image_height, label_length, characters, threshold, preprocess}
├── gov24.onnx
└── ...
```

모델을 재학습했다면 `sync_models.py`를 다시 돌려야 CLI에 반영됩니다.

---

## 샘플 이미지와 배치 확인

`samples/<captcha_id>/`에 캡차별 검증용 이미지 10장씩을 담아 두었습니다(파일명 = 정답).

```bash
./pred.sh                # supreme_court 샘플 전체 인식 + 정답 대조
./pred.sh kshop          # 다른 캡차
pred.cmd                 # Windows
```

```
이미지              정답        예측        신뢰도     시간
--------------------------------------------------------------
001741.png       001741     001741     0.9995     48 ms  O
...
supreme_court: 10/10 일치
```

모두 일치하면 종료 코드 0, 하나라도 틀리면 1입니다(CI에 그대로 물릴 수 있습니다).

---

## 빌드

타깃은 **Linux (x86_64/aarch64 glibc)** 와 **Windows 11 (x86_64)** 입니다.

### Linux

```bash
cd apps/cli
cargo build --release
# → target/release/captcha-cli (약 24MB, ONNX Runtime 정적 링크)
```

`ldd`로 확인하면 `libstdc++`, `libc`만 참조합니다. **`libonnxruntime.so`를 따로 배포할 필요가 없습니다.**

> musl 타깃용 프리빌트 ONNX Runtime은 제공되지 않습니다. Alpine이 필요하면 ONNX Runtime을 직접 빌드해 `ORT_LIB_PATH`로 지정해야 합니다.

### Windows 11

Windows에서 직접 빌드 (권장):

```powershell
# Visual Studio Build Tools(MSVC) + Rust 설치 후
cd apps\cli
cargo build --release
# → target\release\captcha-cli.exe
```

Linux에서 크로스 빌드하려면 [`cargo-xwin`](https://github.com/rust-cross/cargo-xwin)을 씁니다.

```bash
cargo install cargo-xwin
rustup target add x86_64-pc-windows-msvc
cargo xwin build --release --target x86_64-pc-windows-msvc
```

> Windows용 프리빌트에는 DirectML 등이 포함되어 있어 빌드 산출물 디렉터리에 `*.dll`이 함께 생성될 수 있습니다. 생성되었다면 **exe와 같은 폴더에 담아 배포**하세요. (이 저장소에서는 Linux 빌드만 실제로 검증했습니다.)

### 배포 패키지 만들기

```bash
tools/package.sh
# → dist/captcha-cli-0.1.0-linux-x86_64.tar.gz (약 60MB)
```

실행 파일 + `models/` + `samples/` + `README.md` + `pred.sh`/`pred.cmd`를 묶습니다.
압축을 풀면 그 자리에서 바로 동작합니다.

```bash
tar -xzf captcha-cli-0.1.0-linux-x86_64.tar.gz
cd captcha-cli-0.1.0-linux-x86_64
./pred.sh          # supreme_court: 10/10 일치
```

용량 대부분은 모델(6종 55MB)입니다. 서비스 대상만 담으려면 패키징 전에 불필요한
`models/<id>.onnx`, `models/<id>.meta.json`을 지우면 됩니다.

### 배포 레이아웃

```
captcha-cli(.exe)
models/
├── supreme_court.onnx
├── supreme_court.meta.json
└── ...
```

`--models-dir`를 주지 않으면 **실행 파일과 같은 폴더의 `models/`**를 먼저 찾고, 없으면 개발 편의를 위해 크레이트의 `models/`로 폴백합니다.

---

## 파이썬 구현과의 동등성

전처리(`dataclass.py: image_pre_process`)와 디코딩(`core.py: ctc_beam_decode_fixed_length`)을 그대로 포팅했습니다. PIL 동작 중 재현이 까다로운 부분:

- `convert("L")` — ITU-R 601-2 고정소수점 luma (`(R*19595 + G*38470 + B*7471 + 32768) >> 16`)
- `Image.resize()` 기본 필터 — **BICUBIC**(a=-0.5) → `FilterType::CatmullRom`
- `Image.new("RGBA", size, 255)` — 흰색이 아니라 `(255, 0, 0, 0)` (supreme_court 캔버스 여백)

### 검증

```bash
cargo test                                                    # 디코더 단위 테스트 6개
uv run python apps/cli/tools/compare_with_python.py --limit 100   # 파이썬 결과와 대조
```

실측 결과 (각 100장):

| captcha | 문자열 일치 | 신뢰도 최대 오차 |
|---------|-------------|------------------|
| supreme_court | 100/100 | 0.0011 |
| gov24 | 100/100 | 0.0066 |
| wetax | 100/100 | 0.0019 |
| kshop | 70/100 | 0.7757 |

`samples/` 10장 기준으로는 supreme_court·gov24·wetax·kshop 모두 10/10입니다.

### dev 모델은 현재 ONNX로 추론할 수 없습니다

`dev`는 ONNX가 200×50으로 export되어 있는데 메타데이터(학습 이미지에서 감지한 크기)는 250×50입니다.
`build_model()`이 생성자 기본값(200)으로 모델을 만들고 전처리는 감지값(250)으로 리사이즈하기 때문인데,
PyTorch는 CNN이 폭에 유연해서 그냥 동작하지만 ONNX는 입력 크기가 고정이라 거부합니다.
CLI는 이 경우 아래처럼 원인을 짚어 줍니다.

```
Error: 모델 입력 크기(200×50)와 메타데이터 크기(250×50)가 다릅니다.
메타데이터를 고치거나 해당 크기로 ONNX를 다시 export하세요.
```

서비스 대상 4종(supreme_court, gov24, wetax, kshop)은 영향이 없습니다.

전처리 텐서 자체의 차이는 픽셀당 최대 0.10, 평균 0.0004~0.0015 수준입니다(리사이즈 필터 구현 차이).

### kshop 불일치의 원인 — 포팅 문제가 아닙니다

**`model_full.pt`와 `model_full.pt.onnx`의 가중치가 서로 다릅니다.** 동일한 입력 텐서를 두 아티팩트에 넣으면 결과가 갈립니다.

```
정답(파일명): 050862
PyTorch(model_full.pt)  → 050186 (0.59)
Rust CLI(onnx)          → 050862 (0.99)
```

`core.py: train_model`이 학습 종료 시 베스트 체크포인트(`.tmp`)를 `.pt`로 승격한 뒤, JIT/ONNX는 **메모리에 남아 있는 마지막 에폭 가중치**로 export하기 때문입니다. kshop 100장 기준 정확도는 파이썬(.pt) 64%, CLI(.onnx) 86%로 오히려 ONNX 쪽이 좋습니다.

정합성을 맞추려면 export 직전에 베스트 가중치를 다시 로드해야 합니다.
