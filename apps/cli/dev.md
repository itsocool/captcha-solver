# ONNX 캡차 추론 CLI — 개발 스택 검토

이 저장소가 내보내는 ONNX 모델을 로드해, 이미지 한 장을 받아 캡차 문자열을 출력하는 **크로스 플랫폼 CLI**를 만들기 위한 스택 선정 문서입니다.

> **결론: Rust + `ort` + `image`로 구현 완료.** 사용법·빌드·검증 결과는 [README.md](README.md) 참고.
> 타깃은 Linux / Windows 11이며 Linux 빌드는 단일 바이너리 24MB(ONNX Runtime 정적 링크)로 검증했습니다.

---

## 1. 만들 것

```
$ captcha-cli --model supreme_court.onnx --image captcha.png
091082

$ captcha-cli -m supreme_court.onnx -i captcha.png --json
{"prediction":"091082","confidence":0.9999,"elapsed_ms":8}
```

요구 사항

| 항목 | 내용 |
|------|------|
| 입력 | 이미지 파일 경로(또는 stdin), 모델 경로/ID |
| 출력 | 예측 문자열 (옵션: JSON에 신뢰도·소요시간) |
| 플랫폼 | Windows / Linux / macOS (x64, arm64) |
| 배포 | 런타임 설치 없이 실행 가능한 단일 실행 파일 선호 |
| 성능 | 프로세스 기동 포함 100ms 내외 (배치 호출 시 더 중요) |

---

## 2. 먼저 따질 것: 새로 만들 필요가 있나?

이미 `main.py` + `main.spec`(PyInstaller)로 CLI를 빌드하는 경로가 있습니다. 다만 그 경로는 **PyTorch 전체를 번들**하므로 산출물이 수백 MB~GB급이 됩니다.

새 CLI의 실익은 언어 교체가 아니라 **런타임 축소**입니다.

- PyTorch 제거 → ONNX Runtime만 사용 (모델 파일 ~9MB, 런타임 ~15MB)
- 파이썬을 유지하더라도 `onnxruntime + pillow + numpy`만 번들하면 60~90MB 수준으로 줄어듭니다.

**즉, 가장 싼 1단계는 "언어 교체"가 아니라 "PyTorch 의존 제거"입니다.** 그래도 배포 크기·기동 속도가 부족할 때 네이티브 스택으로 넘어가면 됩니다.

---

## 3. 포팅해야 하는 3가지

ONNX 파일만 있으면 되는 게 아닙니다. 파이썬 쪽 전처리와 디코딩을 **바이트 단위로 동일하게** 재현해야 같은 결과가 나옵니다.

### 3.1 모델 시그니처 (실측: `captcha_data/supreme_court/0/model/model.onnx`)

| 항목 | 값 |
|------|-----|
| opset / IR | 17 / 8 (producer: pytorch 2.12) |
| 입력 | `input` · float32 · `[1, 1, 40, 120]` (N, C, H, W), 값 범위 0.0~1.0 |
| 출력 | `output` · float32 · `[30, 1, 11]` (T, N, C) |
| **출력은 로짓입니다** | `log_softmax(axis=2)`를 CLI에서 직접 적용해야 합니다 |
| 배치 | 1로 고정 (`fixed_batch=True`로 export됨) |
| 파일 크기 | 캡차당 8.8~9.3MB |

입력 H/W와 출력 T/C는 캡차마다 다릅니다(예: kshop 263×54). 세션 로드 후 **입력/출력 shape를 읽어서** 쓰는 편이 안전합니다.

### 3.2 전처리 (`dataclass.py: image_pre_process`)

공통 경로:

1. RGBA면 흰 배경에 합성 → RGB
2. 그레이스케일 `L` 변환
3. `0 < threshold < 255`면 `p > threshold ? 255 : p` (gov24는 60)
4. 테두리 2px 크롭
5. `p > 128`인 픽셀을 255로
6. `(W, H)`로 리사이즈
7. `/255.0` float32, `(1,1,H,W)`로 reshape

`supreme_court`는 전용 경로: `crop(3, 1, W-1, H-7)` → `(W,H)` 흰 캔버스에 `(1,1)` 위치로 붙이기 → 위 3~6 수행.

**포팅 함정 두 가지 (실측 확인함)**

- **그레이스케일 계수**: PIL의 `convert("L")`은 ITU-R 601-2 luma — `L = 0.299R + 0.587G + 0.114B` 후 내림. 다른 계수를 쓰면 예측이 흔들립니다.
- **리사이즈 필터**: PIL `Image.resize()`의 기본값은 **BICUBIC**입니다 (Pillow 12 기준). Go/Rust 이미지 라이브러리의 기본값(대개 NEAREST/BILINEAR)을 그대로 쓰면 결과가 달라집니다. Catmull-Rom(a=-0.5) 계열 bicubic으로 맞춰야 합니다.

### 3.3 디코딩 (`core.py: ctc_beam_decode_fixed_length`)

고정 길이 CTC prefix beam search (약 80줄). prefix별로 `p_blank`/`p_nonblank`를 log-sum-exp로 합산하고, 기대 길이를 하드 제약으로 씁니다. 신뢰도는 최종 후보 집합에서 정규화한 사후확률입니다.

- 클래스 0 = blank, 문자 인덱스는 1부터. 문자 순서는 **학습 이미지 파일명에서 추출한 문자들을 정렬한 것**(`sorted(set(chars))`)입니다.
- 그리디(argmax + 반복/blank 제거) 디코딩으로 대체하면 코드는 20줄로 줄지만, **신뢰도 값이 달라지고** 애매한 샘플의 정확도가 떨어집니다. 신뢰도를 임계값으로 쓸 계획이면 beam search를 그대로 포팅하세요.

### 3.4 메타데이터 사이드카 (필수 설계)

문자셋·레이블 길이·threshold·전처리 종류는 지금 파이썬 객체 안에만 있습니다. CLI가 자립하려면 ONNX 옆에 JSON을 함께 내보내야 합니다.

```json
{
  "captcha_id": "supreme_court",
  "rev": 0,
  "image_width": 120,
  "image_height": 40,
  "label_length": 6,
  "characters": "0123456789",
  "threshold": 255,
  "preprocess": "supreme_court",
  "blank_index": 0
}
```

`TrainData`가 이미 전부 가지고 있으므로 export 시점에 한 번 덤프하면 됩니다. (또는 ONNX `metadata_props`에 넣어 파일 하나로 유지하는 방법도 있습니다 — 배포 단순성 면에서 이쪽이 낫습니다.)

---

## 4. 스택 비교

| 스택 | ONNX 바인딩 | 이미지 처리 | 산출물(대략) | 크로스 빌드 | 리스크 |
|------|-------------|-------------|--------------|-------------|--------|
| **Rust** | `ort` 2.x (pykeio) | `image` crate | 단일 바이너리 20~30MB | `cargo-zigbuild` / CI 매트릭스 | 러닝커브, 빌드 시간 |
| **Go** | `onnxruntime_go` (yalue, cgo) | `image` + `x/image/draw` | 바이너리 10MB + **libonnxruntime 동봉 필수** | cgo라 타깃별 빌드 필요 | 단일 파일 아님, cgo 크로스컴파일 번거로움 |
| **C# / .NET** | `Microsoft.ML.OnnxRuntime` (MS 공식) | `ImageSharp` | self-contained 30~60MB | `dotnet publish -r <rid>` | NativeAOT 조합은 별도 검증 필요 |
| **Python** | `onnxruntime` | `Pillow` (동일 코드 재사용) | PyInstaller 60~90MB | 타깃 OS에서 각각 빌드 | 기동 0.3~1s, 번들 크기 |
| **C++** | ORT C/C++ API | `stb_image` 등 | 가장 작음 | 툴체인 3종 관리 | 개발·유지 비용 최대 |
| **Node/Bun** | `onnxruntime-node` | `sharp` | 80~120MB | 네이티브 모듈 재빌드 | 두 개의 네이티브 의존성 |

> 크기는 CPU 전용 ONNX Runtime 기준 대략치입니다. 실제 값은 타깃과 링크 방식에 따라 달라집니다.

---

## 5. 추천

### 1순위 — Rust + `ort` + `image`

단일 바이너리 배포가 목표라면 이 조합이 가장 깔끔합니다.

- `ort`는 `download-binaries` 기능으로 프리빌트 ONNX Runtime을 받아 링크합니다. 별도 설치 없이 바이너리 하나로 끝납니다.
- 런타임 위치를 유연하게 가져가고 싶으면 `load-dynamic` 기능 + `ORT_DYLIB_PATH`로 전환할 수 있습니다.
- 정적 링크가 필요하면 `ORT_LIB_PATH`로 직접 빌드한 정적 라이브러리를 지정합니다.
- 기동 시간이 짧아(수 ms) 배치 호출·외부 프로그램에서 반복 실행하는 형태에 유리합니다.

```toml
[dependencies]
ort = "2.0.0-rc.10"          # ONNX Runtime 바인딩
image = "0.25"               # 디코드/그레이스케일/리사이즈
clap = { version = "4", features = ["derive"] }
serde_json = "1"             # 사이드카 메타 + --json 출력
```

리사이즈는 `image::imageops::resize(..., FilterType::CatmullRom)`으로 PIL BICUBIC에 맞춥니다.

### 2순위 — Go + `onnxruntime_go`

팀이 이미 Go를 쓰고 있고 "실행 파일 + 공유 라이브러리 1개" 배포가 허용된다면 개발 속도가 가장 빠릅니다. 단 cgo 때문에 **타깃 OS/아키텍처마다 빌드해야 하고**, `libonnxruntime.{so,dylib,dll}`을 함께 배포해야 합니다. "단일 파일" 요구가 있으면 탈락입니다.

### 특수 상황

- **배포 대상이 Windows 중심 + .NET 자산 보유** → C# + `Microsoft.ML.OnnxRuntime`. 공식 패키지라 지원이 가장 안정적입니다.
- **가장 빨리 결과가 필요함** → 파이썬에서 PyTorch만 걷어낸 ONNX 전용 CLI. 전처리·디코딩 코드를 그대로 재사용하므로 포팅 리스크가 0이고, 이후 네이티브 포팅의 **정답지(골든 데이터) 생성기**로도 씁니다.

---

## 6. 권장 진행 순서

1. **메타데이터 사이드카 export 추가** (`export_onnx` 시 JSON 동시 생성 또는 `metadata_props` 삽입)
2. **파이썬 ONNX 전용 CLI** 작성 — PyTorch 없이 `onnxruntime`만으로 동일 결과가 나오는지 확인
3. **골든 데이터 생성** — `captcha_data/*/0/images/pred`의 200~500장에 대해 `{파일명, 예측, 신뢰도}` JSON 덤프
4. **네이티브 CLI 구현** (Rust 권장)
5. **동등성 검증** — 골든 데이터와 대조. 문자열 100% 일치, 신뢰도 오차 1e-3 이하를 합격선으로
6. **CI 릴리스 매트릭스** — linux-x64 / linux-arm64 / windows-x64 / macos-arm64

3단계를 건너뛰면 "왜 파이썬과 다른 글자가 나오지"를 눈으로 디버깅하게 됩니다. 전처리 차이(리사이즈 필터·luma 계수)는 육안으로 안 보이고 정확도로만 드러납니다.

---

## 7. 검증 체크리스트

- [ ] 전처리 결과 텐서를 파이썬과 바이트 비교 (`--dump-input` 플래그로 float32 배열 덤프 후 `np.allclose`)
- [ ] 로짓 출력 비교 (같은 입력 → 최대 오차 1e-4 이하)
- [ ] 디코딩 결과 문자열·신뢰도 비교 (골든 데이터 전량)
- [ ] threshold가 걸린 캡차(gov24, 60)와 전용 전처리 캡차(supreme_court) 각각 확인
- [ ] 이미지 포맷: PNG / JPG / RGBA PNG / 그레이스케일 PNG 각각 확인
- [ ] 손상된 이미지·존재하지 않는 경로에 대한 종료 코드 정의

---

## 8. 참고

- ort (Rust) 링킹 옵션: https://github.com/pykeio/ort/blob/main/docs/content/setup/linking.mdx
- ONNX Runtime: https://github.com/microsoft/onnxruntime
- Microsoft.ML.OnnxRuntime (NuGet): https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime
- 이 저장소의 참조 구현: `core.py`(`export_onnx`, `ctc_beam_decode_fixed_length`), `dataclass.py`(`image_pre_process`)
