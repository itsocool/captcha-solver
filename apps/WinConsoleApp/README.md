# WinConsoleApp — captcha (C++ / CMake, Windows + Linux)

**네이티브 C++** 콘솔 실행 파일.
`gov24.cmd` / `supreme_court.cmd` 를 그대로 쓸 수 있는 CLI 문법과 출력 형식입니다.

.NET 런타임 없이 **실행 파일 + `<id>.ort` + `<id>.meta.json`** 만으로 동작합니다
(Windows 는 ONNX Runtime 이 실행 파일에 내장돼 있고, Linux 는 `libonnxruntime.so` 가 옆에 놓입니다).

**Windows x64 / Linux x64·aarch64 모두 빌드**됩니다
(경로를 `std::filesystem::path`로만 다뤄서, `path::value_type`이 ONNX Runtime의 `ORTCHAR_T`와 그대로 맞습니다).
Linux만 필요하다면 `apps/cli`(Rust)가 이미 같은 파이프라인을 제공합니다 — 이쪽은 C++ 툴체인만 있는 환경용입니다.

---

## 사용법

```cmd
captcha.exe -c="gov24" -i="gov24.JPG"
:: 004657   (개행 없이 예측 문자열만)

captcha.exe -c="gov24" -i="a.png" -m="D:\models\gov24.ort" --meta="D:\models\gov24.meta.json"

:: ONNX -> ORT 변환 (학습 파이프라인 밖에서 모델을 받았을 때)
captcha.exe --to-ort gov24.onnx gov24.ort
```

```bash
./captcha -c=gov24 -i=captcha.png     # Linux, 동일한 문법
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-c`, `--captcha-id` | `supreme_court` | 캡차 ID |
| `-i`, `--image-path` | (필수) | 이미지 경로 |
| `-m`, `--model-path` | `<실행 파일 폴더>/<id>.ort` | 모델. 확장자로 포맷을 가리므로 `.onnx` 도 그대로 받습니다 |
| `--meta` | `<모델>.meta.json`, 없으면 `<실행 파일 폴더>/<id>.meta.json` | 메타데이터 |

`-c=값` / `-c 값` 두 형식 모두 받습니다. 지원 이미지: PNG, JPEG, BMP, GIF, TGA, PSD, HDR, PNM.

종료 코드: `0` 성공 / `2` 모델·메타 파일 없음 / `1` 그 외 오류.
인자가 누락되면 stderr 에 안내를 찍고 `1` 을 반환합니다.

---

## 빌드

CMake 3.24+ 와 C++17 컴파일러(**MSVC v143** 또는 **GCC 9+ / Clang 9+** — `<filesystem>` 때문)가 필요합니다.
**최초 configure 때 네트워크**를 씁니다 (ONNX Runtime 프리빌트, stb 헤더를 FetchContent로 받음).

### Windows — PowerShell 스크립트 (권장)

```powershell
cd apps\WinConsoleApp
.\build.ps1          # configure -> build 전부
```

| 스크립트 | 하는 일 |
|----------|---------|
| `build.ps1` | cmake를 찾아 PATH에 얹고 configure 후 빌드 |
| `pack.ps1` | `build.ps1` 호출 후 NSIS 설치 파일 생성, `packages\WinConsoleApp\` 에 배치 |

CMake가 PATH에 없어도 됩니다 — `build.ps1`이 `vswhere`로 Visual Studio를 찾아 번들 CMake를
**이 프로세스의 PATH에만** 추가합니다(전역 PATH는 건드리지 않음).

```powershell
.\build.ps1 -Config Debug     # Debug 빌드
.\build.ps1 -Clean            # build/ 지우고 처음부터 (제너레이터 변경 시 필요)
```

제너레이터는 지정하지 않아 CMake가 **설치된 최신 Visual Studio**를 고릅니다. VS를 업그레이드해도
스크립트를 고칠 필요가 없습니다.

### Windows — 직접 실행

```cmd
cd apps\WinConsoleApp
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
:: -> build\Release\captcha.exe + <id>.ort + <id>.meta.json
```

Windows 빌드는 기본이 **단일 exe**입니다. `onnxruntime.dll` 을 LZMS 로 압축해 exe 리소스에 넣고
(11.5MB -> 3.6MB), CRT 까지 정적 링크하므로 VC++ 재배포 패키지도 필요 없습니다. 실행 시
`%LOCALAPPDATA%\captcha-solver\ort-<크기>\onnxruntime.dll` 로 한 번 풀고 그 뒤로는 재사용합니다.
DLL 을 옆에 두는 예전 방식이 필요하면 `-DCAPTCHA_SINGLE_EXE=OFF`.

VS에 딸린 CMake는 `%ProgramFiles(x86)%\Microsoft Visual Studio\<버전>\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin` 에 있습니다.

### Linux — 직접 실행

```bash
cd apps/WinConsoleApp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# -> build/captcha + libonnxruntime.so* + <id>.ort + <id>.meta.json
```

실행 파일에 `RUNPATH=$ORIGIN` 이 박히고 `libonnxruntime.so*` 가 옆에 복사되므로, **디렉터리째 옮겨도 그대로 동작**합니다.
(`captcha` + `libonnxruntime.so.1` + `<id>.ort` + `<id>.meta.json` 만 있으면 됩니다.)

### 공통 옵션

ONNX Runtime 버전은 `-DORT_VERSION=1.20.1` 로 바꿀 수 있습니다(모델은 opset 17 / IR 8이라 1.13 이상이면 됩니다).
`-DCAPTCHA_COPY_MODELS=OFF` 를 주면 모델 복사를 생략하고,
`-DCAPTCHA_DATA_DIR=<경로>` 로 모델을 가져올 `captcha_data` 위치를 바꿉니다(기본값 `<저장소 루트>/captcha_data`).
`-DCAPTCHA_CONFIG_FILE=<경로>` 는 패키징 설정 JSON 입니다(기본값 `config.json`, 아래 [담을 캡차 유형 고르기](#담을-캡차-유형-고르기)).
`-DCAPTCHA_SINGLE_EXE=OFF` (Windows 전용, 기본 ON) 로 하면 DLL 내장 대신 실행 파일 옆에 복사합니다.

### Windows — 설치 파일 (NSIS)

```powershell
cd apps\WinConsoleApp
.\pack.ps1           # 빌드 -> packages\WinConsoleApp\captcha-solver-1.0.0-win64.exe + config.json
```

CPack의 NSIS 제너레이터를 씁니다. **NSIS가 필요합니다** — 없으면 `pack.ps1`이 설치 방법을 알려줍니다
(`winget install NSIS.NSIS`). CPack을 직접 부르려면 빌드 후:

```cmd
cpack --config build\CPackConfig.cmake -C Release -B build
```

설치본은 `%ProgramFiles%\captcha-solver\bin\` 에 `captcha.exe` + `<id>.ort` + `<id>.meta.json`
(단일 exe가 아니면 `onnxruntime.dll` 도)을 나란히 놓습니다 — 실행 파일이 모델을 자기 폴더에서 찾기
때문에 이 배치가 그대로 실행 조건입니다. 설치 마법사가 `bin` 을 PATH에 추가할지 묻고,
제어판에서 제거할 수 있습니다. 모델은 configure 시점에 잡힌 것이 들어가므로,
`-DCAPTCHA_COPY_MODELS=OFF` 면 실행 파일만 담깁니다.

#### 담을 캡차 유형 고르기

어떤 유형을 설치본에 넣을지는 **`config.json`** 의 `captchas` 배열이 정합니다.
값은 `captcha_data/<id>/` 디렉터리 이름이고, 각 유형의 최신 리비전 모델이 들어갑니다.

```json
{
  "captchas": ["gov24", "iptime"]
}
```

파일을 고치면 다음 `pack.ps1` 실행 때 자동으로 반영됩니다(configure 가 이 파일을 감시합니다).
**`config.json` 이 없으면 `supreme_court` 하나만** 담습니다. 다른 파일을 쓰려면
`-DCAPTCHA_CONFIG_FILE=<경로>` 입니다.

목록에 적은 유형의 모델이 `captcha_data` 에 없으면 configure 가 **실패**합니다 — 유형이 빠진
설치본이 조용히 나가는 것보다 낫기 때문입니다.

버전은 `CMakeLists.txt` 의 `project(... VERSION 1.0.0 ...)` 하나로 정해집니다.

### 대량 정확도 (참고 실측)

`captcha_data/<id>/<rev>/images/pred/` 에서 각 100장을 무작위로 뽑아 파일명(정답)과 대조한 결과입니다.

| captcha | 일치 |
|---------|------|
| supreme_court | 100/100 |
| gov24 | 100/100 |
| wetax | 100/100 |
| default | 100/100 |

---

## 모델과 메타데이터

`.ort` 는 **ORT 포맷**입니다 — ONNX Runtime 이 그래프 최적화까지 마친 결과를 flatbuffer 로 구운 것으로,
세션을 열 때 최적화를 다시 돌리지 않습니다. 학습 쪽 `finalize_artifacts()` 가 `model.ort` 를 만들고,
**빌드는 그 파일을 이름만 바꿔 복사**합니다.

> 학습 파이프라인 밖에서 ONNX 만 받았다면 `captcha --to-ort <in.onnx> <out.ort>` 로 직접 변환할 수 있습니다.
> 최적화 수준은 양쪽 다 `ORT_ENABLE_EXTENDED` 까지입니다. `ORT_ENABLE_ALL` 은 레이아웃 최적화(NCHWc)를
> 포함해 **변환한 기계의 CPU 명령셋에 맞춰 굳으므로**, 다른 CPU 로 배포할 파일에는 쓸 수 없습니다.

모델만으로는 **문자셋·레이블 길이·전처리 종류**를 알 수 없어 사이드카 JSON이 함께 있어야 합니다.

```json
{ "captcha_id": "gov24", "image_width": 200, "image_height": 50,
  "label_length": 6, "characters": "0123456789", "threshold": 60, "preprocess": "default" }
```

학습 쪽 `finalize_artifacts()` 가 `model.meta.json` 으로 저장합니다.

빌드는 **`captcha_data/<id>/<rev>/model/`** 에서 `model.ort` 와 `model.meta.json` 을 가져와
`<id>.ort` / `<id>.meta.json` 으로 이름만 바꿔 놓습니다. `<rev>` 는 숫자 디렉터리 중 **가장 큰 것**을
고르고, 두 파일 중 하나라도 없으면 그 캡차는 건너뜁니다(configure 로그에 표시).
캡차를 추가하면 다음 빌드에 자동으로 딸려옵니다(`file(GLOB CONFIGURE_DEPENDS)`).

```
-- 복사할 모델: gov24(rev=1);iptime(rev=1);supreme_court(rev=1);wetax(rev=1)
```

`captcha_data` 는 저장소에 없으므로, 학습 데이터가 없는 환경에서는 모델이 하나도 복사되지 않습니다
(`config.json` 에 적은 유형이 있으면 configure 가 실패합니다). 다른 위치를 쓰려면 `-DCAPTCHA_DATA_DIR=<경로>`,
모델 복사 자체를 끄려면 `-DCAPTCHA_COPY_MODELS=OFF` 입니다.

---

## 구조

| 파일 | 내용 | 대응 원본 |
|------|------|-----------|
| `src/main.cpp` | 인자 파싱, meta.json 로드, ONNX 세션, 출력 | `apps/cli/src/main.rs` |
| `src/preprocess.cpp` | 그레이스케일·임계값·테두리 제거·리사이즈 | `apps/cli/src/preprocess.rs` |
| `src/decode.cpp` | CTC prefix beam search + log-softmax | `apps/cli/src/decode.rs` |

동일 파이프라인을 완전히 구현한 **Rust CLI(`apps/cli`)를 기준으로 포팅**했습니다.
파이썬 원본과의 동등성 근거는 `apps/cli/README.md` 참고.

### 플랫폼 의존 코드

`preprocess.cpp` / `decode.cpp` / 테스트는 전부 이식 가능(stb + 표준 라이브러리)하고,
`#ifdef _WIN32` 은 **`main.cpp` 안 세 곳뿐**입니다.

| 항목 | Windows | Linux |
|------|---------|-------|
| 진입점 | `wmain(wchar_t**)` | `main(char**)` |
| 실행 파일 경로 | `GetModuleFileNameW` | `readlink("/proc/self/exe")` |
| 오류 메시지 경로 변환 | `WideCharToMultiByte`(콘솔 코드페이지) | 그대로 통과(UTF-8) |

나머지 경로 처리는 `std::filesystem::path` 하나로 끝납니다. `path::value_type` 이 Windows에서
`wchar_t`, POSIX에서 `char` 라 ONNX Runtime의 `ORTCHAR_T` 와 정확히 일치해서 `Ort::Session` 에
`path.c_str()` 를 변환 없이 넘길 수 있습니다.

### 파이썬/Rust와의 미세한 차이

리사이즈 필터가 stb(`STBIR_FILTER_CATMULLROM`)와 Rust `image`(`FilterType::CatmullRom`) 구현 차이로
픽셀당 소수점 아래 수준의 오차를 만듭니다. PIL BICUBIC(a=-0.5)과 동일한 커널이라 예측 문자열은 같고,
신뢰도만 소수점 셋째 자리에서 갈릴 수 있습니다.
