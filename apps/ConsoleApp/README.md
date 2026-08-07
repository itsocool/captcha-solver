# ConsoleApp — captcha (C++ / CMake, Windows + Linux)

`apps/ConsoleApp.v4`(C# / .NET Framework 4.8)를 **네이티브 C++**로 옮긴 콘솔 실행 파일.
CLI 문법과 출력 형식이 같아 기존 `gov24.cmd` / `supreme_court.cmd` 를 그대로 쓸 수 있습니다.

.NET 런타임 없이 **실행 파일 + ONNX Runtime 공유 라이브러리 + `<id>.model` + `<id>.meta.json`** 만으로 동작합니다.

**Windows x64 / Linux x64·aarch64 모두 빌드**됩니다
(경로를 `std::filesystem::path`로만 다뤄서, `path::value_type`이 ONNX Runtime의 `ORTCHAR_T`와 그대로 맞습니다).
Linux만 필요하다면 `apps/cli`(Rust)가 이미 같은 파이프라인을 제공합니다 — 이쪽은 C++ 툴체인만 있는 환경용입니다.

> `apps/ConsoleApp.v4` 는 포팅 **원본**(C# / .NET Framework 4.8)이고, 이 디렉터리가 C++ 포팅본입니다.

---

## 사용법

```cmd
captcha.exe -c="gov24" -i="gov24.JPG"
:: 004657   (개행 없이 예측 문자열만)

captcha.exe -c="gov24" -i="a.png" -m="D:\models\gov24.model" --meta="D:\models\gov24.meta.json"
```

```bash
./captcha -c=gov24 -i=captcha.png     # Linux, 동일한 문법
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-c`, `--captcha-id` | (필수) | 캡차 ID |
| `-i`, `--image-path` | (필수) | 이미지 경로 |
| `-m`, `--model-path` | `<실행 파일 폴더>/<id>.model` | ONNX 모델 |
| `--meta` | `<모델>.meta.json`, 없으면 `<실행 파일 폴더>/<id>.meta.json` | 메타데이터 |

`-c=값` / `-c 값` 두 형식 모두 받습니다(C# 원본과 동일). 지원 이미지: PNG, JPEG, BMP, GIF, TGA, PSD, HDR, PNM.

종료 코드: `0` 성공 / `2` 모델·메타 파일 없음 / `1` 그 외 오류.

> **원본과 다른 점 하나**: 인자 누락 시 C# 원본은 stdout에 안내를 찍고 `0`을 반환하지만
> 여기서는 stderr + `1`입니다. `.cmd` 스크립트는 종료 코드를 보지 않으므로 영향 없습니다.

---

## 빌드

CMake 3.24+ 와 C++17 컴파일러(**MSVC v143** 또는 **GCC 9+ / Clang 9+** — `<filesystem>` 때문)가 필요합니다.
**최초 configure 때 네트워크**를 씁니다 (ONNX Runtime 프리빌트, stb 헤더를 FetchContent로 받음).

### Windows — PowerShell 스크립트 (권장)

```powershell
cd apps\ConsoleApp
.\ctest.ps1          # configure -> build -> test 전부
```

| 스크립트 | 하는 일 |
|----------|---------|
| `config.ps1` | cmake를 찾아 PATH에 얹고 `build/` 생성 |
| `build.ps1` | `config.ps1` 호출 후 빌드 |
| `ctest.ps1` | `build.ps1` 호출 후 테스트 |
| `pred.ps1` | 빌드된 exe로 이미지를 대량 인식해 정답(파일명)과 대조 |

CMake가 PATH에 없어도 됩니다 — `config.ps1`이 `vswhere`로 Visual Studio를 찾아 번들 CMake를
**이 프로세스의 PATH에만** 추가합니다(전역 PATH는 건드리지 않음).

```powershell
.\build.ps1 -Config Debug     # Debug 빌드
.\ctest.ps1 -Filter decode    # 특정 테스트만
.\config.ps1 -Clean           # build/ 지우고 처음부터 (제너레이터 변경 시 필요)
```

제너레이터는 지정하지 않아 CMake가 **설치된 최신 Visual Studio**를 고릅니다. VS를 업그레이드해도
스크립트를 고칠 필요가 없습니다.

### Windows — 직접 실행

```cmd
cd apps\ConsoleApp
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
:: -> build\Release\captcha.exe + onnxruntime.dll + <id>.model + <id>.meta.json
```

VS에 딸린 CMake는 `%ProgramFiles(x86)%\Microsoft Visual Studio\<버전>\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin` 에 있습니다.

### Linux — 셸 스크립트 (권장)

```bash
cd apps/ConsoleApp
./ctest.sh           # configure -> build -> test 전부
```

| 스크립트 | 하는 일 | Windows 대응 |
|----------|---------|--------------|
| `config.sh` | cmake를 찾아 PATH에 얹고 `build/` 생성 | `config.ps1` |
| `build.sh` | `config.sh` 호출 후 빌드 | `build.ps1` |
| `ctest.sh` | `build.sh` 호출 후 테스트 | `ctest.ps1` |
| `pred.sh` | 빌드된 실행 파일로 이미지를 대량 인식해 정답(파일명)과 대조 | `pred.ps1` |

```bash
./build.sh --config Debug     # Debug 빌드
./build.sh --jobs 4           # 동시 컴파일 잡 수 (기본: nproc)
./ctest.sh --filter decode    # 특정 테스트만
./config.sh --clean           # build/ 지우고 처음부터
```

PowerShell 판과 다른 점 두 가지:

- 리눅스 기본 제너레이터(Unix Makefiles / Ninja)는 **단일 구성**이라 빌드 타입을 configure
  시점에 정해야 합니다. 그래서 `config.ps1`에 없는 `--config`를 `config.sh`가 받습니다.
- `build.sh`/`ctest.sh`는 하위 스크립트를 `source`로 부릅니다. 그래야 `config.sh`가
  PATH에 얹은 cmake가 뒤따르는 `cmake --build`/`ctest`에도 유효합니다
  (PowerShell의 `& script`가 같은 프로세스에서 도는 것과 같은 효과).

`-Config Debug` 처럼 PowerShell식 인자도 그대로 받습니다.

### Linux — 직접 실행

```bash
cd apps/ConsoleApp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# -> build/captcha + libonnxruntime.so* + <id>.model + <id>.meta.json
```

실행 파일에 `RUNPATH=$ORIGIN` 이 박히고 `libonnxruntime.so*` 가 옆에 복사되므로, **디렉터리째 옮겨도 그대로 동작**합니다.
(`captcha` + `libonnxruntime.so.1` + `<id>.model` + `<id>.meta.json` 만 있으면 됩니다.)

### 공통 옵션

ONNX Runtime 버전은 `-DORT_VERSION=1.20.1` 로 바꿀 수 있습니다(모델은 opset 17 / IR 8이라 1.13 이상이면 됩니다).
`-DCAPTCHA_COPY_MODELS=OFF` 를 주면 모델 복사를 생략합니다.

### 검증

```cmd
ctest --test-dir build -C Release --output-on-failure    :: Windows
```
```bash
ctest --test-dir build --output-on-failure               # Linux (./ctest.sh 가 빌드까지 해준다)
```

- `decode` — CTC prefix beam search 단위 테스트 7개
- `samples` — `apps/cli/samples/<id>/` 50장을 인식해 파일명(정답)과 대조.
  대상은 `gov24`, `supreme_court`, `kshop`, `wetax`, `default` (`tests/samples.cmake` 의 `IDS`)

실측: Windows(MSVC 17.14 / x64), Linux(GCC 13.3 / x64) 양쪽 **전부 일치**, 예측 바이트 동일.

### pred — 대량 정확도 확인

`captcha_data/<id>/<rev>/images/pred/` 에서 **100장을 무작위로 뽑아** 인식하고 파일명(정답)과 대조합니다.
`rev` 는 `apps/cli/models/<id>.meta.json` 에 적힌 것을 쓰고, 없으면 가장 큰 번호를 고릅니다.
`captcha_data` 가 없으면 `apps/cli/samples/<id>` 10장으로 폴백합니다.

```powershell
.\pred.ps1                        # supreme_court, 100장
.\pred.ps1 gov24                  # 다른 캡차
.\pred.ps1 kshop -Count 50        # 장수 지정
.\pred.ps1 -Seed 42               # 같은 표본 재현
```

```bash
./pred.sh                         # supreme_court, 100장
./pred.sh gov24                   # 다른 캡차
./pred.sh kshop --count 50        # 장수 지정
./pred.sh --seed 42               # 같은 표본 재현
```

```
출처 : ...\captcha_data\supreme_court\0\images\pred  (1061장 중 100장 무작위)

이미지           정답       예측       시간     결과
------------------------------------------------------
073575.png       073575     073575     126 ms   ✅
825890.png       825890     825890     103 ms   ✅
...
------------------------------------------------------
캡차        supreme_court
일치        100/100
일치율      100.0%
총 소요     12.4초  (평균 124 ms)
```

전부 일치하면 종료 코드 0, 하나라도 틀리면 1입니다. 시간은 **프로세스 전체 실행 시간**이라
ONNX 세션 초기화가 포함됩니다(Rust CLI의 내부 측정값보다 큽니다).

추론 자체가 실패한 행은 예측이 `-` 로 나오고 오류 메시지가 뒤에 붙습니다.

```
WpcDdu.png       WpcDdu     -          129 ms   ❌  Error: model input 200x50 != metadata 250x50
```

실측 (`-Seed 42`, 각 100장):

| captcha | 일치 |
|---------|------|
| supreme_court | 100/100 |
| gov24 | 100/100 |
| wetax | 100/100 |
| default | 100/100 |
| kshop | 86/100 |

kshop 86%는 `apps/cli/README.md` 에 기록된 ONNX 기준 정확도와 같은 수치입니다(모델 문제이며 포팅 문제가 아닙니다).
`dev` 는 입력 크기 불일치로 전부 추론 실패합니다.

---

## 모델과 메타데이터

`.model` 은 확장자만 다른 **ONNX 파일**입니다(`gov24.model` 은 `apps/cli/models/gov24.onnx` 와 동일).

ONNX만으로는 **문자셋·레이블 길이·전처리 종류**를 알 수 없어 사이드카 JSON이 함께 있어야 합니다.

```json
{ "captcha_id": "gov24", "image_width": 200, "image_height": 50,
  "label_length": 6, "characters": "0123456789", "threshold": 60, "preprocess": "default" }
```

`apps/cli/tools/sync_models.py` 가 생성하며, `apps/cli/models/` 에 캡차 6종이 준비돼 있습니다.

빌드 시 **`apps/cli/models/*.meta.json` 에 있는 모든 캡차**를 산출물 폴더로 복사합니다
(`default`, `dev`, `gov24`, `kshop`, `supreme_court`, `wetax`). 모델 본체는
`ConsoleApp.v4/<id>.model` 이 있으면 그걸 쓰고, 없으면 `apps/cli/models/<id>.onnx` 를
`<id>.model` 이라는 이름으로 복사합니다. `sync_models.py` 로 모델을 추가하면 다음 빌드에
자동으로 딸려옵니다(`file(GLOB CONFIGURE_DEPENDS)`).

> `dev` 는 ONNX가 200×50으로 export됐는데 메타는 250×50이라 **추론이 안 됩니다**(원인은
> `apps/cli/README.md` 참고). 복사는 되지만 실행하면 크기 불일치 메시지로 끊깁니다.

> `ConsoleApp.v4/supreme_court.model` 은 `apps/cli/models/supreme_court.onnx` 와 **가중치가 다른 리비전**입니다
> (입력·출력 shape는 같아 메타는 그대로 맞고, 샘플 10장도 10/10 통과합니다).
> `gov24.model` 은 `gov24.onnx` 와 바이트 단위로 동일합니다.

---

## 구조

| 파일 | 내용 | 대응 원본 |
|------|------|-----------|
| `src/main.cpp` | 인자 파싱, meta.json 로드, ONNX 세션, 출력 | `ConsoleApp.v4/Program.cs` |
| `src/preprocess.cpp` | 그레이스케일·임계값·테두리 제거·리사이즈 | `apps/cli/src/preprocess.rs` |
| `src/decode.cpp` | CTC prefix beam search + log-softmax | `apps/cli/src/decode.rs` |

`HiWorks.Lib.v4`(C# 엔진 본체)는 이 저장소에 없어, 동일 파이프라인을 완전히 구현한
**Rust CLI(`apps/cli`)를 기준으로 포팅**했습니다. 파이썬 원본과의 동등성 근거는 `apps/cli/README.md` 참고.

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
신뢰도만 소수점 셋째 자리에서 갈릴 수 있습니다. `samples` 테스트가 이를 확인합니다.
