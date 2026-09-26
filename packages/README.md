# 패키지 경계 규칙

`packages/`는 공용 Python 라이브러리와 앱 배포 산출물을 보관한다.

| 경로 | 역할·참조 |
|---|---|
| `python_3.13/` | 현행 `aso-ai` 프로젝트. `apps/web/pyproject.toml`·`uv.lock`과 두 Dockerfile이 이 경로를 사용한다. |
| `cli/` | Rust CLI 설치 파일과 설정. `apps/cli/pack.ps1`의 배포 출력 경로다. |
| `WinConsoleApp/` | Windows Console App 설치 파일과 설정. `apps/WinConsoleApp/pack.ps1`의 배포 출력 경로다. |
| `iptime-captcha-*.zip` | 루트 README에 안내된 Chrome 확장 프로그램 배포본이다. |

- 모델·학습 알고리즘은 `python_3.13/src/aso_ai/core.py`, 실험 보조 로직은 `experiments.py`가 소유한다. 캡차 설정·전처리·실행 엔진은 `apps/web/core/`가 소유한다.
- 모델 코드는 웹 패키지를 런타임에 import하지 않는다. 단건 CLI는 웹 엔진을 사용하므로 웹 패키지와 함께 설치하며, 명령은 웹 프로젝트가 등록한다.
- 미참조 여부는 import뿐 아니라 진입점, 로컬 의존성, Docker COPY, 배포 스크립트·문서까지 확인한다. 배포 산출물은 코드 import가 없어도 유지한다.
- 가상환경·바이트코드·egg-info는 소스가 아닌 생성 파일이다. 이전 경로를 정리할 때 실행 환경의 editable 설치 경로도 확인하고, 미추적 사용자 소스는 삭제 동의 없이 제거하지 않는다.

정리 수용 기준과 검증 명령은 [PKG-001](../docs/00.requirements.md)을 따른다.
