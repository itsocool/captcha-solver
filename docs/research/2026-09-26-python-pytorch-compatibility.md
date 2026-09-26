# Python·PyTorch 안정성 및 호환성 점검

확인일: 2026-09-26. 대상은 당시 main의 웹 앱과 공용 Python 라이브러리다. 이 조사에서는 버전·환경 설정을 변경하지 않았다. 요구사항 정의서에 버전 선택에 대한 별도 ID는 없으며, 실행 검증의 관련 범위는 [EXP-003~005 및 공용 학습 회귀](../00.requirements.md)다. 아래 환경·검증 상태는 조사 시점의 스냅샷이며 후속 환경 구성 및 테스트 결과는 [요구사항 정의서](../00.requirements.md)에 기록했다.

## 판단

**Python 3.13 + PyTorch 2.14.0 + torchvision 0.29.0 + CUDA 12.6은 공식 지원 조합이며 유지할 수 있다.** 다만 PyTorch 2.14.0은 2026-09-02에 출시된 비교적 최근 정식 버전이다. 정식 배포 여부와 이 프로젝트의 학습·추론 안정성 검증은 구분해야 한다. 현재 로컬에는 새 웹 가상환경이 없고 이전 환경이 남아 있어, 실행 환경 정합성부터 맞추는 것을 권고한다.

## 확인한 사실

| 대상 | 저장소 설정 | 근거 |
| --- | --- | --- |
| 웹 Python | 선택 버전 3.13, 허용 범위 `>=3.13` | `apps/web/.python-version:1`, `apps/web/pyproject.toml:9` |
| 웹 PyTorch·torchvision | `2.14.0` / `0.29.0` 고정 | `apps/web/pyproject.toml:18` |
| Linux·Windows 빌드 | 공식 `cu126` 인덱스 | `apps/web/pyproject.toml:40`, `apps/web/uv.lock:1609`, `apps/web/uv.lock:1675` |
| Docker | Python 3.13 기반 이미지 | `Dockerfile:1`, `Dockerfile.cpu:5` |
| 공용 라이브러리 | Python `>=3.13`, torch `>=2.12.0`, torchvision `>=0.27.0` | `packages/python_3.13/pyproject.toml:9` |
| 루트 Python 선택 | 조사 시점에는 3.12였으며 이후 3.13으로 정렬 | `.python-version:1` |

- Python 3.13은 버그 수정 지원 중이며 지원 종료는 2029년 10월이다. 3.12는 보안 수정 단계다. 따라서 호환성만을 이유로 3.12로 낮출 근거는 발견하지 못했다. [Python 공식 지원 현황](https://devguide.python.org/versions/)
- PyTorch 2.14의 공식 지원 표에 Python 3.13과 stable CUDA 12.6이 포함된다. Linux x86·Windows용 CUDA 12.6 빌드는 Pascal을 포함하며 CUDA 13.0/13.2 빌드는 포함하지 않는다. GTX 10xx 지원 목적의 현재 인덱스 선택은 타당하다. [PyTorch 공식 지원 표](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)
- PyTorch 2.14.0은 2026-09-02 정식 릴리스다. torchvision 0.29의 공식 릴리스 설명도 torch 2.14와의 ABI 호환성을 명시한다. [PyTorch 릴리스](https://github.com/pytorch/pytorch/releases/tag/v2.14.0), [torchvision 릴리스](https://github.com/pytorch/vision/releases/tag/v0.29.0)
- 공식 CUDA 12.6 인덱스에 PyTorch 2.14.0의 CPython 3.13용 Linux x86_64 및 Windows wheel이 존재한다. [배포 인덱스](https://download.pytorch.org/whl/cu126/torch/)
- `uv lock --project apps/web --check --offline --no-python-downloads`는 종료 코드 0으로 통과했다. 출력은 `Using CPython 3.13.15`, `Resolved 100 packages in 1ms`다. 이는 잠금 파일 정합성 검사이며 실제 패키지 설치·실행 성공을 뜻하지 않는다.

## 로컬 환경의 불일치

설치된 버전 파일과 `pyvenv.cfg`를 읽어 확인했으며, 프로젝트 core나 torch를 import하지 않았다.

| 위치 | 발견 내용 |
| --- | --- |
| `apps/web/.venv` | 없음. 현재 웹 프로젝트의 실행 환경이 아직 설치되지 않음 |
| `.venv` | Python 3.12 환경, torch `2.12.0+cu130` (`.venv/pyvenv.cfg`, `.venv/lib/python3.12/site-packages/torch/version.py:4`) |
| `packages/python_3.13/aso-ai/.venv` | 이전 미추적 폴더에 남은 환경, torch `2.14.0+cu130` (`lib/python3.13/site-packages/torch/version.py:4`) |

두 잔존 환경은 웹 lock의 `cu126`과 다르다. 이 파일들의 존재만으로 현재 서버가 해당 환경을 사용한다고 단정할 수는 없다. `apps/web/server.sh:34`는 웹 가상환경이 없으면 PATH의 uvicorn으로 대체하므로 잘못된 환경을 선택할 가능성이 있다.

## 권고 사항

1. 버전 조합은 유지하고 Python 3.13의 최신 패치 버전을 사용한다. 조사 후 루트 `.python-version`을 웹과 같은 3.13으로 정리했다.
2. 공식 실행 환경은 `uv sync --project apps/web --locked`로 구성하고 `uv run --project apps/web ...`로 사용한다. 이번 점검에서는 설치를 실행하지 않았다.
3. 운영 재현성을 높이려면 웹 앱의 Python 허용 범위를 `>=3.13,<3.14`로 제한할지 검토한다. 현재 `>=3.13`은 향후 minor 버전까지 허용한다. 공용 라이브러리 범위는 지원 정책과 별도로 판단한다.
4. 공용 라이브러리 단독 설치는 웹 프로젝트의 정확한 버전 고정과 CUDA 인덱스를 적용받지 않는다. 배포용 wheel을 설치할 때도 목적에 맞는 torch·torchvision 버전과 인덱스를 별도로 지정한다. 하한만 있는 라이브러리 의존성이 곧 현재 충돌을 뜻하는 것은 아니다.
5. 최종 운영 안정성 판단에는 새 환경의 pytest와 실제 CPU/GPU 추론, CTC/AMP 학습 1배치, ONNX/ORT export 동등성 확인이 필요하다. 기존 [요구사항 문서의 검증 기록](../00.requirements.md)은 참고 근거이며 이번 실행 결과로 간주하지 않는다.

## 조사 당시 검증 한계

웹 가상환경 부재 및 `nvidia-smi` 명령 부재로 현재 호스트의 GPU·드라이버·CUDA 커널 실행을 검증하지 못했다. 패키지 설치, Docker 빌드, pytest, 학습·추론·export는 실행하지 않았다. 이번 결과는 공식 지원·배포 파일·매니페스트·잠금 파일 및 잔존 환경에 대한 점검이다.
