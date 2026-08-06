## hyper-captcha-resolver — AI 안내서

기본 언어: 한국어

패키지/프로젝트 매니저: uv (의존성은 `pyproject.toml` + `uv.lock`이 유일한 소스입니다. `uv sync`로 설치하고, 의존성 추가는 `uv add`를 사용하세요.)

이 문서는 코드를 빠르게 이해하고 생산적으로 작업하기 위한 핵심 정보만 간결하게 정리합니다. 변경 전후에는 `train.py` / `pred.py` 같은 스크립트를 참고하세요.

### 한눈에 보는 아키텍처
- 주요 모듈: `engine.py` (학습/추론 엔트리), `dataclass.py` (데이터/경로 규약), `base_core.py` (공통 인터페이스)
- 데이터 레이아웃: `captcha_data/<captcha_id>/<rev>/images/{train,pred}` — 이미지 파일명(확장자 제외)이 레이블입니다. 예: `captcha_data/kshop/0/images/train/abc12.png` → 레이블 `abc12`.
- 모델 파일: PyTorch 기반 모델은 `captcha_data/<id>/<rev>/model/model_full.pt` (전체 모델), 필요시 `model_full.pt.onnx` 또는 `model_jit.pt`를 사용합니다. 기존 Keras 포맷(`weights.keras`)은 더 이상 사용하지 않습니다.

### 핵심 디자인 결정(발견한 규칙)
- 레이블과 문자 집합은 학습 데이터에서 자동으로 추출됩니다 (`TrainData.get_train_info`) — 파일명으로부터 `label_length`와 `characters`를 계산합니다.
- 입력 이미지는 Pillow(`PIL`)로 읽고 그레이스케일 변환 및 리사이즈를 수행합니다(`TrainData.image_pre_process`). 따라서 전처리 파이프라인은 `image_width`, `image_height`, `threshold` 값을 기준으로 동작합니다.
- 레이블 처리와 모델 저장 규약은 PyTorch 중심입니다. 모델 저장은 `model_full.pt` 형식을 기본으로 하며, 호환성을 위해 ONNX/JIT 형식도 사용됩니다.

### 개발자 워크플로(빠른 예시)
- 환경 요구: `pyproject.toml`에 `requires-python = "==3.12.*"`가 설정되어 있습니다.
- 의존성 설치: `uv sync` (락파일 고정 설치는 `uv sync --frozen`).
- GPU 체크: `python main.py` 출력과 `nvidia-smi`를 확인하세요.
- 학습: `python train.py` (스크립트에서 `captcha_id`, 하이퍼파라미터를 수정하여 실행).
- 검증/추론: `python pred.py` (기본적으로 `captcha_id='kshop'`).

※ 패키지 매니저(권장)
- `uv` 필수: 설치는 `uv sync`, 실행은 `uv run <cmd>`, 의존성 추가는 `uv add <pkg>`. pip/requirements.txt는 사용하지 않습니다.

### 코드 패턴과 주의사항(에이전트가 알면 좋은 것)
- 파일명 기반 라벨: 모든 이미지 파일명(확장자 제외)이 레이블이므로, 파일명 규칙을 지키세요.
- 전처리 위치: 임계값(`threshold`)과 리사이즈는 `TrainData.image_pre_process`에 있습니다. 변경 시 추론 파이프라인(예: `predict`)과 일관되게 유지해야 합니다.
- 모델 저장 규약: PyTorch 모델은 `model_full.pt`를 기본으로 사용합니다.

### 예시 참조(파일/함수)
- 데이터 클래스: `dataclass.py::TrainData` — `get_image_dir`, `get_data_files`, `get_model_path`
- 모델 루틴: `core.py` 및 `engine.py` — 학습/추론 진입점
- 상수: `dataclass.py::ALPHA_NUMERIC`
- 실행 스크립트: `train.py`, `pred.py`, `main.py`

### 빠른 체크리스트(문제 상황에서)
- GPU 미탐지: `python main.py` 출력과 `nvidia-smi` 확인
- 라벨 불일치/길이 문제: `TrainData`가 `label_length`와 `characters`를 자동 계산하므로, 데이터셋에 레이블이 정상적으로 파일명으로 존재하는지 확인

### 작업 제안(작업을 시작할 때 우선순위)
1. 작은 변경(예: 전처리/threshold 실험) 시 `pred.py`로 빠르게 검증
2. 모델 구조 변경 시 `core.py`와 `engine.py` 연결을 확인
3. 모델 배포는 `model_full.pt` 또는 ONNX/JIT 형식을 사용

더 필요한 섹션이나 예제가 있으면 알려주세요. 불명확한 부분(환경, 데이터 형식 등)이 있으면 구체적으로 질문하겠습니다.
