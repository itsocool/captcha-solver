## hyper-captcha-resolver — AI 안내서

기본 언어: 한국어

패키지/프로젝트 매니저: uv (프로젝트는 `pyproject.toml`을 사용합니다; `uv`로 의존성/lock 관리가 가능하면 우선 사용하고, 없을 경우 `requirements.txt`/pip를 대체 수단으로 사용하세요.)

이 문서는 코드를 빠르게 이해하고 생산적으로 작업하기 위한 핵심 정보만 간결하게 정리합니다. 변경 전후에는 `train.py` / `pred.py` 같은 스크립트를 참고하세요.

### 한눈에 보는 아키텍처
- 주요 모듈: `captchaResolver/core.py` (모델 정의, 학습/추론 로직), `captchaResolver/dataclass.py` (데이터/경로 규약), `captchaResolver/consts.py` (문자 집합 상수)
- 데이터 레이아웃: `captcha_data/<captcha_id>/<rev>/images/{train,pred}` — 이미지 파일명(확장자 제외)이 레이블입니다. 예: `captcha_data/kshop/0/images/train/abc12.png` → 레이블 `abc12`.
- 모델 파일: `captcha_data/<id>/<rev>/model/weights.keras` (전체 모델), `.../.weights.h5` (weights_only). `TrainInfo.get_model_path(weights_only=True)`가 이 규약을 사용합니다.

### 핵심 디자인 결정(발견한 규칙)
- 레이블과 문자 집합은 학습 데이터에서 자동으로 추출됩니다 (`TrainInfo.get_train_info`) — 파일명으로부터 `label_length`와 `characters`를 계산합니다.
- 입력 이미지는 그레이스케일(채널=1)로 읽고 크기(`image_width`, `image_height`)로 리사이즈한 뒤 전치(`tf.transpose(..., perm=[1,0,2])`)합니다. 따라서 모델은 (width, height, 1) 형태를 기대합니다.
- 레이블 처리: Keras `StringLookup`를 통해 문자→정수, CTC 손실(`CTCLayer`)을 사용합니다. 전체 모델 로딩 시 `custom_objects={'CTCLayer': CTCLayer}`가 필요합니다.
- `TrainInfo.threshold` 값이 존재하면 `encode_single_sample`에서 픽셀 임계값을 적용합니다(이 프로젝트에서는 `train.py`에서 `threshold=60` 등으로 설정).

### 개발자 워크플로(빠른 예시)
- 환경 요구: `pyproject.toml`에 `requires-python = ">=3.12"`. README는 TensorFlow 2.20.0 사용을 권장합니다.
- 의존성 설치: 프로젝트 루트의 `requirements.txt`를 사용하세요.
- GPU/TF 체크: `python main.py` — `main.py`는 `nvidia-smi`와 TF GPU 감지를 도와줍니다.
- 학습: `python train.py` (스크립트에서 `captcha_id`, 하이퍼파라미터를 수정하여 실행). 프로젝트는 모델과 weights 저장 동작을 `Model(..., save_model=..., save_weights=...)`로 제어합니다.
- 검증/추론: `python pred.py` (기본적으로 `captcha_id='kshop'`). `Model(..., weights_only=False)`로 전체 모델을 로드하거나 `weights_only=True`로 가중치만 불러올 수 있습니다.

※ 패키지 매니저(권장)
- uv 사용 권장: `pyproject.toml` 기반 의존성 관리를 위해 `uv`가 지원되면 `uv install`(또는 uv의 표준 설치 커맨드)을 사용하세요. 환경에 따라 `python -m pip install -r requirements.txt`로 대체할 수 있습니다.

### Context7 MCP 서버 사용(라이브러리 문서 조회)
- 외부 라이브러리의 최신 문서나 코드 예제가 필요할 때는 Context7 기반 MCP 서버를 사용하여 문서를 조회하세요.
- 권장 순서: 1) 라이브러리 이름을 resolve하여 Context7 호환 라이브러리 ID를 얻고, 2) 받은 ID로 라이브러리 문서를 요청합니다. 예: `tensorflow` → resolve → `"/org/project"` 형태의 ID → 문서 요청.
- 목적: 빠르게 API 변화/버전별 예제 확인, 호환성 문제 조사(예: TF/CUDA 호환성 표준 확인)에 유용합니다.

예시 사용 흐름(권장)

1. 라이브러리 이름으로 ID를 요청(예: `tensorflow`) — 응답으로 Context7 호환 ID(예: `"/org/project"`)를 받습니다.
2. 받은 ID를 사용해 문서 요청을 보냅니다. 파라미터로 `tokens`(최대 토큰 길이)와 `topic`(선택적, 예: `installation`, `compatibility`, `api`)를 지정할 수 있습니다.

간단한 예시(개념적 설명):

- 호출 A: resolve 라이브러리 이름
	- 입력: `tensorflow`
	- 출력(예시): `"/org/project"` (Context7 호환 라이브러리 ID)

- 호출 B: get-library-docs
	- 입력: `context7CompatibleLibraryID="/org/project"`, `tokens=3000`, `topic="compatibility"`
	- 출력: 해당 라이브러리의 관련 문서(호환성 정보, 설치 가이드 등)

팁:
- 라이브러리명이 모호하면 resolve 단계에서 후보 목록을 받고 가장 관련성 높은 ID를 선택하세요.
- `topic`은 문서 길이를 줄이고 특정 영역을 빠르게 찾는 데 유용합니다(예: `ctc`, `keras`, `cuda`).
- 항상 resolve를 먼저 호출하세요 — `get-library-docs`는 Context7 ID 형식을 요구합니다.

### 코드 패턴과 주의사항(에이전트가 알면 좋은 것)
- 파일명 기반 라벨: 모든 이미지 파일명(확장자 제외)이 레이블이므로, 파일명 규칙을 깨지 마세요. (예: 공백, 대소문자 일관성 주의)
- 이미지 포맷: `encode_single_sample`는 `tf.io.decode_png(..., channels=1)`를 사용합니다. PNG가 아닌 포맷은 추가 처리 필요.
- 전처리 위치: 임계값(threshold)과 리사이즈는 `encode_single_sample`에 있습니다. 변경 시 추론 파이프라인(예: `predict`)과 일관되게 유지해야 합니다.
- 모델 저장 규약: 전체 모델 저장(`weights.keras`)과 weights-only(`.weights.h5`) 파일명을 구분합니다. `load_prediction_model`은 `weights_only` 플래그에 따라 다르게 동작합니다.
- 커스텀 레이어: `CTCLayer`는 `@keras.saving.register_keras_serializable(package="Core")`로 등록되어 있지만, 안전하게 로드하려면 `custom_objects`를 제공하세요.

### 예시 참조(파일/함수)
- 데이터 클래스: `captchaResolver/dataclass.py::TrainInfo` — `get_image_dir`, `get_data_files`, `get_model_path`
- 모델 루틴: `captchaResolver/core.py::Model` — `encode_single_sample`, `build_model`, `train_model`, `load_prediction_model`, `predict`, `validate_model`
- 상수: `captchaResolver/consts.py::ALPHA_NUMERIC`
- 실행 스크립트: `train.py`, `pred.py`, `main.py`

### 빠른 체크리스트(문제 상황에서)
- GPU 미탐지: `python main.py` 출력과 `nvidia-smi` 확인 — README의 TF/CUDA 호환성 참고
- 모델 불러오기 에러: `CTCLayer`가 등록되어 있는지, `weights_only` 플래그 값이 맞는지 확인
- 라벨 불일치/길이 문제: `TrainInfo`가 `label_length`와 `characters`를 자동 계산하므로, 데이터셋에 레이블이 정상적으로 파일명으로 존재하는지 확인

### 작업 제안(작업을 시작할 때 우선순위)
1. 작은 변경(예: 전처리/threshold 실험) 시 `pred.py`로 빠르게 검증
2. 모델 구조 변경 시 `build_model`에서 `unit = len(characters)+1` 계산 유지
3. 전체 모델 파일을 배포하거나 재현하려면 `save_model=True`로 저장하고 `custom_objects`를 문서화

더 필요한 섹션이나 예제가 있으면 알려주세요. 불명확한 부분(환경, 데이터 형식 등)이 있으면 구체적으로 질문하겠습니다.
