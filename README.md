(# hyper-captcha-resolver)

이 저장소의 예제는 TensorFlow 2.20.0을 사용하도록 설정되어 있습니다. GPU(엔비디아 CUDA)를 사용해 TensorFlow를 가속하려면 시스템 드라이버와 CUDA/cuDNN이 적절히 설치되어야 합니다.

## 요구사항 (TensorFlow 2.20.0)

- Python: 프로젝트의 `pyproject.toml`은 `requires-python = ">=3.12"`로 설정되어 있습니다. 적절한 파이썬 버전을 사용하세요.
- NVIDIA 드라이버: 최신 드라이버 설치 권장. `nvidia-smi`로 확인하세요.
- CUDA와 cuDNN: TensorFlow 2.20.0과 호환되는 CUDA/cuDNN 버전을 사용해야 합니다. (공식 문서를 확인하세요.)

참고: TensorFlow pip 패키지는 특정 CUDA/CuDNN 버전에서 사전 빌드됩니다. 호환성 문제로 인해 시스템에 설치된 CUDA와 pip 패키지의 요구사항이 일치해야 GPU가 활성화됩니다.

## 빠른 시작 — 검사 및 설치 스크립트

간단한 자동화 스크립트 `scripts/setup_tensorflow_gpu.sh`가 포함되어 있습니다. 이 스크립트는 다음을 수행합니다:

- `nvidia-smi` 검사
- `tensorflow==2.20.0` pip 설치
- Python에서 TensorFlow가 CUDA로 빌드되었는지와 사용 가능한 GPU 장치가 있는지 확인

사용 방법:

```bash
# 저장소 루트에서
bash scripts/setup_tensorflow_gpu.sh
```

스크립트는 시스템 수준의 CUDA/cuDNN 드라이버 설치를 자동으로 수행하지 않습니다. 드라이버나 라이브러리를 설치하려면 운영체제와 GPU 아키텍처에 맞는 공식 NVIDIA 설치 가이드를 따르세요.

## 문제 해결 팁

- `Built with CUDA: False`가 출력될 경우:
	- pip로 설치한 TensorFlow가 CUDA 지원 빌드가 아닌 경우입니다. 일반적으로 `pip install tensorflow`는 GPU 지원을 포함하지만, 환경과 빌드에 따라 달라질 수 있습니다.
	- 시스템에 설치된 CUDA/cuDNN 버전이 TensorFlow가 기대하는 버전과 다를 수 있습니다. TensorFlow 릴리스 노트를 확인하여 호환성 매칭을 하세요.
- `nvidia-smi`가 없거나 GPU가 인식되지 않는 경우:
	- NVIDIA 드라이버가 설치되어 있는지 확인하세요.
	- WSL2 같은 환경에서는 추가 설정이 필요할 수 있습니다.

더 도움이 필요하면 시스템의 `nvidia-smi` 출력과 `python -c "import tensorflow as tf; print(tf.__version__, tf.test.is_built_with_cuda(), tf.config.list_physical_devices('GPU'))"` 출력을 공유해 주세요.

