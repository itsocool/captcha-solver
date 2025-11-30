"""
호환성 레이어: 기존 저장된 모델 로딩을 위한 별칭

기존 모델이 captchaResolver.keras_core 경로로 저장되어 있기 때문에,
해당 모듈을 찾을 수 있도록 re-export 합니다.

새로 학습된 모델은 captchaResolver.backend.tensorflow.core 경로를 사용합니다.
"""

# try:
#     from captchaResolver.backend.tensorflow.core import *
#     from captchaResolver.backend.tensorflow.core import (
#         KerasModel,
#         CTCLayer,
#     )
# except ImportError:
#     # TensorFlow가 설치되지 않은 경우
#     pass
