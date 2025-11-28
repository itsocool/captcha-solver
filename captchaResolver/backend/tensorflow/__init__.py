"""
TensorFlow/Keras Backend for CAPTCHA Recognition
"""

try:
    from captchaResolver.backend.tensorflow.core import (
        KerasModel,
        CTCLayer,
    )
    
    __all__ = [
        'KerasModel',
        'CTCLayer',
    ]
except ImportError:
    # TensorFlow가 설치되지 않은 경우
    __all__ = []
