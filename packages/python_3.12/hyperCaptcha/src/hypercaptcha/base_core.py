"""
Base Core Module for CAPTCHA Recognition Models

이 모듈은 모델의 공통 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
from .dataclass import CaptchaType, TrainData


class BaseModel(ABC):
    """
    CAPTCHA 인식 모델의 추상 기본 클래스

    구체 모델 구현이 상속받아 구현합니다.
    """
    
    def __init__(self, captcha_type: CaptchaType, verbose: int = 1):
        """
        Args:
            captcha_type: 캡차 타입 설정
            verbose: 출력 레벨 (0: 없음, 1: 기본)
        """
        self.captcha_type = captcha_type
        self.train_data: TrainData = captcha_type.train_data
        self.verbose = verbose
        self.model = None
    
    @abstractmethod
    def build_model(self) -> Any:
        """
        모델 아키텍처 생성
        
        Returns:
            생성된 모델 객체
        """
        pass
    
    @abstractmethod
    def train_model(self, *args, **kwargs) -> List[float]:
        """
        모델 학습
        
        Returns:
            학습 손실 히스토리
        """
        pass
    
    @abstractmethod
    def load_prediction_model(self, model_path: str = None) -> Any:
        """
        저장된 모델 로드
        
        Args:
            model_path: 모델 파일 경로 (None이면 기본 경로 사용)
            
        Returns:
            로드된 모델
        """
        pass
    
    @abstractmethod
    def predict(self, image_path: str, **kwargs) -> Tuple[str, float]:
        """
        단일 이미지 예측
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            (예측 텍스트, 신뢰도)
        """
        pass
    
    @abstractmethod
    def split_dataset(self, batch_size: int = 32, train_size: float = 0.9, 
                     shuffle: bool = True) -> Tuple[Any, Any]:
        """
        데이터셋을 학습/검증 세트로 분할
        
        Args:
            batch_size: 배치 크기
            train_size: 학습 데이터 비율
            shuffle: 셔플 여부
            
        Returns:
            (학습 데이터셋, 검증 데이터셋)
        """
        pass
    
    def get_model_path(self) -> str:
        """기본 모델 저장 경로 반환"""
        return self.train_data.get_model_path()
    
    def get_model_base_dir(self) -> str:
        """모델 저장 디렉토리 반환"""
        return self.train_data.get_model_base_dir()
    
    @property
    def characters(self) -> str:
        """사용 가능한 문자 집합 (감지된 값을 우선 사용)"""
        detected = self.train_data.detected_characters
        return "".join(detected) if isinstance(detected, list) else detected
    
    @property
    def label_length(self) -> int:
        """레이블 길이 (감지된 값을 우선 사용)"""
        return self.train_data.detected_label_length

    @property
    def image_width(self) -> int:
        """이미지 너비 (감지된 값을 우선 사용)"""
        return self.train_data.detected_image_width

    @property
    def image_height(self) -> int:
        """이미지 높이 (감지된 값을 우선 사용)"""
        return self.train_data.detected_image_height
