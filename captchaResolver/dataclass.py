import os
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
import captchaResolver.consts as consts 

@dataclass
class TrainInfo:
    """Training data information and configuration.
    
    Attributes:
        id: Unique identifier for the captcha type
        rev: Revision number for the dataset
        desc: Description of the training data
        base_dir: Base directory for captcha data
        train_image_path: Path to training images
        pred_image_path: Path to prediction images
        model_path: Path to model files
        image_width: Width of captcha images
        image_height: Height of captcha images
        label_length: Length of captcha labels
        characters: List of valid characters in captchas
        init: Whether to auto-initialize from data
        threshold: Pixel threshold for preprocessing (0 = disabled)
    """
    id: str
    rev: int = 0
    desc: str = "기본 학습 데이터"
    base_dir: str = "./captcha_data"
    train_image_path: str = "train"
    pred_image_path: str = "pred"
    model_path: str = "model"
    image_width: int = 200
    image_height: int = 50
    label_length: int = 5
    characters: list[str] = field(default_factory=lambda: list(consts.ALPHA_NUMERIC))
    init: bool = True
    threshold: int = 0

    def __post_init__(self) -> None:
        """Initialize training info from data if init=True."""
        if self.init:
            (
                self.train_image_path,
                self.pred_image_path,
                self.model_path,
                self.image_width,
                self.image_height,
                self.label_length,
                self.characters,
                self.threshold
            ) = self.get_train_info()

    def get_train_info(self) -> tuple[str, str, str, int, int, int, list[str], int]:
        """Extract training configuration from existing data."""
        train_image_path = self.get_image_dir(train=True)
        pred_image_path = self.get_image_dir(train=False)
        model_path = self.get_model_path(keras_native=False)
        
        train_data_list = self.get_data_files(train=True)
        
        with Image.open(train_data_list[-1]) as image:
            image_width, image_height = image.size

        labels = [
            os.path.basename(data_path).split(".")[0] for data_path in train_data_list
        ]
        label_length = max([len(label) for label in labels])
        characters = sorted(set(char for label in labels for char in label))
        threshold = 0
        
        return (
            train_image_path,
            pred_image_path,
            model_path,
            image_width,
            image_height,
            label_length,
            characters,
            threshold,
        )

    def get_image_dir(self, train: bool = True) -> str:
        """Get absolute path to image directory.
        
        Args:
            train: If True, return training dir; otherwise prediction dir
            
        Returns:
            Absolute path to image directory
        """
        image_dir = os.path.join(
            self.base_dir, 
            self.id, 
            str(self.rev), 
            'images', 
            'train' if train else 'pred'
        )
        return os.path.abspath(image_dir)

    def get_data_files(self, train: bool = True) -> list[str]:
        """Get list of data file paths.
        
        Args:
            train: If True, return training files; otherwise prediction files
            
        Returns:
            List of absolute paths to PNG image files
        """
        image_dir = self.get_image_dir(train)
        return glob.glob(os.path.join(image_dir, '*.png'))

    def get_labels(self, train: bool = True) -> list[str]:
        """Extract labels from filenames.
        
        Args:
            train: If True, return training labels; otherwise prediction labels
            
        Returns:
            List of label strings extracted from filenames
        """
        return [
            os.path.basename(data_path).split(".")[0]
            for data_path in self.get_data_files(train)
        ]

    def get_model_path(self, keras_native: bool = True) -> str:
        """Get absolute path to model file.
        
        Args:
            weights_only: If True, return path to weights file; 
                         otherwise full model file
            
        Returns:
            Absolute path to model file
        """
        model_path = os.path.join(self.base_dir, self.id, str(self.rev), 'model')
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)

        if keras_native:
            model_path = os.path.join(model_path, "weights.keras")

        return model_path

@dataclass
class CaptchaType:
    """Captcha type definition with associated training data.
    
    Attributes:
        id: Unique identifier for the captcha type
        name: Human-readable name
        desc: Description of the captcha type
        base_dir: Base directory for captcha data
        train_data: Associated training data info (auto-initialized)
    """
    id: str = 'default'
    name: str = '기본캡챠'
    desc: str = '기본 캡챠'
    base_dir: str = './captcha_data'
    train_data: Optional[TrainInfo] = None

    def __post_init__(self) -> None:
        """Initialize training data for this captcha type."""
        self.train_data = TrainInfo(
            id=self.id, 
            desc=self.desc + ' 학습 데이타', 
            base_dir=self.base_dir
        )
