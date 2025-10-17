import os, glob
from dataclasses import dataclass, field
import random
from sre_parse import DIGITS
from typing import Optional
from PIL import Image
from captchaResolver.consts import ALPHA_NUMERIC

@dataclass
class TrainInfo:
    captcha_id: str
    rev: int = 0
    desc: str = "기본 학습 데이터"
    base_dir: str = "./captcha_data"
    image_width: int = 200
    image_height: int = 50
    label_length: int = 6
    characters: list[str] = field(default_factory=lambda: list(DIGITS))
    init: bool = True
    threshold: int = 0

    def __post_init__(self) -> None:
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
        image_dir = os.path.join(
            self.base_dir, 
            self.captcha_id, 
            str(self.rev), 
            'images',
            'train' if train else 'pred'
        )
        return os.path.abspath(image_dir)

    def get_data_files(self, train: bool = True) -> list[str]:
        image_dir = self.get_image_dir(train)
        return glob.glob(os.path.join(image_dir, '*.png'))

    def get_labels(self, train: bool = True) -> list[str]:
        return [
            os.path.basename(data_path).split(".")[0] for data_path in self.get_data_files(train)
        ]

    def get_model_path(self, keras_native: bool = True) -> str:
        model_path = os.path.join(self.base_dir, self.captcha_id, str(self.rev), 'model')
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)

        if keras_native:
            model_path = os.path.join(model_path, "weights.keras")

        return model_path

    def get_pred_image_path(self) -> str:
        image_dir = self.get_image_dir(train=False)
        if isinstance(image_dir, (list, tuple)):
            candidates = [p for p in image_dir if os.path.isfile(p)]
        else:
            candidates = glob.glob(os.path.join(image_dir, '*'))
        if not candidates:
            raise RuntimeError(f"No images found in {image_dir}")
        image_path = random.choice(candidates)
        return image_path

@dataclass
class CaptchaType:
    id: str = 'default'
    name: str = '기본캡챠'
    desc: str = '기본 캡챠'
    train_data: Optional[TrainInfo] = None

    def __post_init__(self) -> None:
        """Initialize training data for this captcha type."""
        self.train_data = TrainInfo(
            captcha_id=self.id,
            desc=self.desc + ' 학습 데이타',
            base_dir="./captcha_data"
        )
