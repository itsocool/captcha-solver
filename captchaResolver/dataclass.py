import os, glob, random, struct
from dataclasses import dataclass, field
from typing import Optional, Final

DIGITS: Final = "0123456789"
LOWER_CASE: Final = "abcdefghijklmnopqrstuvwxyz"
UPPER_CASE: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALPHABET: Final = LOWER_CASE + UPPER_CASE
ALPHA_NUMERIC: Final = DIGITS + ALPHABET

@dataclass
class TrainInfo:
    captcha_id: str = 'default'
    backend: str = 'pytorch'
    rev: int = 0
    desc: str = "기본 학습 데이터"
    captcha_data_base_dir: str = "./captcha_data"
    image_width: int = 200
    image_height: int = 50
    label_length: int = 6
    characters: list[str] = field(default_factory=lambda: list(ALPHA_NUMERIC))
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

    def get_png_size(self,filepath) -> tuple[int, int]:
        with open(filepath, 'rb') as f:
            f.read(16)
            data = f.read(8)
            if len(data) < 8:
                raise ValueError("파일이 너무 짧거나 PNG 형식이 아닙니다.")
            width, height = struct.unpack('!II', data)
            return width, height

    def get_train_info(self) -> tuple[str, str, str, int, int, int, list[str], int]:
        train_image_path = self.get_image_dir(train=True)
        pred_image_path = self.get_image_dir(train=False)
        model_path = self.get_model_path()
        train_data_list = self.get_data_files(train=True)
        image_width, image_height = self.get_png_size(train_data_list[-1])

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
            self.captcha_data_base_dir, 
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

    def get_model_path(self) -> str:
        model_path = os.path.join(self.captcha_data_base_dir, self.captcha_id, str(self.rev), 'model')
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)

        # 모델 파일명 설정, 기본(pytorch): model_full.pth, Keras: weights.keras
        model_file_name = "model_full.pth" 
        
        # self.backend에 따라 모델 파일명 결정
        if self.backend == 'keras':
            model_file_name = "weights.keras"
    
        model_path = os.path.join(model_path, model_file_name)
        return model_path

    def get_model_base_dir(self) -> str:
        model_base_dir = os.path.join(self.captcha_data_base_dir, self.captcha_id, str(self.rev), 'model')
        model_base_dir = os.path.abspath(model_base_dir)

        if not os.path.exists(model_base_dir):
            os.makedirs(model_base_dir, exist_ok=True)

        return model_base_dir

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
            captcha_data_base_dir="./captcha_data"
        )
