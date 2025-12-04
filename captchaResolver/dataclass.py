import os
import glob
from pyexpat import model
import random
import struct
import shutil
import string
from PIL import Image
from typing import Final, List, Tuple
from dataclasses import dataclass, field

DIGITS: Final = string.digits
LOWER_CASE: Final = string.ascii_lowercase
UPPER_CASE: Final = string.ascii_uppercase
ALPHABET: Final = string.ascii_letters
ALPHA_NUMERIC: Final = string.digits + string.ascii_letters
CAPTCHA_CHAR_SETS: Final = "2345678bcdefgmnpwxy"
DEV_CHAR_SETS: Final = "2345678ABCDEFGHKLMNPRSTUVWYZabcdefhklmnoprstuvwyz"

@dataclass
class TrainData:
    captcha_id: str
    backend: str = "pytorch"
    rev: int = 0
    train_data_base_dir: str = "./captcha_data"
    image_width: int = 200
    image_height: int = 50
    label_length: int = 6
    characters: List[str] = field(default_factory=lambda: list(DIGITS))
    threshold: int = 255
    init: bool = False

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
                self.threshold,
            ) = self.get_train_info()

    def get_train_info(self) -> Tuple[str, str, str, int, int, int, List[str], int]:
        train_image_path = self.get_image_dir(train=True)
        pred_image_path = self.get_image_dir(train=False)
        model_path = self.get_model_path()
        train_files = self.get_data_files(train=True)
        iw, ih = self.image_width, self.image_height
        labels: List[str] = []
        characters: List[str] = list(ALPHA_NUMERIC)

        if train_files:
            last = train_files[-1]
            # Try Pillow first for robust detection
            try:
                from PIL import Image

                with Image.open(last) as im:
                    iw, ih = im.size
            except Exception:
                with open(last, "rb") as f:
                    f.seek(16)
                    data = f.read(8)
                    if len(data) >= 8:
                        iw, ih = struct.unpack("!II", data)

            labels = [os.path.basename(p).split(".")[0] for p in train_files]
            if labels:
                label_length = max(len(l) for l in labels)
                characters = sorted(set(ch for l in labels for ch in l))
            else:
                label_length = self.label_length
        else:
            label_length = self.label_length

        return (
            os.path.abspath(train_image_path),
            os.path.abspath(pred_image_path),
            os.path.abspath(model_path),
            int(iw),
            int(ih),
            int(label_length),
            characters,
            int(self.threshold),
        )

    def get_image_dir(self, train: bool = True) -> str:
        image_dir = os.path.join(
            self.train_data_base_dir,
            self.captcha_id,
            str(self.rev),
            "images",
            "train" if train else "pred",
        )
        return os.path.abspath(image_dir)

    def get_data_files(self, train: bool = True) -> List[str]:
        label_length = self.label_length
        image_dir = self.get_image_dir(train)
        if not os.path.exists(image_dir):
            return []
        all_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        # 파일명(확장자 제외) 길이가 label_length와 일치하는 것만 반환
        return [f for f in all_files if len(os.path.basename(f).split(".")[0]) == label_length]

    def get_labels(self, train: bool = True) -> List[str]:
        return [os.path.basename(p).split(".")[0] for p in self.get_data_files(train)]

    def get_model_base_dir(self) -> str:
        model_base_dir = os.path.join(self.train_data_base_dir, self.captcha_id, str(self.rev), "model")
        model_base_dir = os.path.abspath(model_base_dir)
        os.makedirs(model_base_dir, exist_ok=True)
        return model_base_dir

    def get_model_path(self) -> str:
        model_base_dir = self.get_model_base_dir()
        if self.backend == "pytorch":
            model_file_name = "model_full.pt"
        elif self.backend == "keras":
            model_file_name = "weights.keras"
        else:
            model_file_name = "weights.keras"
        return os.path.join(os.path.abspath(model_base_dir), model_file_name)

    def choice_pred_image(self) -> str:
        image_dir = self.get_image_dir(train=False)
        if not os.path.exists(image_dir):
            raise RuntimeError(f"No images directory: {image_dir}")
        candidates = sorted(glob.glob(os.path.join(image_dir, "*")))
        if not candidates:
            raise RuntimeError(f"No images found in {image_dir}")
        return random.choice(candidates)

    def shuffle_train_data(self, train_size=0.9) -> None:
        """Shuffle PNG files between train/pred directories on disk."""
        train_dir = self.get_image_dir(train=True)
        pred_dir = self.get_image_dir(train=False)

        all_images = []
        all_images.extend(glob.glob(os.path.join(train_dir, "*.png")))
        all_images.extend(glob.glob(os.path.join(pred_dir, "*.png")))

        if not all_images:
            raise RuntimeError("이미지 파일이 없습니다.")

        random.shuffle(all_images)
        split_index = int(len(all_images) * train_size)
        train_images = all_images[:split_index]
        pred_images = all_images[split_index:]

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        temp_dir = os.path.join(self.train_data_base_dir, self.captcha_id, str(self.rev), "images", "_temp")
        os.makedirs(temp_dir, exist_ok=True)

        for img_path in all_images:
            shutil.move(img_path, temp_dir)

        for img_path in train_images:
            filename = os.path.basename(img_path)
            shutil.move(os.path.join(temp_dir, filename), os.path.join(train_dir, filename))

        for img_path in pred_images:
            filename = os.path.basename(img_path)
            shutil.move(os.path.join(temp_dir, filename), os.path.join(pred_dir, filename))

        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

        print(f"셔플 완료: 훈련 {len(train_images)}개, 추론 {len(pred_images)}개")

    def image_pre_process(self, image: Image.Image) -> Image.Image:
        """
        이미지 전처리 (train/predict 공통 사용)
        - captcha_id별 특수 전처리 포함
        """
        # captcha_id별 특수 전처리
        if self.captcha_id == 'supreme_court':
            return self._supreme_court_preprocess(image)
        
        # 기본 전처리
        return self._default_preprocess(image)
    
    def _default_preprocess(self, image: Image.Image) -> Image.Image:
        """기본 이미지 전처리"""
        # 1. Transparent background -> White
        if image.mode == 'RGBA':
            background = Image.new('RGBA', image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert('RGB')
        
        # 2. Grayscale
        image = image.convert('L')
        
        # 3. Threshold (밝은 값은 흰색으로)
        if 0 < self.threshold < 255:
            image = image.point(lambda p: 255 if p > self.threshold else p)
            
        # 4. Resize to target size
        image = image.resize((self.image_width, self.image_height))            

        return image

    def _supreme_court_preprocess(self, image: Image.Image) -> Image.Image:
        """대법원 캡챠 전용 전처리"""
        if image.mode == 'RGBA':
            background = Image.new('RGBA', image.size, (255, 255, 255, 255))
            result = Image.alpha_composite(background, image)
        elif image.size[0] > self.image_width and image.size[1] > self.image_height:
            bg_color = 255
            crop = (3, 1, self.image_width-1, self.image_height-7)
            crop_image = image.crop(crop)
            result = Image.new("RGBA", (self.image_width, self.image_height), bg_color)
            result.paste(crop_image, (1, 1))
        else:
            result = image
        
        result = result.convert('RGB').convert('L')
        
        # 4. Resize to target size
        result = result.resize((self.image_width, self.image_height)) 
        
        return result

@dataclass
class CaptchaType:
    captcha_id: str = 'default'
    name: str = '기본캡챠'
    desc: str = '기본 캡챠'
    train_data: TrainData = field(default_factory=lambda: TrainData(captcha_id='default'))

    def __post_init__(self) -> None:
        self.captcha_id=self.train_data.captcha_id
