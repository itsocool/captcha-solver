import os, glob, random, struct, shutil
from dataclasses import dataclass, field
from typing import Optional, Final

DIGITS: Final = "0123456789"
LOWER_CASE: Final = "abcdefghijklmnopqrstuvwxyz"
UPPER_CASE: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALPHABET: Final = LOWER_CASE + UPPER_CASE
ALPHA_NUMERIC: Final = DIGITS + ALPHABET

@dataclass
class TrainData:
    captcha_id: str
    backend: str = 'pytorch'
    rev: int = 0
    train_data_base_dir: str = "./captcha_data"
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

    def get_train_info(self) -> tuple[str, str, str, int, int, int, list[str], int]:
        train_image_path = self.get_image_dir(train=True)
        pred_image_path = self.get_image_dir(train=False)
        model_path = self.get_model_path()
        train_data_list = self.get_data_files(train=True)
        with open(train_data_list[-1], 'rb') as f:
            f.read(16)
            data = f.read(8)
            if len(data) < 8:
                raise ValueError("파일이 너무 짧거나 PNG 형식이 아닙니다.")
            image_width, image_height = struct.unpack('!II', data)
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
            self.train_data_base_dir, 
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

    def get_model_base_dir(self) -> str:
        model_base_dir = os.path.join(self.train_data_base_dir, self.captcha_id, str(self.rev), 'model')
        model_base_dir = os.path.abspath(model_base_dir)
        if not os.path.exists(model_base_dir):
            os.makedirs(model_base_dir, exist_ok=True)
        return model_base_dir

    def get_model_path(self) -> str:
        model_base_dir = self.get_model_base_dir()
        model_path = os.path.abspath(model_base_dir)
        if self.backend == 'pytorch':
            model_file_name = "model_full.pth"
        elif self.backend == 'keras':
            model_file_name = "weights.keras"
        model_path = os.path.join(model_path, model_file_name)
        return model_path

    def choice_pred_image(self) -> str:
        image_dir = self.get_image_dir(train=False)
        if isinstance(image_dir, (list, tuple)):
            candidates = [p for p in image_dir if os.path.isfile(p)]
        else:
            candidates = glob.glob(os.path.join(image_dir, '*'))
        if not candidates:
            raise RuntimeError(f"No images found in {image_dir}")
        image_path = random.choice(candidates)
        return image_path

    def shuffle_train_data(self, train_size=0.9) -> None:
        """
        훈련용/추론용 폴더의 모든 PNG 이미지를 수집하여 셔플한 뒤,
        train_size 비율에 따라 다시 배분합니다.
        """
        train_dir = self.get_image_dir(train=True)
        pred_dir = self.get_image_dir(train=False)
        
        # 두 폴더의 모든 PNG 이미지 수집
        all_images = []
        all_images.extend(glob.glob(os.path.join(train_dir, '*.png')))
        all_images.extend(glob.glob(os.path.join(pred_dir, '*.png')))
        
        if not all_images:
            raise RuntimeError("이미지 파일이 없습니다.")
        
        # 셔플
        random.shuffle(all_images)
        
        # train_size 비율로 분할
        split_index = int(len(all_images) * train_size)
        train_images = all_images[:split_index]
        pred_images = all_images[split_index:]
        
        # 폴더 생성 (존재하지 않는 경우)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        
        # 임시 폴더로 모든 이미지 이동
        temp_dir = os.path.join(self.train_data_base_dir, self.captcha_id, str(self.rev), 'images', '_temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        for img_path in all_images:
            shutil.move(img_path, temp_dir)
        
        # 훈련 폴더로 이동
        for img_path in train_images:
            filename = os.path.basename(img_path)
            shutil.move(os.path.join(temp_dir, filename), os.path.join(train_dir, filename))
        
        # 추론 폴더로 이동
        for img_path in pred_images:
            filename = os.path.basename(img_path)
            shutil.move(os.path.join(temp_dir, filename), os.path.join(pred_dir, filename))
        
        # 임시 폴더 삭제
        os.rmdir(temp_dir)
        
        print(f"셔플 완료: 훈련 {len(train_images)}개, 추론 {len(pred_images)}개")

@dataclass
class CaptchaType:
    captcha_id: str = 'default'
    name: str = '기본캡챠'
    desc: str = '기본 캡챠'
    train_data: Optional[TrainData] = None

    def __post_init__(self) -> None:
        self.captcha_id=self.train_data.captcha_id
