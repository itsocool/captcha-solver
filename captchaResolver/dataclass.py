import os, glob
from PIL import Image
from dataclasses import dataclass, field
import captchaResolver.consts as consts 

@dataclass
class TrainInfo:
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
    characters:list = field(default_factory=lambda: list(consts.ALPHA_NUMERIC))
    init:bool = True
    threshold: int = 0

    def __post_init__(self):
        if self.init == True:
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

    def get_train_info(self):
        train_image_path = self.get_image_dir(train=True)
        pred_image_path = self.get_image_dir(train=False)
        model_path = self.get_model_path()
        
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

    def get_image_dir(self, train=True):
        image_dir = os.path.join(self.base_dir, self.id, str(self.rev), 'images', 'train' if train else 'pred')
        image_dir = os.path.abspath(image_dir)
        return image_dir

    def get_data_files(self, train=True):
        image_dir = self.get_image_dir(train)
        return glob.glob(os.path.join(image_dir, '*.png'))

    def get_labels(self, train=True):
        return [
            os.path.basename(data_path).split(".")[0]
            for data_path in self.get_data_files(train)
        ]

    def get_model_path(self, weights_only=False):
        model_path = os.path.join(self.base_dir, self.id, str(self.rev), 'model')
        model_path = os.path.abspath(model_path)

        if os.path.exists(model_path) == False:
            os.makedirs(model_path)

        if weights_only:
            model_path = os.path.join(model_path, ".weights.h5")
        else:
            model_path = os.path.join(model_path, "weights.keras")

        return model_path

@dataclass
class CaptchaType:
    id: str = 'default'
    name: str = '기본캡챠'
    desc: str = '기본 캡챠'
    base_dir: str = './captcha_data'
    train_data: TrainInfo = None

    def __post_init__(self):
        self.train_data = TrainInfo(id=self.id, desc=self.desc + ' 학습 데이타', base_dir=self.base_dir)
