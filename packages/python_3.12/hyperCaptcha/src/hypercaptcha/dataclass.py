import os
import glob
import functools
import random
import struct
import shutil
import string
from PIL import Image
from typing import Final, List, Tuple
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr


DIGITS: Final = string.digits
LOWER_CASE: Final = string.ascii_lowercase
UPPER_CASE: Final = string.ascii_uppercase
ALPHABET: Final = string.ascii_letters
ALPHA_NUMERIC: Final = string.digits + string.ascii_letters


class _TrainInfo(BaseModel):
    """Auto-detected training information."""
    model_config = ConfigDict(frozen=True)

    image_width: int
    image_height: int
    label_length: int
    characters: str
    threshold: int


class CaptchaType(BaseModel):
    captcha_id: str
    name: str
    desc: str
    train_data: "TrainData"


class TrainData(BaseModel):
    captcha_id: str
    backend: str = "pytorch"
    rev: int = 0
    train_data_base_dir: str = "./captcha_data"
    image_width: int = 200
    image_height: int = 50
    label_length: int = 6
    characters: List[str] = Field(default_factory=list)
    threshold: int = 255
    _train_info: _TrainInfo | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._detect_and_cache()

    def _detect_and_cache(self) -> None:
        """Detect train info from files and cache as a _train_info attribute."""
        train_dir = self.get_image_dir(train=True)
        all_files = sorted(glob.glob(os.path.join(train_dir, "*.png")))
        labels = [os.path.basename(p).split(".")[0] for p in all_files]

        if not labels:
            self._train_info = None
            return

        # Detect image size from the last file
        iw, ih, threshold = self.image_width, self.image_height, self.threshold
        last = all_files[-1]
        try:
            with Image.open(last) as im:
                iw, ih = im.size
        except Exception:
            try:
                with open(last, "rb") as f:
                    f.seek(16)
                    data_bytes = f.read(8)
                    if len(data_bytes) >= 8:
                        iw, ih = struct.unpack("!II", data_bytes)
            except Exception:
                pass

        max_len = max(len(l) for l in labels)
        chars = "".join(sorted(set(ch for l in labels for ch in l)))

        info = _TrainInfo(
            image_width=int(iw),
            image_height=int(ih),
            label_length=int(max_len),
            characters=chars,
            threshold=int(threshold)
        )
        self._train_info = info

    # --- detected values (auto from files, fallback to constructor default) ---

    @property
    def info(self) -> _TrainInfo | None:
        """Cached detection result, or None if no training files exist."""
        return getattr(self, "_train_info", None)

    @property
    def detected_image_width(self) -> int:
        return self.info.image_width if self.info else self.image_width

    @property
    def detected_image_height(self) -> int:
        return self.info.image_height if self.info else self.image_height

    @property
    def detected_label_length(self) -> int:
        return self.info.label_length if self.info else self.label_length

    @property
    def detected_characters(self) -> str:
        return self.info.characters if self.info else "".join(self.characters)

    # --- paths ---

    def get_image_dir(self, train: bool = True) -> str:
        return os.path.abspath(
            os.path.join(
                self.train_data_base_dir,
                self.captcha_id,
                str(self.rev),
                "images",
                "train" if train else "pred",
            )
        )

    def get_data_files(self, train: bool = True) -> List[str]:
        image_dir = self.get_image_dir(train)
        if not os.path.exists(image_dir):
            return []
        all_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        label_length = self.detected_label_length
        return [f for f in all_files if len(os.path.basename(f).split(".")[0]) == label_length]

    def get_labels(self, train: bool = True) -> List[str]:
        return [os.path.basename(p).split(".")[0] for p in self.get_data_files(train)]

    def get_model_base_dir(self) -> str:
        model_base_dir = os.path.abspath(
            os.path.join(self.train_data_base_dir, self.captcha_id, str(self.rev), "model")
        )
        os.makedirs(model_base_dir, exist_ok=True)
        return model_base_dir

    def get_model_path(self) -> str:
        return os.path.join(self.get_model_base_dir(), "model_full.pt")

    def get_train_info(self) -> Tuple[str, str, str, int, int, int, List[str], int]:
        iw = self.detected_image_width
        ih = self.detected_image_height
        ll = self.detected_label_length
        chars = list(self.detected_characters)
        return (
            os.path.abspath(self.get_image_dir(train=True)),
            os.path.abspath(self.get_image_dir(train=False)),
            os.path.abspath(self.get_model_path()),
            iw,
            ih,
            ll,
            chars,
            self.detected_label_length,
        )

    def choice_pred_image(self) -> str:
        image_dir = self.get_image_dir(train=False)
        if not os.path.exists(image_dir):
            raise RuntimeError(f"No images directory: {image_dir}")
        candidates = sorted(glob.glob(os.path.join(image_dir, "*")))
        if not candidates:
            raise RuntimeError(f"No images found in {image_dir}")
        return random.choice(candidates)

    def shuffle_train_data(self, train_size: float = 0.9) -> None:
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
        if self.captcha_id == "supreme_court":
            return self._supreme_court_preprocess(image)
        return self._default_preprocess(image)

    def _default_preprocess(self, image: Image.Image) -> Image.Image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        image = image.convert("L")
        if 0 < self.threshold < 255:
            image = image.point(lambda p: 255 if p > self.threshold else p)
        image = self._remove_border(image)
        image = self._make_background_white(image)
        image = image.resize((self.detected_image_width, self.detected_image_height))
        return image

    def _supreme_court_preprocess(self, image: Image.Image) -> Image.Image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            result = Image.alpha_composite(background, image)
        elif image.size[0] > self.detected_image_width and image.size[1] > self.detected_image_height:
            crop = (3, 1, self.detected_image_width - 1, self.detected_image_height - 7)
            crop_image = image.crop(crop)
            result = Image.new("RGBA", (self.detected_image_width, self.detected_image_height), 255)
            result.paste(crop_image, (1, 1))
        else:
            result = image
        result = result.convert("RGB").convert("L")
        result = self._remove_border(result)
        result = self._make_background_white(result)
        result = result.resize((self.detected_image_width, self.detected_image_height))
        return result

    def _remove_border(self, image: Image.Image, margin: int = 2) -> Image.Image:
        w, h = image.size
        left, top = margin, margin
        right, bottom = w - margin, h - margin
        if right <= left or bottom <= top:
            return image
        return image.crop((left, top, right, bottom))

    def _make_background_white(self, image: Image.Image) -> Image.Image:
        pixels = image.load()
        w, h = image.size
        for y in range(h):
            for x in range(w):
                if pixels[x, y] > 128:
                    pixels[x, y] = 255
        return image
