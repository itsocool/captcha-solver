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

# kshop 원본은 263x54 다. 배경이 왼쪽 검정 → 오른쪽 흰색 가로 그라데이션이고 숫자는
# 그 위에 검게 그려져서, 왼쪽 몇 자리는 배경과 거의 같은 색이 된다. 그래서 관심 영역만
# 잘라내고 그라데이션을 걷어낸다.
#
# 크롭 박스는 PIL 규약대로 (left, top, right, bottom) 이고 결과는 166x48 이다.
# 실측(train 500장) 기준 각 변에서 잘려나가는 양:
#   좌  10  손실 없음 (숫자 획은 x>=16 부터)
#   상   2  손실 없음 (y=0 은 테두리)
#   우 176  손실 없음 (획 최우측이 정확히 176)
#   하  50  숫자 획이 조금 잘리는 이미지 17/500 (3.4%) — 획 최하단은 52 까지 간다
# y=0 과 y=53 의 테두리, x>=176 의 방해용 사선은 모두 빠진다.
KSHOP_SOURCE_SIZE: Final = (263, 54)
KSHOP_CROP: Final = (10, 2, 176, 50)


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

    def build_meta(self) -> dict:
        """모델 옆에 놓이는 사이드카 메타데이터.

        모델 파일만으로는 문자셋·레이블 길이·전처리 종류를 알 수 없어서 Rust CLI /
        Spring Boot / ConsoleApp 이 이 JSON 을 함께 읽는다. 학습이 `model.meta.json` 으로
        저장하고, `sync_models.py` 가 배포용으로 같은 내용을 복사한다.
        """
        td = self.train_data
        return {
            "captcha_id": self.captcha_id,
            "name": self.name,
            "rev": td.rev,
            "image_width": td.detected_image_width,
            "image_height": td.detected_image_height,
            "label_length": td.detected_label_length,
            "characters": td.detected_characters,
            "threshold": td.threshold,
            "preprocess": td.preprocess,
            "blank_index": 0,
        }


class TrainData(BaseModel):
    captcha_id: str
    backend: str = "pytorch"
    rev: int = 0
    # 전처리 종류. 모델 옆 meta.json 에 그대로 실려 Rust CLI / Spring Boot /
    # ConsoleApp 이 같은 값을 보고 같은 전처리를 재현한다.
    preprocess: str = "default"
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

        # 모델 입력 크기는 파일 크기가 아니라 **전처리를 거친 뒤**의 크기다.
        # 크롭하는 전처리(kshop)가 생기면서 둘이 갈렸다. 한 장 돌려서 확정한다.
        # 위에서 _train_info 를 먼저 채워 둔 건 _default_preprocess 의 마지막 resize 가
        # detected_* 를 참조하기 때문이다 (여기서 비워두면 순환한다).
        try:
            with Image.open(last) as im:
                pw, ph = self.image_pre_process(im.copy()).size
        except Exception:
            return
        if (pw, ph) != (int(iw), int(ih)):
            self._train_info = _TrainInfo(
                image_width=int(pw),
                image_height=int(ph),
                label_length=int(max_len),
                characters=chars,
                threshold=int(threshold)
            )

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
        """학습 체크포인트. `torch.save(state_dict)` 로 저장하고 파이썬 추론이 읽는다."""
        return os.path.join(self.get_model_base_dir(), "model.pth")

    def get_export_path(self) -> str:
        """`torch.export` 산출물. 그래프와 가중치를 함께 담은 PyTorch 2 아카이브."""
        return os.path.join(self.get_model_base_dir(), "model.pt2")

    def get_onnx_path(self) -> str:
        """ONNX 산출물. Rust CLI / Spring Boot / ConsoleApp 이 읽는다."""
        return os.path.join(self.get_model_base_dir(), "model.onnx")

    def get_meta_path(self) -> str:
        """사이드카 메타데이터. 내용은 `CaptchaType.build_meta()` 가 만든다."""
        return os.path.join(self.get_model_base_dir(), "model.meta.json")

    def get_ort_path(self) -> str:
        """ORT 포맷 산출물. 최적화까지 끝난 flatbuffer 라 로드가 빠르고 minimal build 도 읽는다."""
        return os.path.join(self.get_model_base_dir(), "model.ort")

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
        if self.preprocess == "supreme_court":
            return self._supreme_court_preprocess(image)
        if self.preprocess == "kshop":
            return self._kshop_preprocess(image)
        return self._default_preprocess(image)

    def _kshop_preprocess(self, image: Image.Image) -> Image.Image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        image = image.convert("L")

        # 추론 입력이 원본 크기가 아닐 수 있어 기준 크기로 맞춘 뒤 자른다.
        if image.size != KSHOP_SOURCE_SIZE:
            image = image.resize(KSHOP_SOURCE_SIZE)
        image = image.crop(KSHOP_CROP)

        return self._flatten_column_background(image)

    @staticmethod
    def _flatten_column_background(image: Image.Image) -> Image.Image:
        """세로 방향으로 균일한 배경 그라데이션을 걷어내 배경을 흰색으로 만든다.

        배경이 x 에만 의존하므로 열마다 가장 밝은 값을 그 열의 배경으로 보고 255 가
        되도록 스케일한다. 잉크(0 부근)는 0 근처에 남아 대비가 전 구간에서 복원된다.
        실측상 왼쪽 끝 배경은 74, 오른쪽 끝은 189 인데 잉크는 어디서나 0 이다.

        ponytail: 열 전체가 획으로 덮이면 그 열의 배경 추정이 획 값이 되어 하얗게
                  날아간다. kshop 은 숫자가 y=6~39 라 위아래에 배경이 남아 안전하다.
                  획이 세로로 꽉 차는 캡차가 생기면 열별 최댓값 대신 상위 분위수로 바꿀 것.
        """
        import numpy as np

        a = np.asarray(image, dtype=np.float32)
        bg = np.maximum(a.max(axis=0, keepdims=True), 1.0)
        out = np.clip(a / bg * 255.0, 0.0, 255.0)
        return Image.fromarray(out.astype(np.uint8), mode="L")

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
