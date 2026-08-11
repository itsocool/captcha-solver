"""모델 입력 크기는 생성자 기본값이 아니라 학습 이미지에서 따라와야 한다.

어긋나면 전처리(detected_* 로 리사이즈)와 모델 빌드(원본 필드)가 다른 크기를 보고
학습이 조용히 망가진다. 실제로 gov24 rev0(138x51)이 기본값 200x50 으로 빌드돼
Val Loss 가 정체하고 ONNX export 검증이 차원 불일치로 실패했다.

    python packages/python_3.12/hyperCaptcha/tests/test_detected_dims.py
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def test_dims_follow_images():
    from hypercaptcha.dataclass import CaptchaType, TrainData
    from hypercaptcha.core import PyTorchModel

    # 크기를 일부러 명시하지 않는다. TrainData 기본값은 200x50 이지만
    # gov24 rev0 의 실제 이미지는 138x51 이다.
    td = TrainData(captcha_id="gov24", rev=0,
                   train_data_base_dir=os.path.join(REPO_ROOT, "captcha_data"))
    assert (td.image_width, td.image_height) == (200, 50), "기본값 전제가 바뀌었다"
    assert (td.detected_image_width, td.detected_image_height) == (138, 51), \
        "gov24 rev0 학습 이미지가 138x51 이 아니다"

    model = PyTorchModel(
        captcha_type=CaptchaType(captcha_id="gov24", name="정부 24", desc="", train_data=td),
        verbose=0,
    )
    assert (model.image_width, model.image_height) == (138, 51), \
        f"모델이 기본값으로 빌드됐다: {model.image_width}x{model.image_height}"
    assert model.label_length == td.detected_label_length == 6


def test_existing_types_unaffected():
    """기존 캡차는 크기를 명시하고 있어 감지값과 같아야 한다 (같지 않으면 배포된 모델이 깨진다)."""
    from hypercaptcha import engine

    for cid, ct in engine.get_captcha_type_list(
            train_data_base_dir=os.path.join(REPO_ROOT, "captcha_data")).items():
        td = ct.train_data
        assert (td.image_width, td.image_height, td.label_length) == \
               (td.detected_image_width, td.detected_image_height, td.detected_label_length), \
               f"{cid}: 명시값과 감지값이 다르다 — 재학습 없이 배포하면 안 된다"


if __name__ == "__main__":
    test_dims_follow_images()
    test_existing_types_unaffected()
    print("OK", file=sys.stderr)
