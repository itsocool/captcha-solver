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


def test_model_input_matches_preprocessed_size():
    """모델 입력 크기는 '전처리를 거친 뒤'의 크기와 같아야 한다.

    kshop 전처리가 263x54 를 180x50 으로 자르면서 파일 크기와 모델 입력이 갈렸다.
    여기서 어긋나면 전처리는 A 크기로 주는데 모델은 B 크기로 만들어져,
    학습이 조용히 망가지거나 ONNX export 검증이 차원 불일치로 죽는다.
    meta.json 도 같은 값을 실어야 Rust CLI / Spring Boot / WinConsoleApp 이 재현할 수 있다.
    """
    import glob

    from PIL import Image

    from hypercaptcha import engine

    for cid, ct in engine.get_captcha_type_list(
            train_data_base_dir=os.path.join(REPO_ROOT, "captcha_data")).items():
        td = ct.train_data
        sample = sorted(glob.glob(os.path.join(td.get_image_dir(train=True), "*.png")))[0]
        with Image.open(sample) as im:
            processed = td.image_pre_process(im.copy()).size

        assert processed == (td.detected_image_width, td.detected_image_height), \
            f"{cid}: 전처리 결과 {processed} 와 모델 입력 크기 " \
            f"{(td.detected_image_width, td.detected_image_height)} 가 다르다"

        meta = ct.build_meta()
        assert (meta["image_width"], meta["image_height"]) == processed, \
            f"{cid}: meta 크기가 전처리 결과와 다르다 — 다른 언어 클라이언트가 틀린 크기로 추론한다"
        assert meta["preprocess"] == td.preprocess, f"{cid}: meta preprocess 불일치"


def test_patience_outlasts_ctc_plateau():
    """조기 종료 기본값이 CTC 초기 정체 구간보다 길어야 한다.

    kshop rev0 실측: epoch 2~28 이 정체이고 그 안에서 무개선이 7에폭 연속까지 난다.
    patience 가 그보다 짧으면 탈출 전에 학습이 죽는다 (실제로 6 일 때 epoch 20 종료).
    """
    sys.path.insert(0, os.path.join(REPO_ROOT, "apps"))
    from web.services.train import PARAM_SPEC

    OBSERVED_PLATEAU_STALL = 7

    for name, (low, high, default) in PARAM_SPEC.items():
        assert low <= default <= high, f"{name}: 기본값 {default} 이 범위 밖이다"

    patience = PARAM_SPEC["early_stopping_patience"][2]
    assert patience > OBSERVED_PLATEAU_STALL * 2 - 1, (
        f"early_stopping_patience 기본값 {patience} 는 정체 구간 무개선 "
        f"{OBSERVED_PLATEAU_STALL} 에폭을 견디기에 짧다"
    )
    assert PARAM_SPEC["epochs"][2] >= 60, "정체 탈출(약 29에폭) 뒤 수렴할 여유가 없다"


if __name__ == "__main__":
    test_dims_follow_images()
    test_model_input_matches_preprocessed_size()
    test_patience_outlasts_ctc_plateau()
    print("OK", file=sys.stderr)
