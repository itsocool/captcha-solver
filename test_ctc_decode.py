"""ctc_beam_decode_fixed_length 자체 점검.

실행: uv run python test_ctc_decode.py
"""
import numpy as np

from hypercaptcha.core import ctc_beam_decode_fixed_length


MAPPING = {0: '', 1: 'a', 2: 'b', 3: 'c'}


def frames(*rows) -> np.ndarray:
    """확률 행 -> log 확률 (T, C). 각 행은 [blank, a, b, c]."""
    probs = np.array(rows, dtype=np.float64)
    assert np.allclose(probs.sum(axis=1), 1.0), "각 프레임 확률 합이 1이어야 합니다"
    return np.log(np.clip(probs, 1e-12, None))


def test_confident_prediction():
    """또렷한 입력은 정답 문자열과 1.0에 가까운 신뢰도를 준다."""
    log_probs = frames(
        [0.01, 0.97, 0.01, 0.01],   # a
        [0.97, 0.01, 0.01, 0.01],   # blank
        [0.01, 0.01, 0.97, 0.01],   # b
    )
    text, conf = ctc_beam_decode_fixed_length(log_probs, MAPPING, expected_length=2)
    assert text == "ab", text
    assert conf > 0.95, conf


def test_tie_gives_half_confidence():
    """두 후보가 동률이면 신뢰도는 0.5 부근."""
    log_probs = frames(
        [0.0, 0.5, 0.5, 0.0],       # a 또는 b
        [1.0, 0.0, 0.0, 0.0],       # blank
    )
    text, conf = ctc_beam_decode_fixed_length(log_probs, MAPPING, expected_length=1)
    assert text in ("a", "b"), text
    assert abs(conf - 0.5) < 1e-6, conf


def test_sums_over_alignments():
    """같은 문자열로 수렴하는 여러 정렬 경로의 확률을 합산한다.

    'a': (a,a)=0.25 + (a,blank)=0.25 = 0.5
    'b': (b,blank)=0.25            = 0.25   ((b,a)는 길이 2라 제외)
    → 0.5 / 0.75 = 2/3. 경로 최댓값만 쓰면 0.25/0.5 = 0.5가 나온다.
    """
    log_probs = frames(
        [0.0, 0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0, 0.0],
    )
    text, conf = ctc_beam_decode_fixed_length(log_probs, MAPPING, expected_length=1)
    assert text == "a", text
    assert abs(conf - 2 / 3) < 1e-6, conf


def test_repeated_characters_need_blank():
    """반복 문자는 사이에 blank가 있어야 두 글자로 디코딩된다."""
    log_probs = frames(
        [0.01, 0.97, 0.01, 0.01],   # a
        [0.97, 0.01, 0.01, 0.01],   # blank
        [0.01, 0.97, 0.01, 0.01],   # a
    )
    text, conf = ctc_beam_decode_fixed_length(log_probs, MAPPING, expected_length=2)
    assert text == "aa", text
    assert conf > 0.9, conf


def test_fixed_length_is_enforced():
    """모든 프레임이 blank 쪽으로 기울어도 기대 길이를 채워 반환한다."""
    log_probs = frames(*[[0.7, 0.1, 0.1, 0.1]] * 6)
    text, conf = ctc_beam_decode_fixed_length(log_probs, MAPPING, expected_length=3)
    assert len(text) == 3, text
    assert 0.0 <= conf <= 1.0, conf


def test_confidence_bounds():
    """신뢰도는 항상 [0, 1] 범위."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        logits = rng.normal(size=(12, 4))
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        text, conf = ctc_beam_decode_fixed_length(log_probs, MAPPING, expected_length=4)
        assert len(text) == 4, text
        assert 0.0 <= conf <= 1.0, conf


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("모든 검사 통과")
