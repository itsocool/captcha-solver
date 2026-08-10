"""ctc_beam_decode_fixed_length 자체 점검.

실행: uv run python test_ctc_decode.py
"""
from itertools import product

import numpy as np

from core import ctc_beam_decode_fixed_length, length_logprob


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


def _brute_force_length_logprob(log_probs: np.ndarray, expected_length: int) -> float:
    """모든 프레임 경로를 열거해 길이 L인 라벨열의 확률 합 (작은 입력 전용)."""
    probs = np.exp(log_probs)
    T, num_classes = log_probs.shape
    total = 0.0
    for path in product(range(num_classes), repeat=T):
        collapsed, prev = [], None
        for c in path:
            if c != prev and c != 0:
                collapsed.append(c)
            prev = c
        if len(collapsed) == expected_length:
            p = 1.0
            for t, c in enumerate(path):
                p *= probs[t, c]
            total += p
    return float(np.log(total)) if total > 0 else float("-inf")


def _random_log_probs(rng, shape) -> np.ndarray:
    logits = rng.normal(size=shape)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def test_length_logprob_matches_brute_force():
    """길이 분포 DP는 경로 완전 열거와 정확히 일치한다."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = int(rng.integers(1, 7))
        num_classes = int(rng.integers(2, 5))
        expected_length = int(rng.integers(1, 4))
        log_probs = _random_log_probs(rng, (T, num_classes))
        got = length_logprob(log_probs, expected_length)
        want = _brute_force_length_logprob(log_probs, expected_length)
        if np.isneginf(got) and np.isneginf(want):
            continue
        assert abs(got - want) < 1e-9, (got, want, T, num_classes, expected_length)


def test_length_distribution_sums_to_one():
    """가능한 모든 길이의 확률을 더하면 1이 된다."""
    rng = np.random.default_rng(1)
    log_probs = _random_log_probs(rng, (8, 4))
    total = np.exp(_brute_force_length_logprob(log_probs, 0))
    total += sum(np.exp(length_logprob(log_probs, L)) for L in range(1, 9))
    assert abs(total - 1.0) < 1e-9, total


def test_confidence_converges_as_beam_widens():
    """beam을 넓히면 신뢰도가 단조 증가하며 고정점에 수렴한다.

    분모(길이 L의 총 확률)는 beam과 무관하게 고정이고, 분자만 beam이 넓어질수록
    실제 P(y|x)에 가까워지기 때문이다. beam 집합으로 정규화하면 분모가 계속
    자라기 때문에 수렴하지 않고 값이 아래로 흘러내린다.
    """
    rng = np.random.default_rng(2)
    for _ in range(10):
        log_probs = _random_log_probs(rng, (16, 4))
        confs = [
            ctc_beam_decode_fixed_length(log_probs, MAPPING, expected_length=3, beam_width=w)[1]
            for w in (5, 10, 30, 60, 120)
        ]
        assert all(a <= b + 1e-12 for a, b in zip(confs, confs[1:])), confs
        assert abs(confs[-1] - confs[-2]) < 1e-9, confs


def _model_like_log_probs(rng, num_frames=30, label_length=6, num_classes=37, peak=0.99):
    """학습된 CTC 모델의 출력 모양을 흉내낸다.

    정해진 프레임에서만 문자가 튀어나오고 나머지는 blank 쪽으로 몰린다.
    """
    probs = np.full((num_frames, num_classes), (1.0 - peak) / (num_classes - 1))
    probs[:, 0] = peak
    for frame in np.linspace(1, num_frames - 2, label_length).astype(int):
        char = int(rng.integers(1, num_classes))
        probs[frame] = (1.0 - peak) / (num_classes - 1)
        probs[frame, char] = peak
    return np.log(probs / probs.sum(axis=1, keepdims=True))


def test_confidence_barely_moves_with_beam_width():
    """학습된 모델을 닮은 출력에서는 beam을 12배 넓혀도 신뢰도가 거의 그대로다.

    분모가 고정이라 남는 변동은 분자가 알짜 정렬 질량을 마저 긁어모으는 몫뿐이다.
    beam 집합으로 정규화하던 방식은 실제 데이터에서 같은 구간에 0.08 넘게 움직였다.
    """
    rng = np.random.default_rng(3)
    mapping = {0: ''}
    mapping.update({i: chr(96 + i) for i in range(1, 37)})
    for _ in range(10):
        log_probs = _model_like_log_probs(rng)
        confs = [
            ctc_beam_decode_fixed_length(log_probs, mapping, expected_length=6, beam_width=w)[1]
            for w in (10, 30, 60, 120)
        ]
        assert max(confs) - min(confs) < 0.01, confs


def test_ambiguous_input_is_not_confident():
    """어느 문자도 뚜렷하지 않으면 신뢰도가 낮아야 한다.

    beam 내부 정규화는 1위와 2위의 비율만 보므로 이런 입력에도 높은 값을 줄 수 있다.
    """
    log_probs = frames(*[[0.4, 0.2, 0.2, 0.2]] * 6)
    text, conf = ctc_beam_decode_fixed_length(log_probs, MAPPING, expected_length=3)
    assert len(text) == 3, text
    assert conf < 0.1, conf


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("모든 검사 통과")
