import numpy as np
import pytest

from speechloom import sampling as m


def test_softmax_uniform():
    assert m.softmax([0, 0]).tolist() == [0.5, 0.5]


def test_softmax_shift():
    assert m.softmax([1000, 1000]).tolist() == [0.5, 0.5]


def test_softmax_mask():
    assert m.softmax([0, -np.inf]).tolist() == [1, 0]


def test_topk_ties():
    assert np.flatnonzero(np.isfinite(m.top_k([1, 1, 1], 2))).tolist() == [0, 1]


def test_topp_prefix():
    assert np.flatnonzero(np.isfinite(m.top_p(np.log([0.6, 0.3, 0.1]), 0.8))).tolist() == [0, 1]


def test_topp_one():
    assert np.isfinite(m.top_p([0, 1, 2], 1)).tolist() == [True, True, True]


def test_greedy_tie():
    assert m.sample_token([1, 1], 0) == 0


def test_repetition_both_signs():
    assert m.repetition_penalty([2, -2, 0], [0, 1, 1], 2).tolist() == [1, -4, 0]


def test_allowlist():
    assert np.flatnonzero(np.isfinite(m.allowed_tokens([1, 2, 3], [0, 2]))).tolist() == [0, 2]


def test_empty_logits():
    with pytest.raises(ValueError):
        m.softmax([])


def test_nan_logits():
    with pytest.raises(ValueError):
        m.softmax([np.nan, 0])


def test_all_masked():
    with pytest.raises(ValueError):
        m.softmax([-np.inf, -np.inf])


def test_positive_infinity():
    with pytest.raises(ValueError):
        m.softmax([np.inf, 0])


def test_zero_temperature():
    with pytest.raises(ValueError):
        m.softmax([0], 0)


def test_negative_temperature():
    with pytest.raises(ValueError):
        m.sample_token([0], -1)


def test_zero_k():
    with pytest.raises(ValueError):
        m.top_k([1], 0)


def test_oversized_k():
    with pytest.raises(ValueError):
        m.top_k([1], 2)


def test_zero_p():
    with pytest.raises(ValueError):
        m.top_p([1], 0)


def test_high_p():
    with pytest.raises(ValueError):
        m.top_p([1], 1.1)


def test_low_penalty():
    with pytest.raises(ValueError):
        m.repetition_penalty([1], [], 0.5)


def test_unknown_previous():
    with pytest.raises(ValueError):
        m.repetition_penalty([1], [2])


def test_empty_allowlist():
    with pytest.raises(ValueError):
        m.allowed_tokens([1], [])


def test_masked_allowlist():
    with pytest.raises(ValueError):
        m.allowed_tokens([0, -np.inf], [1])


def test_seeded_sampling():
    assert m.sample_token([1, 2, 3], 1, 7) == m.sample_token([1, 2, 3], 1, 7)


def test_softmax_invariant_shift():
    assert m.softmax([1, 2, 3]) == pytest.approx(m.softmax([-9, -8, -7]))
