import numpy as np
import pytest

from speechloom import batching as m


def test_padding_mask():
    assert m.padding_mask([1, 3], 4).tolist() == [
        [False, True, True, True],
        [False, False, False, True],
    ]


def test_causal_mask():
    assert m.causal_mask(3).tolist() == [
        [False, True, True],
        [False, False, True],
        [False, False, False],
    ]


def test_weighted_labels():
    assert m.loss_weights(np.array([[4, 260, -100]]), text_weight=1, audio_weight=2).tolist() == [
        [1, 2, 0]
    ]


def test_budget_padded():
    assert m.token_budget_batches([2, 5, 2], 10) == [[0, 1], [2]]


def test_budget_empty():
    assert m.token_budget_batches([], 4) == []


def test_buckets_edges():
    assert m.length_buckets([2, 3, 5, 8], (2, 5)) == {0: [0], 1: [1, 2], 2: [3]}


def test_empty_lengths():
    with pytest.raises(ValueError):
        m.padding_mask([])


def test_zero_lengths():
    with pytest.raises(ValueError):
        m.padding_mask([0])


def test_too_short_padding():
    with pytest.raises(ValueError):
        m.padding_mask([5], 4)


def test_zero_causal():
    with pytest.raises(ValueError):
        m.causal_mask(0)


def test_empty_sequences():
    with pytest.raises(ValueError):
        m.shift_targets([], 10)


def test_short_sequence():
    with pytest.raises(ValueError):
        m.shift_targets([[1]], 10)


def test_pad_in_sequence():
    with pytest.raises(ValueError):
        m.shift_targets([[1, 0, 2]], 10)


def test_out_of_vocab():
    with pytest.raises(ValueError):
        m.shift_targets([[1, 11]], 10)


def test_positive_ignore():
    with pytest.raises(ValueError):
        m.shift_targets([[1, 2]], 10, ignore_index=0)


def test_negative_weight():
    with pytest.raises(ValueError):
        m.loss_weights([1], text_weight=-1)


def test_float_labels():
    with pytest.raises(ValueError):
        m.loss_weights([1.0])


def test_zero_budget():
    with pytest.raises(ValueError):
        m.token_budget_batches([1], 0)


def test_oversized_sequence():
    with pytest.raises(ValueError):
        m.token_budget_batches([11], 10)


def test_unsorted_buckets():
    with pytest.raises(ValueError):
        m.length_buckets([1], (5, 2))


def test_duplicate_buckets():
    with pytest.raises(ValueError):
        m.length_buckets([1], (2, 2))
