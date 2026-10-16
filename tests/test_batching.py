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


def test_teacher_forcing_shift():
    batch = m.shift_targets([[1, 4, 5, 2], [1, 2]], 10)
    assert batch['input_ids'].tolist() == [[1, 4, 5], [1, 0, 0]]
    assert batch['labels'].tolist() == [[4, 5, 2], [2, -100, -100]]
    assert batch['lengths'].tolist() == [3, 1]


def test_padded_budget_invariant():
    lengths = [1, 9, 2, 2, 8, 4, 7, 3]
    batches = m.token_budget_batches(lengths, 16)
    assert [i for batch in batches for i in batch] == list(range(len(lengths)))
    assert all(max(lengths[i] for i in batch) * len(batch) <= 16 for batch in batches)


def test_padding_inverse_lengths():
    lengths = [1, 3, 5]
    assert (~m.padding_mask(lengths)).sum(1).tolist() == lengths



def test_loss_weights_all_text():
    """Labels below audio_offset all receive text_weight."""
    labels = np.array([4, 100, 259])
    assert m.loss_weights(labels, text_weight=3, audio_weight=5).tolist() == [3, 3, 3]


def test_loss_weights_all_audio():
    """Labels at or above audio_offset all receive audio_weight."""
    labels = np.array([260, 300, 515])
    assert m.loss_weights(labels, text_weight=3, audio_weight=5).tolist() == [5, 5, 5]


def test_loss_weights_all_ignored():
    """Negative labels all receive weight zero."""
    labels = np.array([-100, -1, -50])
    assert m.loss_weights(labels).tolist() == [0, 0, 0]


def test_loss_weights_zero_text():
    """Zero text_weight supervises audio only."""
    labels = np.array([4, 260, -100])
    assert m.loss_weights(labels, text_weight=0, audio_weight=1).tolist() == [0, 1, 0]


def test_loss_weights_zero_audio():
    """Zero audio_weight supervises text only."""
    labels = np.array([4, 260, -100])
    assert m.loss_weights(labels, text_weight=1, audio_weight=0).tolist() == [1, 0, 0]


def test_loss_weights_custom_offset():
    """Non-default audio_offset shifts the text/audio boundary."""
    labels = np.array([4, 10, 20])
    assert m.loss_weights(labels, audio_offset=10, text_weight=1, audio_weight=2).tolist() == [1, 2, 2]


def test_loss_weights_2d():
    """Two-dimensional label arrays are handled correctly."""
    labels = np.array([[4, 260], [-100, 300]])
    result = m.loss_weights(labels, text_weight=1, audio_weight=2)
    assert result.tolist() == [[1, 2], [0, 2]]
