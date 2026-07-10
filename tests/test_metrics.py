import pytest

from speechloom import metrics as m


def test_percentile_median():
    assert m.percentile([0, 10], 0.5) == 5


def test_percentile_min():
    assert m.percentile([2, 1], 0) == 1


def test_percentile_max():
    assert m.percentile([2, 1], 1) == 2


def test_rtf():
    assert m.real_time_factor(0.5, 2) == 0.25


def test_rtf_zero():
    assert m.real_time_factor(0, 2) == 0


def test_first_token():
    assert m.first_token_latency(10, 10.25) == 0.25


def test_gap_equal():
    assert m.inter_token_gaps([1, 1, 2]).tolist() == [0, 1]


def test_gap_empty():
    assert m.inter_token_gaps([]).tolist() == []


def test_snr_identical():
    assert m.waveform_snr([1, 2], [1, 2]) == float('inf')


def test_snr_zero_ref():
    assert m.waveform_snr([0], [1]) == -float('inf')


def test_snr_unit():
    assert m.waveform_snr([1], [0]) == 0


def test_bootstrap_constant():
    assert m.bootstrap_mean([4, 4, 4], repetitions=20) == (4, 4)


def test_accuracy_ignore():
    assert m.token_accuracy([1, 9, 4], [1, -100, 3]) == 0.5


def test_accuracy_all_ignored():
    assert m.token_accuracy([1], [-100]) == 0


def test_empty_percentile():
    with pytest.raises(ValueError):
        m.percentile([])


def test_invalid_quantile():
    with pytest.raises(ValueError):
        m.percentile([1], 1.1)


def test_negative_elapsed():
    with pytest.raises(ValueError):
        m.real_time_factor(-1, 1)


def test_zero_audio_duration():
    with pytest.raises(ValueError):
        m.real_time_factor(1, 0)


def test_reversed_clock():
    with pytest.raises(ValueError):
        m.first_token_latency(2, 1)


def test_nonmonotonic_tokens():
    with pytest.raises(ValueError):
        m.inter_token_gaps([2, 1])


def test_empty_snr():
    with pytest.raises(ValueError):
        m.waveform_snr([], [])


def test_unaligned_snr():
    with pytest.raises(ValueError):
        m.waveform_snr([1, 2], [1])


def test_empty_bootstrap():
    with pytest.raises(ValueError):
        m.bootstrap_mean([])


def test_zero_confidence():
    with pytest.raises(ValueError):
        m.bootstrap_mean([1], confidence=0)


def test_unit_confidence():
    with pytest.raises(ValueError):
        m.bootstrap_mean([1], confidence=1)
