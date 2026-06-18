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
