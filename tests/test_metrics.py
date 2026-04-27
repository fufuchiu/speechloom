import pytest

from speechloom import metrics as m


def test_percentile_median():
    assert m.percentile([0, 10], 0.5) == 5


def test_percentile_min():
    assert m.percentile([2, 1], 0) == 1


def test_percentile_max():
    assert m.percentile([2, 1], 1) == 2
