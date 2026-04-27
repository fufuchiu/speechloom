import pytest

from speechloom import metrics as m


def test_percentile_median():
    assert m.percentile([0, 10], 0.5) == 5
