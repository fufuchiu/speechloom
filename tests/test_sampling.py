import numpy as np
import pytest

from speechloom import sampling as m


def test_softmax_uniform():
    assert m.softmax([0, 0]).tolist() == [0.5, 0.5]
