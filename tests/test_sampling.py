import numpy as np
import pytest

from speechloom import sampling as m


def test_softmax_uniform():
    assert m.softmax([0, 0]).tolist() == [0.5, 0.5]


def test_softmax_shift():
    assert m.softmax([1000, 1000]).tolist() == [0.5, 0.5]


def test_softmax_mask():
    assert m.softmax([0, -np.inf]).tolist() == [1, 0]
