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
