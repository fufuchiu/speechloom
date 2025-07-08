import numpy as np
import pytest

from speechloom import validation as m


def test_integer_rejects_0():
    with pytest.raises(ValueError):
        m.integer(-1)


def test_integer_rejects_1():
    with pytest.raises(ValueError):
        m.integer(True)
