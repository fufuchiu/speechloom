import numpy as np
import pytest

from speechloom import validation as m


def test_integer_rejects_0():
    with pytest.raises(ValueError):
        m.integer(-1)


def test_integer_rejects_1():
    with pytest.raises(ValueError):
        m.integer(True)


def test_integer_rejects_2():
    with pytest.raises(ValueError):
        m.integer(1.5)


def test_integer_rejects_3():
    with pytest.raises(ValueError):
        m.integer('2')


def test_integer_rejects_4():
    with pytest.raises(ValueError):
        m.integer(None)
