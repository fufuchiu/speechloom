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


def test_real_rejects_0():
    with pytest.raises(ValueError):
        m.real(True)


def test_real_rejects_1():
    with pytest.raises(ValueError):
        m.real('1')


def test_real_rejects_2():
    with pytest.raises(ValueError):
        m.real(None)


def test_real_rejects_3():
    with pytest.raises(ValueError):
        m.real(float('nan'))


def test_real_rejects_4():
    with pytest.raises(ValueError):
        m.real(float('inf'))


def test_real_rejects_5():
    with pytest.raises(ValueError):
        m.real(-float('inf'))


def test_vector_rejects_0():
    with pytest.raises(ValueError):
        m.vector([[1]])


def test_vector_rejects_1():
    with pytest.raises(ValueError):
        m.vector(1)


def test_vector_rejects_2():
    with pytest.raises(ValueError):
        m.vector([float('nan')])


def test_vector_rejects_3():
    with pytest.raises(ValueError):
        m.vector([float('inf')])


def test_integer_zero():
    assert m.integer(0) == 0


def test_numpy_integer():
    assert m.integer(np.int64(4)) == 4


def test_real_zero():
    assert m.real(0, 0, 1) == 0
