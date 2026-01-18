import pytest

from speechloom import conversation as m


def test_unknown_role():
    with pytest.raises(ValueError):
        m.Turn('tool', 'x')


def test_empty_turn():
    with pytest.raises(ValueError):
        m.Turn('user')
