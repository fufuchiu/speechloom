import pytest

from speechloom import conversation as m


def test_unknown_role():
    with pytest.raises(ValueError):
        m.Turn('tool', 'x')


def test_empty_turn():
    with pytest.raises(ValueError):
        m.Turn('user')


def test_system_audio():
    with pytest.raises(ValueError):
        m.Turn('system', audio='x.wav')


def test_empty_audio():
    with pytest.raises(ValueError):
        m.Turn('user', audio='')


def test_nonstring_text():
    with pytest.raises(ValueError):
        m.Turn('user', text=1)
