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


def test_no_turns():
    with pytest.raises(ValueError):
        m.validate_turns([])


def test_system_only():
    with pytest.raises(ValueError):
        m.validate_turns([m.Turn('system', 's')])


def test_assistant_first():
    with pytest.raises(ValueError):
        m.validate_turns([m.Turn('assistant', 'a')])


def test_double_user():
    with pytest.raises(ValueError):
        m.validate_turns([m.Turn('user', 'u'), m.Turn('user', 'v')])


def test_system_midway():
    with pytest.raises(ValueError):
        m.validate_turns([m.Turn('user', 'u'), m.Turn('system', 's')])


def test_untyped_turn():
    with pytest.raises(ValueError):
        m.validate_turns([{'role': 'user', 'text': 'x'}])
