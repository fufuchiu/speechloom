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


def test_bad_example_fields():
    with pytest.raises(ValueError):
        m.validate_example({'audio': 'a.wav', 'target_ids': [1, 2], 'extra': 1})


def test_missing_bos():
    with pytest.raises(ValueError):
        m.validate_example({'audio': 'a.wav', 'target_ids': [4, 2]})


def test_missing_eos():
    with pytest.raises(ValueError):
        m.validate_example({'audio': 'a.wav', 'target_ids': [1, 4]})


def test_target_padding():
    with pytest.raises(ValueError):
        m.validate_example({'audio': 'a.wav', 'target_ids': [1, 0, 2]})


def test_zero_turn_budget():
    with pytest.raises(ValueError):
        m.truncate_turns([m.Turn('user', 'u')], 0)


def test_audio_only_user():
    assert m.Turn('user', audio='a.wav').text == ''
