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


def test_valid_exchange():
    assert len(m.validate_turns([m.Turn('user', 'u'), m.Turn('assistant', 'a')])) == 2


def test_valid_system():
    assert len(m.validate_turns([m.Turn('system', 's'), m.Turn('user', 'u')])) == 2


def test_truncate_pairs():
    turns = [
        m.Turn('system', 's'),
        m.Turn('user', 'u1'),
        m.Turn('assistant', 'a1'),
        m.Turn('user', 'u2'),
        m.Turn('assistant', 'a2'),
    ]
    assert [x.text for x in m.truncate_turns(turns, 3)] == ['s', 'u2', 'a2']


def test_preserve_pending_user():
    turns = [m.Turn('user', 'u1'), m.Turn('assistant', 'a1'), m.Turn('user', 'u2')]
    assert [x.text for x in m.truncate_turns(turns, 1)] == ['u2']


def test_reject_incomplete_budget():
    with pytest.raises(ValueError):
        m.truncate_turns([m.Turn('user', 'u'), m.Turn('assistant', 'a')], 1)


def test_conversation_roundtrip():
    import tempfile
    from pathlib import Path

    rows = [[m.Turn('user', audio='u.wav'), m.Turn('assistant', '你好')]]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'turns.jsonl'
        m.save_conversations(p, rows)
        assert m.load_conversations(p) == rows


def test_load_invalid_json():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'turns.jsonl'
        p.write_text('{\n')
        with pytest.raises(ValueError, match='line 1'):
            m.load_conversations(p)


def test_load_wrong_envelope():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'turns.jsonl'
        p.write_text('{"other":[]}')
        with pytest.raises(ValueError, match='line 1'):
            m.load_conversations(p)


def test_load_unknown_field():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'turns.jsonl'
        p.write_text('{"turns":[{"role":"user","text":"x","extra":1}]}')
        with pytest.raises(ValueError, match='line 1'):
            m.load_conversations(p)


def test_load_bad_sequence():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'turns.jsonl'
        p.write_text('{"turns":[{"role":"assistant","text":"x"}]}')
        with pytest.raises(ValueError, match='line 1'):
            m.load_conversations(p)


def test_supervised_response():
    assert m.response_example('a.wav', 'a', [0])['target_ids'] == [1, 101, 3, 260, 2]


def test_save_invalid_turns_rejected():
    '''save_conversations raises before writing invalid data.'''
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'turns.jsonl'
        with pytest.raises(ValueError):
            m.save_conversations(p, [[m.Turn('user', 'u'), m.Turn('user', 'v')]])
        assert not p.exists()


def test_validate_example_small_vocab():
    '''validate_example with custom vocabulary_size rejects out-of-range IDs.'''
    with pytest.raises(ValueError):
        m.validate_example({
            'audio': 'a.wav',
            'target_ids': [1, 100, 2],
        }, vocabulary_size=50)


def test_turn_text_and_audio():
    '''A user turn can carry both text and audio.'''
    turn = m.Turn('user', 'listen', audio='a.wav')
    assert turn.text == 'listen'
    assert turn.audio == 'a.wav'


def test_load_empty_file():
    '''An empty JSONL file yields an empty conversation list.'''
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'turns.jsonl'
        p.write_text('')
        assert m.load_conversations(p) == []


def test_single_user_turn():
    '''A single user turn is a valid conversation prefix.'''
    turns = m.validate_turns([m.Turn('user', 'hello')])
    assert len(turns) == 1
