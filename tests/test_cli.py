import json
import tempfile
from pathlib import Path

import pytest

from speechloom.cli import main


def test_tokenize_basic(capsys):
    """tokenize without audio-codes encodes text with BOS/EOS boundaries."""
    assert main(['tokenize', 'hello']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['tokens'] == 7  # BOS + 5 bytes + EOS
    assert output['ids'][0] == 1
    assert output['ids'][-1] == 2


def test_tokenize_with_audio_codes(capsys):
    """tokenize with --audio-codes packs text, separator, and audio tokens."""
    assert main(['tokenize', 'hi', '--audio-codes', '0,128']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['ids'] == [1, 108, 109, 3, 260, 388, 2]


def test_tokenize_unicode(capsys):
    """tokenize handles multi-byte UTF-8 characters."""
    assert main(['tokenize', '\u4f60\u597d']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['tokens'] == 8  # BOS + 6 bytes + EOS


def test_audit_valid(capsys):
    """audit prints conversation and turn counts from a valid JSONL."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'turns.jsonl'
        p.write_text(
            json.dumps(
                {
                    'turns': [
                        {'role': 'user', 'text': 'hello'},
                        {'role': 'assistant', 'text': 'world'},
                    ]
                }
            )
            + chr(10)
        )
        assert main(['audit', str(p)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {'conversations': 1, 'turns': 2}


def test_audit_missing_file():
    """audit with a nonexistent manifest exits with code 2."""
    with pytest.raises(SystemExit) as exc:
        main(['audit', '/nonexistent/file.jsonl'])
    assert exc.value.code == 2


def test_tokenize_empty_text(capsys):
    """tokenize with empty text produces just BOS and EOS."""
    assert main(['tokenize', '']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {'ids': [1, 2], 'tokens': 2}


def test_missing_command():
    """Missing subcommand exits with code 2."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
