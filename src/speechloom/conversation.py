"""Typed local conversation records with explicit modality boundaries."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .tokens import TokenLayout, pack_response
from .validation import integer, token_ids


@dataclass(frozen=True)
class Turn:
    role: str
    text: str = ''
    audio: str | None = None

    def __post_init__(self):
        if self.role not in ('system', 'user', 'assistant'):
            raise ValueError('unknown conversation role')
        if not isinstance(self.text, str):
            raise ValueError('text must be a string')
        if self.audio is not None and (not isinstance(self.audio, str) or not self.audio.strip()):
            raise ValueError('audio must be a nonempty local path')
        if self.role == 'system' and self.audio is not None:
            raise ValueError('system turns cannot carry audio')
        if not self.text and self.audio is None:
            raise ValueError('turn must contain text or audio')


def validate_turns(turns) -> list[Turn]:
    """Check optional leading system turn followed by alternating user/assistant."""
    values = list(turns)
    if not values:
        raise ValueError('conversation cannot be empty')
    expected = 'user'
    for index, turn in enumerate(values):
        if not isinstance(turn, Turn):
            raise ValueError('conversation entries must be Turn instances')
        if index == 0 and turn.role == 'system':
            continue
        if turn.role != expected:
            raise ValueError(f'expected {expected} turn')
        expected = 'assistant' if expected == 'user' else 'user'
    if values[-1].role == 'system':
        raise ValueError('system-only conversation is incomplete')
    return values


def load_conversations(path: str | Path) -> list[list[Turn]]:
    """Read JSONL with strict per-turn fields and line-numbered diagnostics."""
    result = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or set(row) != {'turns'}
                or not isinstance(row['turns'], list)
            ):
                raise ValueError('record must contain only a turns array')
            result.append(validate_turns([Turn(**value) for value in row['turns']]))
        except (ValueError, TypeError) as exc:
            raise ValueError(f'line {number}: {exc}') from exc
    return result


def save_conversations(path: str | Path, conversations) -> None:
    """Validate all conversations before writing Unicode JSONL."""
    rows = [{'turns': [asdict(turn) for turn in validate_turns(turns)]} for turns in conversations]
    Path(path).write_text(
        ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows),
        encoding='utf-8',
    )


def response_example(
    audio_path: str, text: str, codes, layout: TokenLayout = TokenLayout()
) -> dict:
    """Build a supervised audio-input / multimodal-output example."""
    Turn('user', audio=audio_path)
    return {'audio': audio_path, 'target_ids': pack_response(text, codes, layout)}


def truncate_turns(turns, max_turns: int) -> list[Turn]:
    """Drop complete oldest user/assistant pairs while preserving the system turn."""
    max_turns = integer(max_turns, 1)
    values = validate_turns(turns)
    prefix = values[:1] if values[0].role == 'system' else []
    body = values[len(prefix) :]
    while len(prefix) + len(body) > max_turns and len(body) > 2:
        body = body[2:]
    if len(prefix) + len(body) > max_turns:
        raise ValueError('turn budget cannot preserve a complete final exchange')
    return validate_turns(prefix + body)
