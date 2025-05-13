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
