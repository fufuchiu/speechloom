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
