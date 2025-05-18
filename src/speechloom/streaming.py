"""Bounded, cancellable speech stream state machines."""

import codecs
from collections import deque
from dataclasses import dataclass

import numpy as np

from .audio import pcm_decode, pcm_encode
from .validation import integer, vector


@dataclass(frozen=True)
class StreamEvent:
    sequence: int
    kind: str
    payload: str

    def __post_init__(self):
        integer(self.sequence)
        if self.kind not in ('text', 'audio', 'done', 'cancelled'):
            raise ValueError('unknown event kind')
        if not isinstance(self.payload, str):
            raise ValueError('payload must be a string')
        if self.kind in ('done', 'cancelled') and self.payload:
            raise ValueError('terminal events cannot contain a payload')
