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


class EventQueue:
    """Single-producer bounded queue; overflow never silently drops events."""

    def __init__(self, capacity: int = 32):
        self.capacity = integer(capacity, 1)
        self._events = deque()
        self._next = 0
        self.state = 'open'

    def push(self, kind: str, payload: str = '') -> StreamEvent:
        if self.state != 'open':
            raise ValueError('stream is no longer open')
        event = StreamEvent(self._next, kind, payload)
        if len(self._events) >= self.capacity:
            raise BufferError('consumer must drain the stream before more events arrive')
        self._events.append(event)
        self._next += 1
        if kind in ('done', 'cancelled'):
            self.state = kind
        return event

    def pop(self) -> StreamEvent | None:
        return self._events.popleft() if self._events else None

    def drain(self) -> list[StreamEvent]:
        result = list(self._events)
        self._events.clear()
        return result

    def cancel(self) -> StreamEvent:
        if self.state != 'open':
            raise ValueError('stream is no longer open')
        self._events.clear()
        return self.push('cancelled')

    @property
    def pending(self) -> int:
        return len(self._events)


class UTF8Stream:
    """Decode tokens incrementally without emitting partial multibyte text."""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder('utf-8')('strict')
        self.closed = False

    def feed(self, payload: bytes) -> str:
        if self.closed:
            raise ValueError('text stream is closed')
        return self._decoder.decode(payload, final=False)

    def finish(self) -> str:
        if self.closed:
            raise ValueError('text stream is closed')
        text = self._decoder.decode(b'', final=True)
        self.closed = True
        return text
