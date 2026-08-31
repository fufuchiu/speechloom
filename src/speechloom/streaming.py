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
        if self._events:
            self._next = self._events[0].sequence
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


class AudioRing:
    """A bounded waveform buffer; callers explicitly choose drop-oldest behavior."""

    def __init__(self, capacity: int = 16000, drop_oldest: bool = False):
        self.capacity = integer(capacity, 1)
        self.drop_oldest = drop_oldest
        self._samples = np.empty(0)
        self.dropped = 0

    def append(self, samples) -> None:
        x = vector(samples)
        excess = max(0, len(self._samples) + len(x) - self.capacity)
        if excess and not self.drop_oldest:
            raise BufferError('audio buffer capacity exceeded')
        combined = np.concatenate((self._samples, x))
        self._samples = combined[-self.capacity :]
        self.dropped += excess

    def take(self, count: int) -> np.ndarray:
        count = integer(count)
        if count > len(self._samples):
            raise ValueError('not enough buffered samples')
        result = self._samples[:count].copy()
        self._samples = self._samples[count:]
        return result

    @property
    def size(self) -> int:
        return len(self._samples)


class PCMStream:
    """Accept arbitrary byte chunks, exposing complete fixed-size audio frames."""

    def __init__(self, frame_samples: int = 320):
        self.frame_bytes = integer(frame_samples, 1) * 2
        self._pending = bytearray()
        self.closed = False

    def feed(self, payload: bytes) -> list[np.ndarray]:
        if self.closed:
            raise ValueError('PCM stream is closed')
        self._pending.extend(payload)
        frames = []
        while len(self._pending) >= self.frame_bytes:
            frames.append(pcm_decode(bytes(self._pending[: self.frame_bytes])))
            del self._pending[: self.frame_bytes]
        return frames

    def finish(self) -> list[np.ndarray]:
        if self.closed:
            raise ValueError('PCM stream is closed')
        if len(self._pending) % 2:
            raise ValueError('truncated PCM sample')
        result = [pcm_decode(bytes(self._pending))] if self._pending else []
        self._pending.clear()
        self.closed = True
        return result


def validate_events(events) -> list[StreamEvent]:
    """Require contiguous sequence IDs and a single final terminal event."""
    values = list(events)
    for index, event in enumerate(values):
        if not isinstance(event, StreamEvent):
            raise ValueError('events must be StreamEvent instances')
        if event.sequence != index:
            raise ValueError('event sequence is not contiguous from zero')
        if event.kind in ('done', 'cancelled') and index != len(values) - 1:
            raise ValueError('events follow a terminal event')
    return values


def encode_audio_event(samples) -> str:
    """Represent small PCM chunks as hex for line-oriented debug traces."""
    return pcm_encode(samples).hex()


def decode_audio_event(payload: str) -> np.ndarray:
    """Decode a hex PCM event, rejecting malformed or truncated payloads."""
    try:
        raw = bytes.fromhex(payload)
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid hexadecimal audio event') from exc
    return pcm_decode(raw)
