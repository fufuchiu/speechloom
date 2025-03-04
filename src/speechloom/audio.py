"""Streaming-friendly waveform utilities with explicit sample boundaries."""

import base64
import binascii
from dataclasses import dataclass

import numpy as np

from .validation import integer, real, vector


@dataclass(frozen=True)
class AudioFormat:
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2

    def __post_init__(self):
        integer(self.sample_rate, 1)
        if self.channels != 1 or isinstance(self.channels, bool):
            raise ValueError('only mono audio is supported')
        if self.sample_width != 2 or isinstance(self.sample_width, bool):
            raise ValueError('only PCM16 is supported')


def pcm_encode(samples) -> bytes:
    """Encode clipped mono samples as little-endian PCM16."""
    return np.clip(np.rint(vector(samples) * 32768), -32768, 32767).astype('<i2').tobytes()


def pcm_decode(payload: bytes) -> np.ndarray:
    """Reject truncated samples and decode PCM16."""
    if len(payload) % 2:
        raise ValueError('truncated PCM16 sample')
    return np.frombuffer(payload, dtype='<i2').astype(float) / 32768


def audio_to_base64(samples) -> str:
    """Serialize PCM16 as standard ASCII base64."""
    return base64.b64encode(pcm_encode(samples)).decode('ascii')


def audio_from_base64(payload: str, max_bytes: int = 3200000) -> np.ndarray:
    """Decode strict base64 under a bounded payload limit."""
    limit = integer(max_bytes, 1)
    if not isinstance(payload, str) or len(payload) > 4 * ((limit + 2) // 3):
        raise ValueError('invalid or oversized base64 audio')
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('invalid base64 audio') from exc
    if len(raw) > limit:
        raise ValueError('decoded audio exceeds limit')
    return pcm_decode(raw)


def chunk_audio(samples, chunk_size: int = 320) -> list[np.ndarray]:
    """Split without dropping or padding the final chunk."""
    size = integer(chunk_size, 1)
    x = vector(samples)
    return [x[start : start + size].copy() for start in range(0, len(x), size)]


def pad_audio(waveforms, multiple: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Batch nonempty waveforms and return exact unpadded lengths."""
    multiple = integer(multiple, 1)
    items = [vector(x) for x in waveforms]
    if not items or any(not len(x) for x in items):
        raise ValueError('audio batch must contain nonempty waveforms')
    lengths = np.array([len(x) for x in items], dtype=np.int64)
    width = ((int(lengths.max()) + multiple - 1) // multiple) * multiple
    padded = np.zeros((len(items), width))
    for i, x in enumerate(items):
        padded[i, : len(x)] = x
    return padded, lengths
