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
