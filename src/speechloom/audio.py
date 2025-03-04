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
