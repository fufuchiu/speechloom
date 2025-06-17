"""Differentiable waveform-to-token model with a causal multimodal decoder."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .validation import integer, real


@dataclass(frozen=True)
class SpeechConfig:
    audio_bins: int = 256
    d_model: int = 32
    heads: int = 4
    encoder_layers: int = 1
    decoder_layers: int = 1
    max_tokens: int = 1024
    max_audio_samples: int = 32000

    def __post_init__(self):
        integer(self.audio_bins, 2)
        for key in (
            'd_model',
            'heads',
            'encoder_layers',
            'decoder_layers',
            'max_tokens',
            'max_audio_samples',
        ):
            integer(getattr(self, key), 1, key)
        if self.d_model % self.heads or self.d_model % 2:
            raise ValueError('d_model must be even and divisible by heads')

    @property
    def vocabulary_size(self) -> int:
        return 260 + self.audio_bins
