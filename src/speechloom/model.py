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


def sinusoidal_positions(length: int, width: int, device=None, dtype=None) -> torch.Tensor:
    """Construct paired sine/cosine absolute position embeddings."""
    length, width = integer(length, 1), integer(width, 2)
    if width % 2:
        raise ValueError('position width must be even')
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    scales = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000) / width)
    )
    result = torch.empty(length, width, device=device)
    result[:, 0::2] = torch.sin(positions * scales)
    result[:, 1::2] = torch.cos(positions * scales)
    return result.to(dtype=dtype) if dtype is not None else result
