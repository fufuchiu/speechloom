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


class SpeechModel(nn.Module):
    """Conv waveform encoder and Transformer text/audio-token decoder."""

    def __init__(self, config: SpeechConfig = SpeechConfig()):
        super().__init__()
        self.config = config
        self.frontend = nn.Conv1d(1, config.d_model, kernel_size=31, stride=16, padding=15)
        encoder = nn.TransformerEncoderLayer(
            config.d_model, config.heads, config.d_model * 4, dropout=0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder, config.encoder_layers, enable_nested_tensor=False
        )
        decoder = nn.TransformerDecoderLayer(
            config.d_model, config.heads, config.d_model * 4, dropout=0, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder, config.decoder_layers)
        self.embedding = nn.Embedding(config.vocabulary_size, config.d_model, padding_idx=0)
        self.output = nn.Linear(config.d_model, config.vocabulary_size)

    def encode(
        self, audio: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            audio.ndim != 2
            or not all(audio.shape)
            or audio.shape[1] > self.config.max_audio_samples
            or not torch.isfinite(audio).all()
        ):
            raise ValueError('invalid audio tensor or audio context budget exceeded')
        if (
            lengths.dtype not in (torch.int32, torch.int64)
            or lengths.shape != (audio.shape[0],)
            or (lengths <= 0).any()
            or (lengths > audio.shape[1]).any()
        ):
            raise ValueError('invalid audio lengths')
        valid = (
            torch.arange(audio.shape[1], device=audio.device)[None, :]
            < lengths.to(audio.device)[:, None]
        )
        hidden = self.frontend(audio.masked_fill(~valid, 0)[:, None, :]).transpose(1, 2)
        encoded_lengths = (lengths.to(audio.device) + 15) // 16
        mask = (
            torch.arange(hidden.shape[1], device=audio.device)[None, :] >= encoded_lengths[:, None]
        )
        hidden = hidden + sinusoidal_positions(
            hidden.shape[1], self.config.d_model, hidden.device, hidden.dtype
        )
        return self.encoder(hidden, src_key_padding_mask=mask), mask

    def decode(
        self, input_ids: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor
    ) -> torch.Tensor:
        if (
            input_ids.ndim != 2
            or not all(input_ids.shape)
            or input_ids.shape[1] > self.config.max_tokens
            or input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError('invalid decoder tokens or context budget exceeded')
        if (input_ids < 0).any() or (input_ids >= self.config.vocabulary_size).any():
            raise ValueError('decoder ID outside vocabulary')
        if (
            memory.ndim != 3
            or memory.shape[0] != input_ids.shape[0]
            or memory.shape[2] != self.config.d_model
            or not torch.isfinite(memory).all()
        ):
            raise ValueError('invalid encoder memory')
        if (
            memory_mask.shape != memory.shape[:2]
            or memory_mask.dtype != torch.bool
            or memory_mask.all(1).any()
        ):
            raise ValueError('invalid memory padding mask')
        padding = input_ids == 0
        lengths = (~padding).sum(1)
        expected = (
            torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] >= lengths[:, None]
        )
        if (lengths == 0).any() or not torch.equal(padding, expected):
            raise ValueError('padding must follow a nonempty token prefix')
        hidden = self.embedding(input_ids) * math.sqrt(self.config.d_model)
        hidden = hidden + sinusoidal_positions(
            input_ids.shape[1], self.config.d_model, hidden.device, hidden.dtype
        )
        causal = torch.triu(
            torch.ones(
                input_ids.shape[1], input_ids.shape[1], device=input_ids.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        decoded = self.decoder(
            hidden,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=padding,
            memory_key_padding_mask=memory_mask,
        )
        return self.output(decoded)

    def forward(
        self, audio: torch.Tensor, lengths: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        memory, mask = self.encode(audio, lengths)
        return self.decode(input_ids, memory, mask)
