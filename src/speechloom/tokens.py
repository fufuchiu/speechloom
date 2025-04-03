"""Lossless byte text and disjoint audio-token vocabulary layout."""

from dataclasses import dataclass

from .validation import integer, token_ids


@dataclass(frozen=True)
class TokenLayout:
    audio_bins: int = 256
    codebooks: int = 1
    pad: int = 0
    bos: int = 1
    eos: int = 2
    separator: int = 3

    def __post_init__(self):
        integer(self.audio_bins, 2)
        integer(self.codebooks, 1)
        if (self.pad, self.bos, self.eos, self.separator) != (0, 1, 2, 3) or any(
            isinstance(v, bool) for v in (self.pad, self.bos, self.eos, self.separator)
        ):
            raise ValueError('reserved IDs must be pad=0, bos=1, eos=2, separator=3')

    @property
    def audio_offset(self) -> int:
        return 260

    @property
    def vocabulary_size(self) -> int:
        return self.audio_offset + self.audio_bins * self.codebooks


def encode_text(text: str, boundaries: bool = False) -> list[int]:
    """Encode UTF-8 bytes at offset four; optionally surround by BOS/EOS."""
    if not isinstance(text, str):
        raise ValueError('text must be a string')
    ids = [byte + 4 for byte in text.encode('utf-8')]
    return [1, *ids, 2] if boundaries else ids
