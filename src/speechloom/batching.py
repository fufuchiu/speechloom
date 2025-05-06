"""Teacher-forcing batches, attention masks and budgeted sequence packing."""

import numpy as np

from .validation import integer, token_ids


def padding_mask(lengths, max_length: int | None = None) -> np.ndarray:
    """Boolean mask with True at padded positions (PyTorch convention)."""
    values = [integer(n, 1) for n in lengths]
    if not values:
        raise ValueError('lengths cannot be empty')
    width = max(values) if max_length is None else integer(max_length, 1)
    if max(values) > width:
        raise ValueError('length exceeds padded width')
    return np.arange(width)[None, :] >= np.array(values)[:, None]


def causal_mask(length: int) -> np.ndarray:
    """True strictly above the diagonal, preventing access to future tokens."""
    length = integer(length, 1)
    return np.triu(np.ones((length, length), dtype=bool), k=1)


def shift_targets(
    sequences, vocabulary_size: int, pad_id: int = 0, ignore_index: int = -100
) -> dict:
    """Produce decoder inputs and one-step-shifted labels without supervising pad."""
    pad_id = token_ids([pad_id], vocabulary_size)[0]
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int) or ignore_index >= 0:
        raise ValueError('ignore_index must be a negative integer')
    values = [token_ids(sequence, vocabulary_size) for sequence in sequences]
    if not values or any(len(sequence) < 2 or pad_id in sequence for sequence in values):
        raise ValueError('each unpadded sequence needs at least two non-pad tokens')
    lengths = np.array([len(sequence) - 1 for sequence in values], dtype=np.int64)
    width = int(lengths.max())
    inputs = np.full((len(values), width), pad_id, dtype=np.int64)
    labels = np.full_like(inputs, ignore_index)
    for i, sequence in enumerate(values):
        inputs[i, : lengths[i]] = sequence[:-1]
        labels[i, : lengths[i]] = sequence[1:]
    return {
        'input_ids': inputs,
        'labels': labels,
        'lengths': lengths,
        'padding_mask': padding_mask(lengths, width),
    }


def loss_weights(
    labels, audio_offset: int = 260, text_weight: float = 1, audio_weight: float = 1
) -> np.ndarray:
    """Construct per-token modality weights with zero at ignored labels."""
    from .validation import real

    offset = integer(audio_offset, 4)
    text_weight, audio_weight = real(text_weight, 0), real(audio_weight, 0)
    x = np.asarray(labels)
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError('labels must be integers')
    return np.where(x < 0, 0, np.where(x >= offset, audio_weight, text_weight)).astype(float)
