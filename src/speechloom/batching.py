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
