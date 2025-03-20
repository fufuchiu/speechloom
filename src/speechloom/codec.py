"""Transparent scalar mu-law codec and trainable residual vector quantization."""

import numpy as np

from .validation import integer, token_ids, vector


def mulaw_encode(samples, bins: int = 256) -> np.ndarray:
    """Quantize clipped audio using mu=bins-1 companding."""
    bins = integer(bins, 2)
    x = np.clip(vector(samples), -1, 1)
    mu = bins - 1
    companded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((companded + 1) * 0.5 * mu + 0.5).astype(np.int64)
