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


def mulaw_decode(codes, bins: int = 256) -> np.ndarray:
    """Decode scalar mu-law IDs to approximate normalized amplitudes."""
    bins = integer(bins, 2)
    ids = np.array(token_ids(codes, bins))
    mu = bins - 1
    value = 2 * ids / mu - 1
    return np.sign(value) * np.expm1(np.abs(value) * np.log1p(mu)) / mu


def nearest_codes(vectors, codebook) -> np.ndarray:
    """Assign vectors to nearest Euclidean centroids, ties choosing lowest ID."""
    x, c = np.asarray(vectors, dtype=float), np.asarray(codebook, dtype=float)
    if (
        x.ndim != 2
        or c.ndim != 2
        or not len(c)
        or not c.shape[1]
        or x.shape[1] != c.shape[1]
        or not np.isfinite(x).all()
        or not np.isfinite(c).all()
    ):
        raise ValueError('finite compatible vector and codebook matrices required')
    # Batch distances avoid allocating an N*K*D tensor.
    distances = np.maximum((x * x).sum(1, keepdims=True) + (c * c).sum(1) - 2 * x @ c.T, 0)
    return distances.argmin(1)
