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


def fit_codebook(vectors, size: int = 16, iterations: int = 20, seed: int = 0) -> np.ndarray:
    """Fit deterministic Lloyd k-means, preserving empty centroids."""
    x = np.asarray(vectors, dtype=float)
    size, iterations = integer(size, 1), integer(iterations, 1)
    if x.ndim != 2 or len(x) < size or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('need at least size finite training vectors')
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size, replace=False)].copy()
    for _ in range(iterations):
        ids = nearest_codes(x, centers)
        next_centers = centers.copy()
        for i in range(size):
            group = x[ids == i]
            if len(group):
                next_centers[i] = group.mean(0)
        if np.allclose(centers, next_centers, rtol=0, atol=1e-12):
            break
        centers = next_centers
    return centers


def residual_encode(vectors, codebooks) -> np.ndarray:
    """Quantize each residual in turn; shape (frames, codebooks)."""
    residual = np.asarray(vectors, dtype=float).copy()
    books = list(codebooks)
    if not books:
        raise ValueError('at least one codebook is required')
    width = None
    for book in books:
        book = np.asarray(book, dtype=float)
        if book.ndim != 2 or not all(book.shape):
            raise ValueError('invalid codebook')
        if width is not None and book.shape[1] != width:
            raise ValueError('codebook vector dimensions differ')
        width = book.shape[1]
    columns = []
    for book in books:
        book = np.asarray(book, dtype=float)
        ids = nearest_codes(residual, book)
        columns.append(ids)
        residual -= book[ids]
    return np.stack(columns, axis=1)


def residual_decode(codes, codebooks) -> np.ndarray:
    """Sum residual codebook vectors after checking every ID and dimension."""
    books = [np.asarray(book, dtype=float) for book in codebooks]
    ids = np.asarray(codes)
    if not books or ids.ndim != 2 or ids.shape[1] != len(books):
        raise ValueError('codebook count and code columns differ')
    width = None
    result = None
    for column, book in enumerate(books):
        if book.ndim != 2 or not all(book.shape) or not np.isfinite(book).all():
            raise ValueError('invalid codebook')
        if width is not None and book.shape[1] != width:
            raise ValueError('codebook vector dimensions differ')
        width = book.shape[1]
        checked = token_ids(ids[:, column], len(book))
        if result is None:
            result = np.zeros((len(ids), width))
        result += book[checked]
    return result


def codebook_perplexity(codes, size: int) -> float:
    """Exponentiated assignment entropy; empty observations have perplexity zero."""
    ids = token_ids(codes, integer(size, 1))
    if not ids:
        return 0.0
    counts = np.bincount(ids, minlength=size)
    p = counts[counts > 0] / len(ids)
    return float(np.exp(-np.sum(p * np.log(p))))
