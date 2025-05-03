"""Numerically stable token sampling and explicit generation budgets."""

import numpy as np

from .validation import integer, real, token_ids


def checked_logits(logits) -> np.ndarray:
    """Allow masked (-inf) outcomes but require at least one finite candidate."""
    x = np.asarray(logits, dtype=float)
    if (
        x.ndim != 1
        or not len(x)
        or np.isnan(x).any()
        or np.isposinf(x).any()
        or not np.isfinite(x).any()
    ):
        raise ValueError('invalid logits or no available outcomes')
    return x.copy()


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    """Stable softmax with finite positive temperature."""
    temperature = real(temperature, 0)
    if temperature == 0:
        raise ValueError('temperature must be positive')
    x = checked_logits(logits)
    x = (x - x.max()) / temperature
    weights = np.exp(x)
    return weights / weights.sum()


def top_k(logits, k: int) -> np.ndarray:
    """Keep k largest logits, resolving equal logits by lower token ID."""
    x = checked_logits(logits)
    k = integer(k, 1)
    if k > len(x):
        raise ValueError('k exceeds vocabulary size')
    order = np.argsort(-x, kind='stable')
    x[order[k:]] = -np.inf
    return x


def top_p(logits, p: float = 0.9) -> np.ndarray:
    """Retain the smallest sorted prefix whose cumulative mass reaches p."""
    x = checked_logits(logits)
    p = real(p, 0, 1)
    if p == 0:
        raise ValueError('p must be greater than zero')
    order = np.argsort(-x, kind='stable')
    cumulative = np.cumsum(softmax(x)[order])
    count = min(len(x), int(np.searchsorted(cumulative, p, side='left')) + 1)
    x[order[count:]] = -np.inf
    return x


def repetition_penalty(logits, previous, penalty: float = 1.0) -> np.ndarray:
    """Reduce positive repeated logits and increase the magnitude of negative ones."""
    x = checked_logits(logits)
    penalty = real(penalty, 1)
    for token in set(token_ids(previous, len(x))):
        x[token] = x[token] / penalty if x[token] > 0 else x[token] * penalty
    return x


def sample_token(logits, temperature: float = 1, seed: int | None = None) -> int:
    """Sample with a local RNG, or choose the lowest-ID argmax at zero temperature."""
    temperature = real(temperature, 0)
    x = checked_logits(logits)
    if temperature == 0:
        return int(x.argmax())
    return int(np.random.default_rng(seed).choice(len(x), p=softmax(x, temperature)))


def allowed_tokens(logits, allowed) -> np.ndarray:
    """Mask every token outside an explicit, nonempty allowlist."""
    x = checked_logits(logits)
    ids = token_ids(allowed, len(x))
    result = np.full_like(x, -np.inf)
    result[ids] = x[ids]
    return checked_logits(result)
