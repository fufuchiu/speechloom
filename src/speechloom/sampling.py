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
