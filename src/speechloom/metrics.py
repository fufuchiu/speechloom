"""Deterministic speech interaction metrics and reproducible confidence intervals."""

import numpy as np

from .validation import integer, real, vector


def percentile(values, q: float = 0.95) -> float:
    """Linear-interpolated percentile of nonempty finite observations."""
    x = vector(values)
    if not len(x):
        raise ValueError('percentile requires observations')
    return float(np.quantile(x, real(q, 0, 1)))


def real_time_factor(elapsed: float, audio_seconds: float) -> float:
    """Wall time divided by positive audio duration."""
    elapsed, audio = real(elapsed, 0), real(audio_seconds, 0)
    if audio == 0:
        raise ValueError('audio duration must be positive')
    return elapsed / audio
