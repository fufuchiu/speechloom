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


def first_token_latency(start: float, first_token: float) -> float:
    """Latency from request start to first output using one monotonic clock."""
    start, end = real(start), real(first_token)
    if end < start:
        raise ValueError('first token precedes request start')
    return end - start


def inter_token_gaps(timestamps) -> np.ndarray:
    """Check monotonic timestamps and return consecutive delays."""
    x = vector(timestamps)
    gaps = np.diff(x)
    if (gaps < 0).any():
        raise ValueError('timestamps must be nondecreasing')
    return gaps


def waveform_snr(reference, generated) -> float:
    """SNR for equal-length nonempty waveforms; identical signals return +inf."""
    ref, hyp = vector(reference), vector(generated)
    if not len(ref) or ref.shape != hyp.shape:
        raise ValueError('waveforms must be nonempty and aligned')
    noise = float(np.sum((ref - hyp) ** 2))
    signal = float(np.sum(ref**2))
    if noise == 0:
        return float('inf')
    if signal == 0:
        return -float('inf')
    return float(10 * np.log10(signal / noise))
