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


def bootstrap_mean(
    values, confidence: float = 0.95, repetitions: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap for independent observations, with a local RNG."""
    x = vector(values)
    confidence = real(confidence, 0, 1)
    repetitions = integer(repetitions, 1)
    if not len(x) or confidence in (0, 1):
        raise ValueError('nonempty values and confidence strictly between zero and one required')
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(repetitions)])
    tail = (1 - confidence) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def token_accuracy(predicted, expected, ignore_index: int = -100) -> float:
    """Token accuracy over the same shaped arrays, ignoring label padding."""
    a, c = np.asarray(predicted), np.asarray(expected)
    if (
        a.shape != c.shape
        or not np.issubdtype(a.dtype, np.integer)
        or not np.issubdtype(c.dtype, np.integer)
    ):
        raise ValueError('matching integer token arrays required')
    mask = c != ignore_index
    return float((a[mask] == c[mask]).mean()) if mask.any() else 0.0


def summarize_latency(timestamps, start: float) -> dict:
    """Summarize first output and inter-token gap statistics."""
    times = vector(timestamps)
    if not len(times):
        raise ValueError('at least one output timestamp required')
    gaps = inter_token_gaps(times)
    return {
        'first_token_seconds': first_token_latency(start, times[0]),
        'tokens': len(times),
        'mean_gap_seconds': float(gaps.mean()) if len(gaps) else 0.0,
        'p95_gap_seconds': percentile(gaps, 0.95) if len(gaps) else 0.0,
    }
