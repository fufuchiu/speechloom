"""Strict contracts for research data and model configuration."""

import math
import numbers

import numpy as np


def integer(value, minimum: int = 0, name: str = 'value') -> int:
    """Accept integral scalars, rejecting booleans and values below a bound."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def real(value, minimum: float | None = None, maximum: float | None = None) -> float:
    """Check a finite real number and optional inclusive bounds."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError('expected a finite real number')
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError('number is outside allowed bounds')
    return float(value)


def vector(values) -> np.ndarray:
    """Copy a finite one-dimensional numeric array."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('expected a finite vector')
    return x.copy()


def token_ids(values, vocabulary_size: int) -> list[int]:
    """Validate a sequence of nonnegative vocabulary IDs."""
    size = integer(vocabulary_size, 1)
    result = [integer(v, 0, 'token') for v in values]
    if any(v >= size for v in result):
        raise ValueError('token ID outside vocabulary')
    return result
