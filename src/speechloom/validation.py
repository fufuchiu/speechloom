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
