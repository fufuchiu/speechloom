"""Strict contracts for research data and model configuration."""

import math
import numbers

import numpy as np


def integer(value, minimum: int = 0, name: str = 'value') -> int:
    """Accept integral scalars, rejecting booleans and values below a bound."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)
