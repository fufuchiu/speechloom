"""Regression tests for codebook dimension consistency in residual_encode."""

import numpy as np
import pytest

from speechloom import codec as m


def test_mismatched_codebook_dimensions():
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
    book1 = np.array([[1.0, 2.0], [0.0, 0.0]])
    book2 = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])  # 3D vectors
    with pytest.raises(ValueError, match='dimensions differ'):
        m.residual_encode(vectors, [book1, book2])


def test_invalid_codebook_shape():
    vectors = np.array([[1.0, 2.0]])
    book = np.array([1.0, 2.0])  # 1D, not 2D
    with pytest.raises(ValueError, match='invalid codebook'):
        m.residual_encode(vectors, [book])


def test_matching_dimensions_ok():
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
    book1 = np.array([[1.0, 2.0], [0.0, 0.0]])
    book2 = np.array([[0.5, 0.5], [0.0, 0.0]])
    result = m.residual_encode(vectors, [book1, book2])
    assert result.shape == (2, 2)
