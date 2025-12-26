import numpy as np
import pytest

from speechloom import codec as m


def test_mu_endpoints():
    assert m.mulaw_encode([-1, 0, 1]).tolist() == [0, 128, 255]


def test_mu_decode_ends():
    assert m.mulaw_decode([0, 255]).tolist() == pytest.approx([-1, 1])


def test_mu_clipping():
    assert m.mulaw_encode([-2, 2]).tolist() == [0, 255]


def test_mu_empty():
    assert m.mulaw_encode([]).tolist() == []


def test_decode_empty():
    assert m.mulaw_decode([]).tolist() == []


def test_nearest_simple():
    assert m.nearest_codes([[0], [9]], [[0], [10]]).tolist() == [0, 1]


def test_nearest_tie():
    assert m.nearest_codes([[1]], [[0], [2]]).tolist() == [0]


def test_nearest_empty():
    assert m.nearest_codes(np.empty((0, 2)), [[0, 0]]).tolist() == []


def test_one_cluster():
    assert m.fit_codebook([[0], [2], [4]], 1).tolist() == [[2]]


def test_perplexity_one():
    assert m.codebook_perplexity([0, 0], 4) == 1


def test_perplexity_uniform():
    assert m.codebook_perplexity([0, 1, 2, 3], 4) == pytest.approx(4)


def test_perplexity_empty():
    assert m.codebook_perplexity([], 4) == 0


def test_one_bin():
    with pytest.raises(ValueError):
        m.mulaw_encode([0], 1)


def test_negative_code():
    with pytest.raises(ValueError):
        m.mulaw_decode([-1])


def test_large_code():
    with pytest.raises(ValueError):
        m.mulaw_decode([256])


def test_float_code():
    with pytest.raises(ValueError):
        m.mulaw_decode([1.0])


def test_shape_mismatch():
    with pytest.raises(ValueError):
        m.nearest_codes([[1, 2]], [[1]])


def test_empty_book():
    with pytest.raises(ValueError):
        m.nearest_codes([[1]], [])


def test_nan_vector():
    with pytest.raises(ValueError):
        m.nearest_codes([[float('nan')]], [[1]])


def test_too_few_vectors():
    with pytest.raises(ValueError):
        m.fit_codebook([[1]], 2)


def test_zero_iterations():
    with pytest.raises(ValueError):
        m.fit_codebook([[1]], 1, 0)


def test_zero_books():
    with pytest.raises(ValueError):
        m.residual_encode([[1]], [])


def test_wrong_columns():
    with pytest.raises(ValueError):
        m.residual_decode([[0, 0]], [[[1]]])


def test_decode_out_of_range():
    with pytest.raises(ValueError):
        m.residual_decode([[2]], [[[1], [2]]])


def test_decode_dimension_mismatch():
    with pytest.raises(ValueError):
        m.residual_decode([[0, 0]], [[[1]], [[1, 2]]])


def test_invalid_perplexity():
    with pytest.raises(ValueError):
        m.codebook_perplexity([4], 4)


def test_mulaw_monotonic():
    ids = m.mulaw_encode(np.linspace(-1, 1, 1000))
    assert (np.diff(ids) >= 0).all()
    x = m.mulaw_decode(range(256))
    assert (np.diff(x) > 0).all()


def test_mulaw_odd_symmetry():
    x = np.linspace(0.01, 1, 100)
    assert m.mulaw_encode(x) + m.mulaw_encode(-x) == pytest.approx(np.full(100, 255))


def test_residual_roundtrip():
    books = [np.array([[0, 0], [2, 2]]), np.array([[0, 0], [1, -1]])]
    x = np.array([[0, 0], [3, 1]])
    ids = m.residual_encode(x, books)
    assert ids.tolist() == [[0, 0], [1, 1]]
    assert m.residual_decode(ids, books) == pytest.approx(x)


def test_kmeans_reproducible():
    x = np.random.default_rng(4).normal(size=(40, 3))
    assert m.fit_codebook(x, 4, 10, 7) == pytest.approx(m.fit_codebook(x, 4, 10, 7))


def test_kmeans_empty_cluster_finite():
    centers = m.fit_codebook([[1], [1], [1], [1]], 3)
    assert np.isfinite(centers).all()
    assert centers.tolist() == [[1], [1], [1]]
