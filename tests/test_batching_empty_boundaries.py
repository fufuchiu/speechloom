"""Regression test for length_buckets with empty boundaries."""

import pytest

from speechloom import batching as m


def test_empty_boundaries_rejected():
    with pytest.raises(ValueError, match='cannot be empty'):
        m.length_buckets([10, 20, 30], boundaries=())


def test_empty_boundaries_list():
    with pytest.raises(ValueError, match='cannot be empty'):
        m.length_buckets([10], boundaries=[])
