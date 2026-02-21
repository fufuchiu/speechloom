import numpy as np
import pytest

from speechloom import streaming as m


def test_negative_sequence():
    with pytest.raises(ValueError):
        m.StreamEvent(-1, 'text', 'x')
