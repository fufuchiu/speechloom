import numpy as np
import pytest

from speechloom import codec as m


def test_mu_endpoints():
    assert m.mulaw_encode([-1, 0, 1]).tolist() == [0, 128, 255]
