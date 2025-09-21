import numpy as np
import pytest

from speechloom import audio as m


def test_format_rate():
    assert m.AudioFormat().sample_rate == 16000
