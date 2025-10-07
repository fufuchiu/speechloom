import numpy as np
import pytest

from speechloom import audio as m


def test_format_rate():
    assert m.AudioFormat().sample_rate == 16000


def test_pcm_empty():
    assert m.pcm_decode(b'').tolist() == []
