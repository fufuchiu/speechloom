import numpy as np
import pytest

from speechloom import audio as m


def test_format_rate():
    assert m.AudioFormat().sample_rate == 16000


def test_pcm_empty():
    assert m.pcm_decode(b'').tolist() == []


def test_pcm_bounds():
    assert m.pcm_decode(bytes.fromhex('00800000ff7f')).tolist() == [-1, 0, 32767 / 32768]


def test_pcm_clipping():
    assert m.pcm_encode([-3, 0, 3]).hex() == '00800000ff7f'


def test_base64_zero():
    assert m.audio_to_base64([0]) == 'AAA='
