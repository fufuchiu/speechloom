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


def test_base64_empty():
    assert m.audio_from_base64('').tolist() == []


def test_chunk_empty():
    assert m.chunk_audio([]) == []


def test_chunk_tail():
    assert [len(v) for v in m.chunk_audio([1] * 7, 3)] == [3, 3, 1]


def test_energy_empty():
    assert m.rms_energy([]) == 0


def test_energy_unit():
    assert m.rms_energy([-1, 1]) == 1


def test_crossfade_none():
    assert m.crossfade([1], [2], 0).tolist() == [1, 2]


def test_crossfade_equal():
    assert m.crossfade([1, 1], [1, 1], 2).tolist() == [1, 1]
