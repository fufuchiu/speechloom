import numpy as np
import pytest

from speechloom import streaming as m


def test_negative_sequence():
    with pytest.raises(ValueError):
        m.StreamEvent(-1, 'text', 'x')


def test_unknown_kind():
    with pytest.raises(ValueError):
        m.StreamEvent(0, 'token', 'x')


def test_terminal_payload():
    with pytest.raises(ValueError):
        m.StreamEvent(0, 'done', 'x')


def test_zero_queue():
    with pytest.raises(ValueError):
        m.EventQueue(0)


def test_zero_ring():
    with pytest.raises(ValueError):
        m.AudioRing(0)


def test_zero_pcm():
    with pytest.raises(ValueError):
        m.PCMStream(0)
