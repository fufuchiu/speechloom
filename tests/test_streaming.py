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


def test_bad_hex():
    with pytest.raises(ValueError):
        m.decode_audio_event('zz')


def test_odd_pcm_hex():
    with pytest.raises(ValueError):
        m.decode_audio_event('ff')


def test_queue_fifo_and_sequence():
    q = m.EventQueue(3)
    q.push('text', 'a')
    q.push('text', 'b')
    assert q.pop().sequence == 0
    assert q.pop().payload == 'b'
    assert q.pop() is None


def test_queue_overflow_atomic():
    q = m.EventQueue(1)
    q.push('text', 'a')
    with pytest.raises(BufferError):
        q.push('text', 'b')
    assert q.pop().payload == 'a'
    assert q.push('text', 'c').sequence == 1


def test_queue_terminal():
    q = m.EventQueue()
    q.push('done')
    with pytest.raises(ValueError):
        q.push('text', 'late')
    assert q.state == 'done'


def test_cancel_discards_pending():
    q = m.EventQueue(1)
    q.push('audio', '0000')
    event = q.cancel()
    assert event.kind == 'cancelled'
    assert q.pending == 1
    assert q.drain() == [event]


def test_cancel_closed():
    q = m.EventQueue()
    q.push('done')
    with pytest.raises(ValueError):
        q.cancel()
