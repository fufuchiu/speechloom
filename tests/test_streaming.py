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


def test_drain_idempotent():
    q = m.EventQueue()
    q.push('text', 'a')
    assert len(q.drain()) == 1
    assert q.drain() == []


def test_utf8_arbitrary_boundaries():
    s = m.UTF8Stream()
    raw = '你好🙂'.encode()
    parts = [s.feed(bytes([v])) for v in raw]
    parts.append(s.finish())
    assert ''.join(parts) == '你好🙂'


def test_utf8_incomplete():
    s = m.UTF8Stream()
    s.feed(bytes([0xE4]))
    with pytest.raises(UnicodeDecodeError):
        s.finish()


def test_utf8_closed():
    s = m.UTF8Stream()
    s.finish()
    with pytest.raises(ValueError):
        s.feed(b'a')


def test_utf8_double_finish():
    s = m.UTF8Stream()
    s.finish()
    with pytest.raises(ValueError):
        s.finish()


def test_ring_fifo():
    r = m.AudioRing(4)
    r.append([1, 2, 3])
    assert r.take(2).tolist() == [1, 2]
    r.append([4, 5])
    assert r.take(3).tolist() == [3, 4, 5]


def test_ring_overflow_atomic():
    r = m.AudioRing(2)
    r.append([1, 2])
    with pytest.raises(BufferError):
        r.append([3])
    assert r.take(2).tolist() == [1, 2]


def test_ring_drop_oldest():
    r = m.AudioRing(3, True)
    r.append([1, 2])
    r.append([3, 4, 5])
    assert r.take(3).tolist() == [3, 4, 5]
    assert r.dropped == 2


def test_ring_large_chunk():
    r = m.AudioRing(2, True)
    r.append([1, 2, 3, 4])
    assert r.take(2).tolist() == [3, 4]
    assert r.dropped == 2


def test_ring_insufficient():
    r = m.AudioRing(2)
    with pytest.raises(ValueError):
        r.take(1)


def test_ring_zero_take():
    r = m.AudioRing(2)
    r.append([1])
    assert r.take(0).tolist() == []
    assert r.size == 1


def test_pcm_boundary_reconstruction():
    from speechloom.audio import pcm_encode

    s = m.PCMStream(2)
    raw = pcm_encode([0, 0.5, -1])
    frames = []
    for byte in raw:
        frames.extend(s.feed(bytes([byte])))
    frames.extend(s.finish())
    assert np.concatenate(frames).tolist() == [0, 0.5, -1]


def test_pcm_truncated_final():
    s = m.PCMStream()
    s.feed(b'x')
    with pytest.raises(ValueError):
        s.finish()
    assert not s.closed


def test_pcm_closed():
    s = m.PCMStream()
    s.finish()
    with pytest.raises(ValueError):
        s.feed(b'')


def test_pcm_double_finish():
    s = m.PCMStream()
    s.finish()
    with pytest.raises(ValueError):
        s.finish()


def test_event_gap():
    with pytest.raises(ValueError):
        m.validate_events([m.StreamEvent(1, 'text', 'a')])
