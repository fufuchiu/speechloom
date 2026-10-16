"""Regression tests for __len__ dunder on AudioRing and EventQueue."""

from speechloom import streaming as m


def test_queue_len_empty():
    q = m.EventQueue(4)
    assert len(q) == 0


def test_queue_len_after_push():
    q = m.EventQueue(4)
    q.push('text', 'a')
    q.push('text', 'b')
    assert len(q) == 2


def test_queue_len_after_pop():
    q = m.EventQueue(4)
    q.push('text', 'a')
    q.pop()
    assert len(q) == 0


def test_queue_len_matches_pending():
    q = m.EventQueue(4)
    q.push('text', 'x')
    assert len(q) == q.pending


def test_ring_len_empty():
    r = m.AudioRing(8)
    assert len(r) == 0


def test_ring_len_after_append():
    r = m.AudioRing(8)
    r.append([1, 2, 3])
    assert len(r) == 3


def test_ring_len_after_take():
    r = m.AudioRing(8)
    r.append([1, 2, 3])
    r.take(2)
    assert len(r) == 1


def test_ring_len_matches_size():
    r = m.AudioRing(8)
    r.append([1, 2, 3, 4])
    assert len(r) == r.size
