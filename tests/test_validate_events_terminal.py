"""Regression tests for validate_events terminal event requirement."""

import pytest

from speechloom import streaming as m


def test_empty_events_rejected():
    with pytest.raises(ValueError, match='cannot be empty'):
        m.validate_events([])


def test_missing_terminal_rejected():
    with pytest.raises(ValueError, match='terminal event'):
        m.validate_events([m.StreamEvent(0, 'text', 'hello')])


def test_valid_done_terminal():
    events = [m.StreamEvent(0, 'text', 'hello'), m.StreamEvent(1, 'done', '')]
    assert len(m.validate_events(events)) == 2


def test_valid_cancelled_terminal():
    events = [m.StreamEvent(0, 'audio', 'ff'), m.StreamEvent(1, 'cancelled', '')]
    assert len(m.validate_events(events)) == 2


def test_multiple_non_terminal_rejected():
    with pytest.raises(ValueError, match='terminal event'):
        m.validate_events(
            [
                m.StreamEvent(0, 'text', 'a'),
                m.StreamEvent(1, 'text', 'b'),
            ]
        )
