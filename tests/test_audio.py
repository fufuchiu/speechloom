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


def test_crossfade_one():
    assert m.crossfade([0, 0], [1, 1], 1).tolist() == [0, 0.5, 1]


def test_sample_seconds():
    assert m.samples_to_seconds(160, 16000) == 0.01


def test_round_half_up():
    assert m.seconds_to_samples(0.5, 1) == 1


def test_round_below_half():
    assert m.seconds_to_samples(0.49, 1) == 0


def test_zero_rate():
    with pytest.raises(ValueError):
        m.AudioFormat(0)


def test_stereo():
    with pytest.raises(ValueError):
        m.AudioFormat(channels=2)


def test_wrong_width():
    with pytest.raises(ValueError):
        m.AudioFormat(sample_width=1)


def test_pcm_truncated():
    with pytest.raises(ValueError):
        m.pcm_decode(b'x')


def test_base64_invalid():
    with pytest.raises(ValueError):
        m.audio_from_base64('!!!!')


def test_base64_whitespace():
    with pytest.raises(ValueError):
        m.audio_from_base64('AA A=')


def test_base64_excess():
    with pytest.raises(ValueError):
        m.audio_from_base64('AAAAAA==', 2)


def test_base64_odd():
    with pytest.raises(ValueError):
        m.audio_from_base64('AA==')


def test_zero_chunk():
    with pytest.raises(ValueError):
        m.chunk_audio([], 0)


def test_empty_batch():
    with pytest.raises(ValueError):
        m.pad_audio([])


def test_empty_member():
    with pytest.raises(ValueError):
        m.pad_audio([[1], []])


def test_zero_multiple():
    with pytest.raises(ValueError):
        m.pad_audio([[1]], 0)


def test_excess_overlap():
    with pytest.raises(ValueError):
        m.crossfade([1], [2], 2)


def test_negative_overlap():
    with pytest.raises(ValueError):
        m.crossfade([1], [2], -1)


def test_negative_time():
    with pytest.raises(ValueError):
        m.seconds_to_samples(-1)


def test_invalid_clock_rate():
    with pytest.raises(ValueError):
        m.seconds_to_samples(1, 0)


def test_padded_batch_lengths():
    x, n = m.pad_audio([[1, 2, 3], [4]], 4)
    assert x.tolist() == [[1, 2, 3, 0], [4, 0, 0, 0]]
    assert n.tolist() == [3, 1]


def test_chunk_reconstruction():
    x = np.linspace(-1, 1, 37)
    assert np.concatenate(m.chunk_audio(x, 8)) == pytest.approx(x)


def test_base64_roundtrip():
    x = np.array([-1, -0.5, 0, 0.5])
    assert m.audio_from_base64(m.audio_to_base64(x)) == pytest.approx(x)


def test_crossfade_reversal():
    a = np.array([1, 2, 3.0])
    c = np.array([4, 5, 6.0])
    assert m.crossfade(a, c, 2) == pytest.approx(m.crossfade(c[::-1], a[::-1], 2)[::-1])


def test_audio_format_rejects_float_channels_and_sample_width():
    for values in [{'channels': 1.0}, {'sample_width': 2.0}]:
        with pytest.raises(ValueError):
            m.AudioFormat(**values)


def test_pad_single_waveform():
    '''A single waveform is returned unpadded with its exact length.'''
    x, n = m.pad_audio([[1, 2, 3]])
    assert x.tolist() == [[1, 2, 3]]
    assert n.tolist() == [3]


def test_pad_identical_lengths():
    '''Waveforms of equal length need no padding.'''
    x, n = m.pad_audio([[1, 2], [3, 4]])
    assert x.tolist() == [[1, 2], [3, 4]]
    assert n.tolist() == [2, 2]


def test_pad_round_up_multiple():
    '''Width rounds up to the next multiple.'''
    x, n = m.pad_audio([[1, 2, 3]], multiple=4)
    assert x.shape == (1, 4)
    assert x[0].tolist() == [1, 2, 3, 0]


def test_chunk_exact():
    '''chunk_size equal to sample count yields one chunk.'''
    chunks = m.chunk_audio([1, 2, 3], 3)
    assert len(chunks) == 1
    assert chunks[0].tolist() == [1, 2, 3]


def test_chunk_larger_than_input():
    '''chunk_size larger than sample count yields one short chunk.'''
    chunks = m.chunk_audio([1, 2], 10)
    assert len(chunks) == 1
    assert chunks[0].tolist() == [1, 2]


def test_pcm_boundary_values():
    '''Boundary samples -1 and +1 encode to min and max int16.'''
    payload = m.pcm_encode([-1, 1])
    assert m.pcm_decode(payload).tolist() == [-1, 32767 / 32768]


def test_base64_exact_limit():
    '''A payload exactly at the limit decodes successfully.'''
    raw = b'\x00\x00'  # 2 bytes
    encoded = __import__('base64').b64encode(raw).decode('ascii')
    result = m.audio_from_base64(encoded, max_bytes=2)
    assert result.tolist() == [0.0]
