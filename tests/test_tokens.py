import pytest

from speechloom import tokens as m


def test_utf8_roundtrip_0():
    assert m.decode_text(m.encode_text('', True)) == ''


def test_utf8_roundtrip_1():
    assert m.decode_text(m.encode_text('hello', True)) == 'hello'


def test_utf8_roundtrip_2():
    assert m.decode_text(m.encode_text('广州大学', True)) == '广州大学'


def test_utf8_roundtrip_3():
    assert m.decode_text(m.encode_text('语音大模型', True)) == '语音大模型'


def test_utf8_roundtrip_4():
    assert m.decode_text(m.encode_text('ＡＢＣ', True)) == 'ＡＢＣ'


def test_utf8_roundtrip_5():
    assert m.decode_text(m.encode_text('café', True)) == 'café'


def test_utf8_roundtrip_6():
    assert m.decode_text(m.encode_text('café', True)) == 'café'


def test_utf8_roundtrip_7():
    assert m.decode_text(m.encode_text('🙂🙃', True)) == '🙂🙃'


def test_utf8_roundtrip_8():
    assert m.decode_text(m.encode_text('مرحبا', True)) == 'مرحبا'


def test_utf8_roundtrip_9():
    assert m.decode_text(m.encode_text('音声認識', True)) == '音声認識'


def test_utf8_roundtrip_10():
    assert m.decode_text(m.encode_text('음성 인식', True)) == '음성 인식'


def test_utf8_roundtrip_11():
    assert m.decode_text(m.encode_text('नमस्ते', True)) == 'नमस्ते'


def test_utf8_roundtrip_12():
    assert m.decode_text(m.encode_text('Привет', True)) == 'Привет'


def test_utf8_roundtrip_13():
    assert m.decode_text(m.encode_text('Αλφα', True)) == 'Αλφα'


def test_utf8_roundtrip_14():
    assert m.decode_text(m.encode_text('a\nb\tc', True)) == 'a\nb\tc'


def test_utf8_roundtrip_15():
    assert m.decode_text(m.encode_text('\x00', True)) == '\x00'


def test_utf8_roundtrip_16():
    assert m.decode_text(m.encode_text('a b', True)) == 'a b'


def test_utf8_roundtrip_17():
    assert m.decode_text(m.encode_text('12345', True)) == '12345'


def test_utf8_roundtrip_18():
    assert m.decode_text(m.encode_text('x+y=2', True)) == 'x+y=2'


def test_utf8_roundtrip_19():
    assert m.decode_text(m.encode_text('ﬁle', True)) == 'ﬁle'


def test_utf8_roundtrip_20():
    assert m.decode_text(m.encode_text('Straße', True)) == 'Straße'


def test_utf8_roundtrip_21():
    assert m.decode_text(m.encode_text('é🙂中文', True)) == 'é🙂中文'


def test_utf8_roundtrip_22():
    assert m.decode_text(m.encode_text('你好，世界！', True)) == '你好，世界！'


def test_utf8_roundtrip_23():
    assert m.decode_text(m.encode_text('𠀀', True)) == '𠀀'


def test_utf8_roundtrip_24():
    assert m.decode_text(m.encode_text('\u200d', True)) == '\u200d'


def test_utf8_roundtrip_25():
    assert m.decode_text(m.encode_text('line\r\nend', True)) == 'line\r\nend'


def test_utf8_roundtrip_26():
    assert m.decode_text(m.encode_text('👩\u200d💻', True)) == '👩\u200d💻'


def test_utf8_roundtrip_27():
    assert m.decode_text(m.encode_text('a\xa0b', True)) == 'a\xa0b'


def test_utf8_roundtrip_28():
    assert m.decode_text(m.encode_text('引用“声音”', True)) == '引用“声音”'


def test_utf8_roundtrip_29():
    assert m.decode_text(m.encode_text('\\path/file', True)) == '\\path/file'


def test_layout_vocab():
    assert m.TokenLayout().vocabulary_size == 516


def test_layout_books():
    assert m.TokenLayout(16, 3).vocabulary_size == 308


def test_audio_map():
    assert m.encode_audio([0, 255]) == [260, 515]


def test_audio_inverse():
    assert m.decode_audio([260, 515]) == [0, 255]


def test_second_codebook():
    assert m.encode_audio([0, 15], m.TokenLayout(16, 2), 1) == [276, 291]


def test_special_modality():
    assert m.modality(3) == 'special'


def test_text_modality():
    assert m.modality(259) == 'text'


def test_audio_modality():
    assert m.modality(260) == 'audio'


def test_empty_response():
    assert m.unpack_response(m.pack_response('', [])) == ('', [])


def test_invalid_bins():
    with pytest.raises(ValueError):
        m.TokenLayout(1)


def test_invalid_books():
    with pytest.raises(ValueError):
        m.TokenLayout(codebooks=0)


def test_reserved_change():
    with pytest.raises(ValueError):
        m.TokenLayout(pad=3)


def test_nonstring():
    with pytest.raises(ValueError):
        m.encode_text(1)


def test_audio_in_text():
    with pytest.raises(ValueError):
        m.decode_text([260])


def test_invalid_utf8():
    with pytest.raises(ValueError):
        m.decode_text([259])


def test_invalid_error_policy():
    with pytest.raises(ValueError):
        m.decode_text([], errors='surprise')


def test_wrong_audio_region():
    with pytest.raises(ValueError):
        m.decode_audio([259])


def test_cross_book():
    with pytest.raises(ValueError):
        m.decode_audio([276], m.TokenLayout(16, 2), 0)


def test_book_too_large():
    with pytest.raises(ValueError):
        m.encode_audio([], m.TokenLayout(), 1)


def test_audio_id_too_large():
    with pytest.raises(ValueError):
        m.encode_audio([256])


def test_bad_grammar():
    with pytest.raises(ValueError):
        m.unpack_response([1, 2])


def test_double_separator():
    with pytest.raises(ValueError):
        m.unpack_response([1, 3, 3, 2])


def test_reserved_in_text():
    with pytest.raises(ValueError):
        m.unpack_response([1, 1, 3, 2])


def test_response_roundtrip():
    text, codes = m.unpack_response(m.pack_response('你好', [0, 128, 255]))
    assert text == '你好'
    assert codes == [0, 128, 255]
