from speechloom import __version__


def test_version():
    assert __version__.split('.') == ['0', '1', '0']
