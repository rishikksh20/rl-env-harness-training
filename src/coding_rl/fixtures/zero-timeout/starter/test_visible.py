from timeout import normalize_timeout


def test_positive():
    assert normalize_timeout(5) == 5


def test_omitted():
    assert normalize_timeout() == 30


def test_custom_default():
    assert normalize_timeout(default=7) == 7
