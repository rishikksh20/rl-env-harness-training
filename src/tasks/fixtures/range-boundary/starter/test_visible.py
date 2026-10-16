from bounds import in_range


def test_inside():
    assert in_range(2, 0, 4) is True


def test_below():
    assert in_range(-1, 0, 4) is False


def test_above():
    assert in_range(5, 0, 4) is False
