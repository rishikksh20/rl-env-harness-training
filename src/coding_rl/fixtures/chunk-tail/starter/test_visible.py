from chunks import chunk_items


def test_pairs():
    assert chunk_items([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_singletons():
    assert chunk_items([1, 2], 1) == [[1], [2]]


def test_empty():
    assert chunk_items([], 3) == []
