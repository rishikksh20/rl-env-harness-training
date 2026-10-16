from options import choose_option


def test_value():
    assert choose_option("chosen") == "chosen"


def test_omitted():
    assert choose_option() == "fallback"


def test_default():
    assert choose_option(default="other") == "other"
