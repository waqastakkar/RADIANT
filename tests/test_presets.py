from radiant import base_config, small_config, tiny_config


def test_tiny_overrides():
    a = tiny_config()
    b = tiny_config(d_model=128)
    assert a.d_model == 64
    assert b.d_model == 128


def test_size_ordering():
    a = tiny_config().d_model
    b = small_config().d_model
    c = base_config().d_model
    assert a < b < c
