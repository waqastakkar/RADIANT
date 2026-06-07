import pytest
import torch

from radiant import RadiantModel, tiny_config


@pytest.mark.parametrize("n_loops", [1, 2, 3, 4])
def test_variable_n_loops_same_model(n_loops):
    cfg = tiny_config()
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 5))
    with torch.no_grad():
        out = model(x, n_loops=n_loops)
    assert out.n_loops_executed == n_loops
    assert out.logits.shape == (2, 5, cfg.vocab_size)


def test_n_loops_zero_rejected():
    cfg = tiny_config()
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with pytest.raises(ValueError):
        model(x, n_loops=0)


def test_loops_beyond_max_with_learned_signal_rejected():
    cfg = tiny_config(iteration_signal_kind="learned")
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with pytest.raises(ValueError):
        model(x, n_loops=cfg.max_loops + 1)


def test_loops_beyond_max_with_sinusoidal_only_allowed():
    cfg = tiny_config(iteration_signal_kind="sinusoidal")
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out = model(x, n_loops=cfg.max_loops + 2)
    assert out.n_loops_executed == cfg.max_loops + 2


def test_loops_beyond_max_with_kind_none_allowed():
    cfg = tiny_config(iteration_signal_kind="none")
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out = model(x, n_loops=cfg.max_loops + 2)
    assert out.n_loops_executed == cfg.max_loops + 2


def test_default_n_loops_uses_train_setting():
    cfg = tiny_config(n_loops_train=2)
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out = model(x)  # no n_loops kwarg
    assert out.n_loops_executed == 2
