import torch

from radiant import IterationSignal, tiny_config


def test_signal_zero_at_init():
    """tanh(gate=0) * adapter(zero_weight)(...) == 0 vector."""
    cfg = tiny_config(iteration_signal_kind="both")
    sig = IterationSignal(cfg)
    out = sig(0, device=torch.device("cpu"), dtype=torch.float32)
    assert out.shape == (cfg.d_model,)
    assert torch.all(out == 0.0)


def test_kind_none_returns_zeros():
    cfg = tiny_config(iteration_signal_kind="none")
    sig = IterationSignal(cfg)
    for t in (0, 3, 100):
        out = sig(t, device=torch.device("cpu"), dtype=torch.float32)
        assert torch.all(out == 0.0)


def test_sinusoidal_supports_t_beyond_max_loops():
    cfg = tiny_config(iteration_signal_kind="sinusoidal")
    sig = IterationSignal(cfg)
    out = sig(cfg.max_loops + 10, device=torch.device("cpu"), dtype=torch.float32)
    # At init, output is gate*adapter(...) = 0.
    assert torch.all(out == 0.0)


def test_after_training_signal_distinct_per_loop():
    """If the gate and adapter are trained, different loops produce different signals."""
    cfg = tiny_config(iteration_signal_kind="both")
    sig = IterationSignal(cfg)
    # Force non-zero parameters: random init for adapter, set gate to 1.
    torch.nn.init.normal_(sig.adapter.weight, std=0.1)
    with torch.no_grad():
        sig.gate.fill_(1.0)
    a = sig(0, device=torch.device("cpu"), dtype=torch.float32)
    b = sig(1, device=torch.device("cpu"), dtype=torch.float32)
    assert not torch.allclose(a, b, atol=1e-4)


def test_negative_t_rejected():
    cfg = tiny_config()
    sig = IterationSignal(cfg)
    import pytest
    with pytest.raises(ValueError):
        sig(-1, device=torch.device("cpu"), dtype=torch.float32)
