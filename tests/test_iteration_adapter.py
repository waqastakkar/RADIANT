import pytest
import torch

from radiant import IterationAdapter, tiny_config


def test_adapter_identity_at_init():
    """scale == 0 and up.weight == 0 -> adapter returns h unchanged."""
    cfg = tiny_config()
    adapter = IterationAdapter(cfg)
    h = torch.randn(2, 4, cfg.d_model)
    with torch.no_grad():
        h_out = adapter(h, t=0)
    assert torch.allclose(h_out, h, atol=1e-6)


def test_adapter_t_clamped():
    cfg = tiny_config()
    adapter = IterationAdapter(cfg)
    h = torch.randn(2, 4, cfg.d_model)
    # Should not raise even past max_loops.
    with torch.no_grad():
        adapter(h, t=cfg.max_loops + 10)


def test_negative_t_rejected():
    cfg = tiny_config()
    adapter = IterationAdapter(cfg)
    h = torch.randn(2, 4, cfg.d_model)
    with pytest.raises(ValueError):
        adapter(h, t=-1)


def test_adapter_per_loop_scale_makes_outputs_differ():
    cfg = tiny_config()
    adapter = IterationAdapter(cfg)
    # Force non-trivial weights AND distinct per-loop scales.
    torch.nn.init.normal_(adapter.up.weight, std=0.1)
    with torch.no_grad():
        adapter.scale[0].fill_(0.5)
        adapter.scale[1].fill_(-0.5)
    h = torch.randn(2, 4, cfg.d_model)
    with torch.no_grad():
        a = adapter(h, t=0)
        b = adapter(h, t=1)
    assert not torch.allclose(a, b, atol=1e-4)


def test_adapter_grad_flow():
    cfg = tiny_config()
    adapter = IterationAdapter(cfg)
    h = torch.randn(2, 4, cfg.d_model, requires_grad=True)
    out = adapter(h, t=0)
    out.pow(2).sum().backward()
    assert h.grad is not None
    assert adapter.scale.grad is not None
    assert torch.isfinite(adapter.scale.grad).all()
