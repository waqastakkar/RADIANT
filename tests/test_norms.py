import torch

from radiant import RMSNorm


def test_rmsnorm_unit_when_weight_one():
    m = RMSNorm(8)
    x = torch.randn(4, 6, 8) * 5.0
    y = m(x)
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_dtype_preserved():
    m = RMSNorm(8)
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(2, 3, 8, dtype=dtype)
        assert m(x).dtype == dtype


def test_rmsnorm_grad_flows():
    m = RMSNorm(8)
    x = torch.randn(2, 3, 8, requires_grad=True)
    m(x).pow(2).sum().backward()
    assert x.grad is not None
    assert m.weight.grad is not None
    assert torch.isfinite(m.weight.grad).all()
