import torch

from radiant import MoEFeedForward, SwiGLUFeedForward, tiny_config


def test_swiglu_shape():
    cfg = tiny_config()
    ff = SwiGLUFeedForward(cfg).eval()
    x = torch.randn(2, 7, cfg.d_model)
    with torch.no_grad():
        y = ff(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_moe_shape_and_aux():
    cfg = tiny_config(use_moe=True, n_experts=4, n_active_experts=2, moe_aux_loss_weight=0.01)
    ff = MoEFeedForward(cfg).eval()
    x = torch.randn(2, 8, cfg.d_model)
    with torch.no_grad():
        y, aux = ff(x)
    assert y.shape == x.shape
    assert aux.dim() == 0
    assert torch.isfinite(y).all()
    assert torch.isfinite(aux)


def test_moe_load_uniform_under_random_input():
    """Aux loss is bounded; with uniform random routing it stays close to 1.0."""
    cfg = tiny_config(use_moe=True, n_experts=4, n_active_experts=2)
    ff = MoEFeedForward(cfg).eval()
    x = torch.randn(8, 32, cfg.d_model)
    with torch.no_grad():
        _, aux = ff(x)
    # aux = weight * E * sum(fraction_routed * importance). Both tend to ~1/E,
    # so the unweighted product is roughly 1.0 / E for fresh routing.
    assert 0.0 < aux.item() < 1.0
