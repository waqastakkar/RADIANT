import pytest
import torch

from radiant import RadiantModel, tiny_config


@pytest.mark.parametrize("n_loops", [1, 4, 8])
def test_grads_reach_stem_for_any_n_loops(n_loops):
    """A backward through n_loops produces finite, non-zero grads on stem.embed."""
    cfg = tiny_config(max_loops=8, iteration_signal_kind="sinusoidal")
    model = RadiantModel(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    out = model(x, n_loops=n_loops)
    target = torch.zeros_like(out.logits)
    loss = (out.logits - target).pow(2).mean()
    loss.backward()
    g = model.stem.token_embed.weight.grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert g.abs().sum() > 0


def test_grads_reach_recurrent_blocks():
    cfg = tiny_config()
    model = RadiantModel(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    out = model(x, n_loops=cfg.max_loops)
    out.logits.sum().backward()
    for p in model.refinement.core_blocks.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_grads_reach_state_anchor_gates():
    cfg = tiny_config()
    model = RadiantModel(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model(x, n_loops=cfg.max_loops)
    out.logits.sum().backward()
    sa = model.refinement.state_anchor
    assert sa.beta_logits.grad is not None
    assert sa.gamma_logits.grad is not None
    assert torch.isfinite(sa.beta_logits.grad).all()
    assert torch.isfinite(sa.gamma_logits.grad).all()


def test_no_nan_on_long_unroll():
    cfg = tiny_config()
    model = RadiantModel(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model(x, n_loops=cfg.max_loops)
    assert torch.isfinite(out.logits).all()
    out.logits.sum().backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
