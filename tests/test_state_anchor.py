"""Tests for the StateAnchorUpdate recurrence transition.

At init we expect:
    h_next = h + beta_t * core_out + 0          (Anchor(e) is zero-init)
    pre_core(h, 0) = norm(h)                    (signal_t is zero-init)

We don't construct Core here -- we drive ``update`` and ``pre_core``
directly with synthetic tensors.
"""

import torch

from radiant import StateAnchorUpdate, tiny_config


def test_anchor_proj_zero_init():
    sa = StateAnchorUpdate(tiny_config())
    assert torch.all(sa.anchor_proj.weight == 0.0)


def test_precompute_anchor_returns_zero_at_init():
    cfg = tiny_config()
    sa = StateAnchorUpdate(cfg)
    e = torch.randn(2, 4, cfg.d_model)
    anchor_e = sa.precompute_anchor(e)
    assert torch.all(anchor_e == 0.0)


def test_state_anchor_can_be_disabled():
    cfg = tiny_config(use_state_anchor=False)
    sa = StateAnchorUpdate(cfg)
    e = torch.randn(2, 4, cfg.d_model)
    anchor_e = sa.precompute_anchor(e)
    assert torch.all(anchor_e == 0.0)
    assert torch.allclose(sa.gamma(0), torch.tensor(0.0))


def test_beta_gamma_at_init_match_state_gate_init_scale():
    cfg = tiny_config(state_gate_init_scale=0.1)
    sa = StateAnchorUpdate(cfg)
    for t in range(cfg.max_loops):
        assert torch.allclose(sa.beta(t), torch.tensor(0.1), atol=1e-5)
        assert torch.allclose(sa.gamma(t), torch.tensor(0.1), atol=1e-5)


def test_t_beyond_max_loops_clamps():
    cfg = tiny_config()
    sa = StateAnchorUpdate(cfg)
    # Should not raise; uses last gate.
    sa.beta(cfg.max_loops + 5)
    sa.gamma(cfg.max_loops + 100)


def test_update_residual_form_at_init():
    """h_next - h == beta_0 * core_out at init (gamma * Anchor(e) == 0)."""
    cfg = tiny_config()
    sa = StateAnchorUpdate(cfg).eval()
    h = torch.randn(2, 4, cfg.d_model)
    core_out = torch.randn(2, 4, cfg.d_model)
    e = torch.randn_like(h)
    anchor_e = sa.precompute_anchor(e)
    with torch.no_grad():
        h_next = sa.update(h, core_out, anchor_e, t=0)
    expected = h + sa.beta(0) * core_out
    assert torch.allclose(h_next, expected, atol=1e-6)


def test_gradients_flow_to_gates():
    cfg = tiny_config()
    sa = StateAnchorUpdate(cfg)
    h = torch.randn(2, 4, cfg.d_model, requires_grad=True)
    core_out = torch.randn(2, 4, cfg.d_model)
    e = torch.randn_like(h.detach())
    anchor_e = sa.precompute_anchor(e)
    h_next = sa.update(h, core_out, anchor_e, t=0)
    h_next.sum().backward()
    assert sa.beta_logits.grad is not None
    assert sa.gamma_logits.grad is not None
    assert torch.isfinite(sa.beta_logits.grad).all()
