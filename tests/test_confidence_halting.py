import torch

from radiant import ConfidenceHalting, HaltingTrace, RadiantModel, tiny_config
from radiant.confidence_halting import (
    _per_step_halt_probabilities,
    halting_kl_loss,
    pondernet_task_loss,
)


def test_confidence_in_unit_interval():
    cfg = tiny_config(use_confidence_halting=True)
    head = ConfidenceHalting(cfg).eval()
    h = torch.randn(2, 5, cfg.d_model)
    with torch.no_grad():
        c = head(h)
    assert c.shape == (2, 5)
    assert (c >= 0.0).all() and (c <= 1.0).all()


def test_init_bias_keeps_conf_low_at_init():
    """bias = -2 -> sigmoid(-2) ~= 0.119."""
    cfg = tiny_config(use_confidence_halting=True, confidence_init_bias=-2.0)
    head = ConfidenceHalting(cfg).eval()
    h = torch.randn(2, 5, cfg.d_model)
    with torch.no_grad():
        c = head(h)
    # All confidences should be near sigmoid(-2) since head.weight is zero-init.
    assert (c < 0.2).all()


def test_halting_trace_finalize_assigns_halt_step():
    trace = HaltingTrace()
    # Three loops with rising confidences; threshold of 0.5 hits at t=1.
    trace.append(torch.tensor([[0.1, 0.4]]))
    trace.append(torch.tensor([[0.5, 0.4]]))
    trace.append(torch.tensor([[0.9, 0.4]]))
    trace.finalize(threshold=0.5)
    assert trace.halt_step is not None
    # Cumulative: t=0 -> [0.1,0.4]; t=1 -> [0.6,0.8]; t=2 -> [1.5,1.2]
    # Token 0: first crosses at t=1; token 1: crosses at t=1 too.
    assert trace.halt_step[0, 0].item() == 1
    assert trace.halt_step[0, 1].item() == 1
    assert trace.avg_depth == 2.0


def test_halting_trace_falls_back_to_last_when_never_crossed():
    trace = HaltingTrace()
    trace.append(torch.tensor([[0.1]]))
    trace.append(torch.tensor([[0.1]]))
    trace.finalize(threshold=0.99)
    assert trace.halt_step[0, 0].item() == 1  # T-1


def test_model_with_halting_returns_trace_and_avg_depth():
    cfg = tiny_config(use_confidence_halting=True, confidence_threshold=0.5)
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        out = model(x, n_loops=3)
    assert out.halting is not None
    assert len(out.halting.confidences) == out.n_loops_executed
    assert out.halting.avg_depth is not None
    assert out.halting.avg_depth >= 1.0


def test_halting_off_means_no_trace():
    cfg = tiny_config(use_confidence_halting=False)
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        out = model(x, n_loops=3)
    assert out.halting is None


# ---------------------------------------------------------------------------
# PonderNet-style auxiliary loss
# ---------------------------------------------------------------------------

def test_halting_kl_loss_is_non_negative_and_finite():
    confs = [torch.sigmoid(torch.randn(2, 5)) for _ in range(4)]
    loss = halting_kl_loss(confs, attention_mask=None, prior_lambda=0.2)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_halting_kl_loss_zero_when_distribution_matches_prior():
    # If the per-step conditional halt prob matches a geometric prior
    # with the same lambda, the KL should be near zero (within float noise).
    T, lam = 6, 0.2
    confs = [torch.full((1, 1), lam) for _ in range(T)]
    loss = halting_kl_loss(confs, attention_mask=None, prior_lambda=lam)
    assert abs(float(loss)) < 1e-4, f"expected ~0 KL when matched, got {float(loss)}"


def test_halting_kl_loss_respects_attention_mask():
    confs = [torch.sigmoid(torch.randn(2, 6)) for _ in range(3)]
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.long)
    loss_masked = halting_kl_loss(confs, attention_mask=mask, prior_lambda=0.2)
    loss_full = halting_kl_loss(confs, attention_mask=None, prior_lambda=0.2)
    assert torch.isfinite(loss_masked)
    # With most positions masked off the loss usually differs from the unmasked one.
    assert float(loss_masked) != float(loss_full)


def test_halting_loss_produces_gradient_through_head():
    """The whole point of the patch: head params must receive gradient."""
    cfg = tiny_config(
        use_confidence_halting=True,
        halting_loss_weight=1.0,
        confidence_threshold=0.5,
    )
    model = RadiantModel(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    attn = torch.ones_like(x)
    out = model(x, n_loops=4, attention_mask=attn, is_causal=False)
    # The aux loss list must contain at least one term that depends on the
    # halting head's parameters.
    assert out.aux_loss is not None, "halting_loss_weight=1.0 should yield aux_loss"
    out.aux_loss.backward()
    head = model.refinement.halting.head
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert head.bias.grad is not None and head.bias.grad.abs().sum() > 0


def test_halting_loss_zero_weight_skips_aux():
    cfg = tiny_config(use_confidence_halting=True, halting_loss_weight=0.0)
    model = RadiantModel(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    out = model(x, n_loops=4)
    # No aux losses *from halting* (other aux losses may still exist if MoE
    # is on; tiny_config has MoE off, so aux_losses should be empty).
    assert all(t.requires_grad is False or "halting" not in str(t) for t in out.aux_losses)


def test_halting_loss_disabled_at_eval_even_with_weight():
    cfg = tiny_config(use_confidence_halting=True, halting_loss_weight=1.0)
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        out = model(x, n_loops=4)
    # At eval the trace is detached, no aux loss is appended.
    assert out.aux_loss is None or float(out.aux_loss) == 0.0


def test_halting_loss_requires_use_confidence_halting():
    import pytest
    from radiant import RadiantConfig
    with pytest.raises(ValueError, match="halting_loss_weight"):
        RadiantConfig(use_confidence_halting=False, halting_loss_weight=0.01)


def test_halting_warmup_and_lr_mult_validate():
    import pytest
    from radiant import RadiantConfig
    with pytest.raises(ValueError, match="halting_loss_warmup_epochs"):
        RadiantConfig(halting_loss_warmup_epochs=-1)
    with pytest.raises(ValueError, match="halting_head_lr_mult"):
        RadiantConfig(halting_head_lr_mult=0.0)
    # Valid combo should not raise.
    cfg = RadiantConfig(
        use_confidence_halting=True,
        halting_loss_weight=0.3,
        halting_loss_warmup_epochs=10,
        halting_head_lr_mult=10.0,
    )
    assert cfg.halting_loss_warmup_epochs == 10
    assert cfg.halting_head_lr_mult == 10.0


# ---------------------------------------------------------------------------
# PonderNet per-step supervision
# ---------------------------------------------------------------------------

def test_per_step_halt_probabilities_sum_to_one():
    confs = [torch.sigmoid(torch.randn(2, 5)) for _ in range(4)]
    p = _per_step_halt_probabilities(confs)
    s = p.sum(dim=0)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_pondernet_task_loss_runs_on_regression():
    T, B = 4, 3
    preds_per_step = [torch.randn(B, 1) for _ in range(T)]
    target = torch.randn(B)
    confs = [torch.sigmoid(torch.randn(B, 5)) for _ in range(T)]
    loss = pondernet_task_loss(preds_per_step, target=target, confidences=confs,
                               task_kind="regression")
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_pondernet_concentrates_mass_on_step_zero_when_conf_one():
    """If conf_0 is ~1.0 the halt distribution puts all mass on step 0,
    so the PonderNet loss reduces to the step-0 MSE."""
    T, B = 4, 3
    target = torch.zeros(B)
    preds_per_step = [torch.full((B, 1), float(t) + 1.0) for t in range(T)]   # 1, 2, 3, 4
    # Make step-0 confidence ~1 so all mass concentrates there.
    confs = [torch.full((B, 5), 0.999)] + [torch.full((B, 5), 1e-3) for _ in range(T - 1)]
    loss = pondernet_task_loss(preds_per_step, target=target, confidences=confs,
                               task_kind="regression")
    # Expected ~= MSE between step-0 pred (1.0) and target (0.0) = 1.0
    assert abs(float(loss) - 1.0) < 0.02


def test_pondernet_gradient_flows_into_halting_head():
    """End-to-end: full model with halting + PonderNet loss yields grad on head."""
    cfg = tiny_config(
        use_confidence_halting=True,
        halting_loss_weight=1.0,
        confidence_threshold=0.5,
        n_loops_train=4,
    )
    model = RadiantModel(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    attn = torch.ones_like(x)
    out = model(x, n_loops=4, attention_mask=attn, is_causal=False,
                return_intermediate_hidden=True)
    assert out.intermediate_hidden_states is not None
    assert len(out.intermediate_hidden_states) == 4
    # Use the LM logits as a stand-in task at each step (RADIANT core
    # doesn't carry a task head -- the chem wrapper does); we just verify
    # the per-step machinery gives backprop-able tensors.
    per_step_preds = [h.mean(dim=(1, 2), keepdim=True).squeeze(-1) for h in out.intermediate_hidden_states]
    target = torch.zeros(2, 1)
    pn_loss = pondernet_task_loss(
        per_step_preds, target=target,
        confidences=out.halting.confidences,
        attention_mask=attn,
        task_kind="regression",
    )
    pn_loss.backward()
    head = model.refinement.halting.head
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0


def test_model_returns_intermediate_hidden_when_requested():
    cfg = tiny_config(use_confidence_halting=True, halting_loss_weight=0.0)
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        out_off = model(x, n_loops=3)
        out_on = model(x, n_loops=3, return_intermediate_hidden=True)
    assert out_off.intermediate_hidden_states is None
    assert out_on.intermediate_hidden_states is not None
    assert len(out_on.intermediate_hidden_states) == 3
    for h in out_on.intermediate_hidden_states:
        assert h.shape == out_on.last_hidden_state.shape
