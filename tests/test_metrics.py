import torch

from radiant import (
    RadiantModel,
    LoopMetrics,
    halting_summary,
    spectral_radius_estimate,
    tiny_config,
)


def test_loop_metrics_records_correct_count():
    cfg = tiny_config()
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        out = model(x, n_loops=3, return_loop_metrics=True)
    assert out.loop_metrics is not None
    assert len(out.loop_metrics.norms) == 3
    # cos_to_prev has T-1 entries because there's no "previous" at t=0.
    assert len(out.loop_metrics.cos_to_prev) == 2


def test_loop_metrics_finite():
    cfg = tiny_config()
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out = model(x, n_loops=4, return_loop_metrics=True)
    for v in out.loop_metrics.norms:
        assert v == v and v != float("inf")  # finite
    for v in out.loop_metrics.cos_to_prev:
        assert -1.01 <= v <= 1.01


def test_halting_summary_with_halting_enabled():
    cfg = tiny_config(use_confidence_halting=True, confidence_threshold=0.5)
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        out = model(x, n_loops=3)
    summary = halting_summary(out.halting)
    assert "avg_depth" in summary
    assert summary["avg_depth"] >= 1.0


def test_spectral_radius_runs():
    """Smoke test: spectral_radius_estimate returns a finite positive number."""
    cfg = tiny_config()
    model = RadiantModel(cfg).eval()
    B, S = 1, 4
    x = torch.randint(0, cfg.vocab_size, (B, S))
    with torch.no_grad():
        e, _ = model.stem(x, model.rope_cos[:S], model.rope_sin[:S], None, True)
    sigma = spectral_radius_estimate(
        model.refinement, e, e, model.rope_cos[:S], model.rope_sin[:S],
        t=0, attn_mask=None, is_causal=True, n_iter=4,
    )
    assert sigma > 0.0
    assert sigma == sigma  # not NaN
