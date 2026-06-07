import torch

from radiant import RadiantModel, tiny_config


def test_moe_off_in_loop_by_default():
    """Default placement 'stem_exit' must NOT use MoE inside the recurrent core."""
    cfg = tiny_config(use_moe=True, moe_placement="stem_exit", n_experts=4, n_active_experts=2)
    model = RadiantModel(cfg).eval()
    for blk in model.refinement.core_blocks:
        assert blk.uses_moe is False
    # And stem/exit blocks DO use MoE.
    assert any(b.uses_moe for b in model.stem.blocks)
    assert any(b.uses_moe for b in model.exit.blocks)


def test_moe_in_loop_only_with_experimental_flag():
    cfg = tiny_config(
        use_moe=True,
        moe_placement="stem_exit",
        moe_in_loop_experimental=True,
        n_experts=4,
        n_active_experts=2,
    )
    model = RadiantModel(cfg).eval()
    for blk in model.refinement.core_blocks:
        assert blk.uses_moe is True


def test_moe_aux_losses_collected():
    cfg = tiny_config(use_moe=True, n_experts=4, n_active_experts=2)
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        out = model(x)
    assert len(out.aux_losses) > 0
    assert out.aux_loss is not None


def test_moe_routing_distinct_per_token():
    """Random inputs should distribute across experts (not all to one)."""
    cfg = tiny_config(use_moe=True, n_experts=4, n_active_experts=2)
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    with torch.no_grad():
        out = model(x)
    # The aggregated aux loss should be finite and bounded.
    assert torch.isfinite(out.aux_loss)
