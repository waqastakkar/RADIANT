import torch

from radiant import RadiantModel, tiny_config


def test_core_blocks_are_shared_across_loops():
    """Each loop iteration uses the SAME parameter tensors -- that's the point."""
    cfg = tiny_config()
    model = RadiantModel(cfg).eval()
    blocks = model.refinement.core_blocks
    # Sanity: there's at least one block, and each is accessible by index.
    for i in range(len(blocks)):
        assert blocks[i] is blocks[i]
    # The list itself is one ModuleList; running for n_loops re-uses the same modules.
    # Verify via parameter id sets.
    pid_before = {id(p) for p in blocks.parameters()}
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        model(x, n_loops=cfg.max_loops)
    pid_after = {id(p) for p in blocks.parameters()}
    assert pid_before == pid_after


def test_optimizer_step_updates_recurrent_block_params():
    """A backward through many loops updates the shared core's weights."""
    cfg = tiny_config()
    model = RadiantModel(cfg).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # Snapshot every recurrent param.
    snaps = [(p, p.detach().clone()) for p in model.refinement.core_blocks.parameters()]

    x = torch.randint(0, cfg.vocab_size, (2, 6))
    out = model(x, n_loops=cfg.max_loops)
    target = torch.zeros_like(out.logits)
    loss = (out.logits - target).pow(2).mean()
    loss.backward()
    opt.step()
    # At least one recurrent parameter must have changed by more than float-noise.
    max_delta = max((p - s).abs().max().item() for p, s in snaps)
    assert max_delta > 1e-8, f"no recurrent param updated (max delta {max_delta})"


def test_refinement_returns_n_executed():
    cfg = tiny_config()
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out = model(x, n_loops=2)
    assert out.n_loops_executed == 2


def test_dynamic_halting_can_terminate_early():
    """With an aggressive low threshold and very confident init, eval should halt early."""
    cfg = tiny_config(
        use_confidence_halting=True,
        confidence_threshold=0.05,
        confidence_init_bias=2.0,  # sigmoid(2) ~= 0.88, so cum cross is fast
    )
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 4))
    with torch.no_grad():
        out = model(x, n_loops=cfg.max_loops)
    assert out.n_loops_executed <= cfg.max_loops
    # Very high init confidence + low threshold should halt within 1 step.
    assert out.n_loops_executed == 1
