import torch

from radiant import tiny_config
from radiant import RegressionHead
from radiant_chem import RadiantChemConfig, RadiantChemModel, SmilesTokenizer
from radiant_chem.objectives import MaskedLMLoss, RegressionLoss
from radiant_chem.tasks import TaskRegistry, TaskSpec


CORPUS = ["CCO", "c1ccccc1", "CC(=O)O", "Cc1ccncc1", "Brc1ccc(N)cc1"]


def _setup(*, use_halting=False, with_tasks=True):
    tok = SmilesTokenizer.from_corpus(CORPUS)
    base = tiny_config(
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_id,
        max_seq_len=64,
        use_confidence_halting=use_halting,
    )
    cfg = RadiantChemConfig(base=base)
    tasks = TaskRegistry()
    if with_tasks:
        tasks.register(TaskSpec(name="logp", kind="regression", target_column="logP", num_outputs=1))
        tasks.register(TaskSpec(name="cls", kind="classification", target_column="active", num_outputs=2))
    model = RadiantChemModel(cfg, tasks)
    return tok, model


def test_chem_model_forward_shapes():
    tok, model = _setup()
    model.eval()
    ids, attn = tok.encode_batch(CORPUS)
    with torch.no_grad():
        out = model(ids, attention_mask=attn, n_loops=2)
    B = ids.size(0)
    assert out.base.logits.shape == (B, ids.size(1), tok.vocab_size)
    assert out.pooled.shape == (B, model.cfg.base.d_model)
    assert out.task_outputs["logp"].shape == (B, 1)
    assert out.task_outputs["cls"].shape == (B, 2)


def test_chem_mlm_forward_finite_loss():
    tok, model = _setup(with_tasks=False)
    model.train()
    ids, attn = tok.encode_batch(CORPUS)
    # Simulate masking: pick first non-special token in each row.
    labels = torch.full_like(ids, -100)
    for b in range(ids.size(0)):
        for j in range(ids.size(1)):
            if attn[b, j].item() == 1 and ids[b, j].item() not in (tok.bos_id, tok.eos_id, tok.pad_id):
                labels[b, j] = ids[b, j]
                ids[b, j] = tok.mask_id
                break
    logits = model.forward_mlm(ids, attention_mask=attn, n_loops=2)
    loss = MaskedLMLoss()(logits, labels)
    assert torch.isfinite(loss)


def test_chem_property_grad_flow():
    tok, model = _setup()
    model.train()
    ids, attn = tok.encode_batch(CORPUS)
    targets = torch.randn(ids.size(0), 1)
    out = model(ids, attention_mask=attn, n_loops=2)
    loss = RegressionLoss()(out.task_outputs["logp"], targets)
    loss.backward()
    g = model.task_heads["logp"].proj.weight.grad
    assert g is not None and torch.isfinite(g).all()


def test_regression_head_can_use_mlp_path_without_changing_shape():
    cfg = tiny_config(d_model=32, vocab_size=16, max_seq_len=16)
    head = RegressionHead(cfg, num_outputs=1, hidden_dim=48, dropout=0.1)
    h = torch.randn(4, 7, cfg.d_model)
    mask = torch.ones(4, 7, dtype=torch.long)
    out = head(h, mask)
    assert out.shape == (4, 1)
    loss = RegressionLoss(kind="huber", huber_beta=0.5)(out.squeeze(-1), torch.randn(4))
    loss.backward()
    assert head.proj.weight.grad is not None
    assert torch.isfinite(head.proj.weight.grad).all()


def test_chem_config_regression_head_knobs_round_trip(tmp_path):
    cfg = RadiantChemConfig(
        base=tiny_config(),
        regression_head_hidden_dim=128,
        regression_head_dropout=0.2,
    )
    path = tmp_path / "chem_config.json"
    cfg.to_json(path)
    loaded = RadiantChemConfig.from_json(path)
    assert loaded.regression_head_hidden_dim == 128
    assert loaded.regression_head_dropout == 0.2


def test_chem_per_step_task_outputs_have_correct_shapes():
    """When return_per_step_task=True, each task gets one prediction per loop step."""
    tok, model = _setup(use_halting=True)
    model.eval()
    ids, attn = tok.encode_batch(CORPUS)
    with torch.no_grad():
        out = model(ids, attention_mask=attn, n_loops=3, return_per_step_task=True)
    assert out.per_step_task_outputs is not None
    assert set(out.per_step_task_outputs.keys()) == {"logp", "cls"}
    for name, expected_shape in [("logp", (ids.size(0), 1)), ("cls", (ids.size(0), 2))]:
        preds_per_step = out.per_step_task_outputs[name]
        assert len(preds_per_step) == 3
        for p in preds_per_step:
            assert p.shape == expected_shape


def test_chem_embed_pooled_shape():
    tok, model = _setup(with_tasks=False)
    ids, attn = tok.encode_batch(CORPUS)
    emb = model.embed_pooled(ids, attention_mask=attn, n_loops=2)
    assert emb.shape == (ids.size(0), model.cfg.base.d_model)


def test_add_task_after_init():
    tok, model = _setup(with_tasks=False)
    model.add_task(TaskSpec(name="solubility", kind="regression", target_column="logS", num_outputs=1))
    assert "solubility" in model.task_heads
    ids, attn = tok.encode_batch(CORPUS[:2])
    with torch.no_grad():
        out = model(ids, attention_mask=attn, n_loops=2)
    assert out.task_outputs["solubility"].shape == (2, 1)


def test_chem_with_halting_returns_trace():
    tok, model = _setup(use_halting=True, with_tasks=False)
    model.eval()
    ids, attn = tok.encode_batch(CORPUS[:3])
    with torch.no_grad():
        out = model(ids, attention_mask=attn, n_loops=2)
    assert out.base.halting is not None
    assert out.base.halting.avg_depth is not None
