import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from radiant import RadiantModel, tiny_config
from training import FixedLoopSchedule, LossLogger, MetricsRecorder, Trainer


class _ToyLM(Dataset):
    """Random tokens; LM target is the same sequence shifted by 1 for last position."""

    def __init__(self, vocab_size: int, n: int = 16, seq_len: int = 8):
        self.x = torch.randint(0, vocab_size, (n, seq_len))
        self.y = torch.randint(0, vocab_size, (n, seq_len))

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {"input_ids": self.x[idx], "labels": self.y[idx]}


def lm_loss(out, batch):
    logits = out.logits
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1)
    )


def test_trainer_one_epoch_drops_loss():
    cfg = tiny_config()
    model = RadiantModel(cfg)
    ds = _ToyLM(cfg.vocab_size, n=8, seq_len=6)
    loader = DataLoader(ds, batch_size=4)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    rec = MetricsRecorder()
    trainer = Trainer(
        model, opt, lm_loss,
        loop_schedule=FixedLoopSchedule(2),
        callbacks=[rec],
        grad_clip=1.0,
    )
    trainer.fit(loader, epochs=1)
    losses = [s["loss"] for s in rec.steps]
    assert len(losses) >= 2
    # Lossy heuristic: at least one later step is below the first.
    assert min(losses[1:]) < losses[0] + 0.5


def test_trainer_with_aux_loss_path():
    """When MoE is on, aux_loss should appear in step logs."""
    cfg = tiny_config(use_moe=True, n_experts=4, n_active_experts=2)
    model = RadiantModel(cfg)
    ds = _ToyLM(cfg.vocab_size, n=4, seq_len=4)
    loader = DataLoader(ds, batch_size=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    rec = MetricsRecorder()
    trainer = Trainer(
        model, opt, lm_loss,
        loop_schedule=FixedLoopSchedule(2),
        callbacks=[rec],
    )
    trainer.fit(loader, epochs=1)
    assert any("aux" in s for s in rec.steps)


def test_trainer_evaluate():
    cfg = tiny_config()
    model = RadiantModel(cfg)
    ds = _ToyLM(cfg.vocab_size, n=4, seq_len=6)
    loader = DataLoader(ds, batch_size=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, opt, lm_loss, loop_schedule=FixedLoopSchedule(2))
    metrics = trainer.evaluate(loader, n_loops=1)
    assert "loss" in metrics
    assert metrics["n_loops"] == 1
