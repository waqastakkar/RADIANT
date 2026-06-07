"""Reference: a standard non-recurrent transformer for RADIANT baselines.

Builds a vanilla transformer with the SAME building blocks (RMSNorm,
GQAAttention, SwiGLUFeedForward, RoPE) but no recurrent loop -- depth is
applied once, top to bottom. Two convenience constructors:

    same_param_baseline(cfg, n_loops_target)
        Replicates the parameter budget of a RadiantModel evaluated for
        ``n_loops_target`` loops. Achieves this by stacking
        ``stem + (refinement * 1 + exit)`` once -- weight-shared depth
        with the same number of UNIQUE parameters.

    same_compute_baseline(cfg, n_loops_target)
        Replicates the FLOPs of RADIANT unrolled for ``n_loops_target``
        loops by stacking ``n_loops_target`` independent refinement
        blocks (no weight sharing).

The script trains both baselines and a RADIANT model on the synthetic
shift-LM task and prints the comparison.

Usage::

    python examples/baseline_transformer.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from radiant import (
    RadiantConfig,
    RadiantModel,
    RMSNorm,
    TransformerBlock,
    build_rope_cache,
    tiny_config,
)
from training import FixedLoopSchedule, MetricsRecorder, Trainer


# ----------------------------------------------------------------------
# A plain stacked transformer using RADIANT's building blocks.
# ----------------------------------------------------------------------
class StackedTransformer(nn.Module):
    """Vanilla pre-norm transformer with RoPE attention and SwiGLU FFN.

    Use ``share_blocks=True`` to weight-tie the entire stack into a single
    block (a same-parameter baseline against RADIANT); ``False`` for an
    independent-weights baseline.
    """

    def __init__(
        self,
        cfg: RadiantConfig,
        *,
        depth: int,
        share_blocks: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.depth = depth
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        nn.init.trunc_normal_(self.token_embed.weight, std=cfg.initializer_range, a=-2 * cfg.initializer_range, b=2 * cfg.initializer_range)

        if share_blocks:
            self._shared_block = TransformerBlock(cfg, moe=False)
            self.blocks = nn.ModuleList([self._shared_block])  # logically depth times
        else:
            self.blocks = nn.ModuleList([TransformerBlock(cfg, moe=False) for _ in range(depth)])
        self.share_blocks = share_blocks

        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, theta=cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, n_loops=None, attention_mask=None, is_causal=True, return_loop_metrics=False):
        # Signature mirrors RadiantModel so the same Trainer can drive it.
        S = input_ids.size(1)
        h = self.token_embed(input_ids)
        cos, sin = self.rope_cos[:S], self.rope_sin[:S]
        if self.share_blocks:
            for _ in range(self.depth):
                h, _ = self._shared_block(h, cos, sin, None, is_causal)
        else:
            for blk in self.blocks:
                h, _ = blk(h, cos, sin, None, is_causal)
        h = self.final_norm(h)

        # Return an object with the same surface as RadiantOutput so
        # downstream code (Trainer, examples) doesn't branch.
        class _Out:
            def __init__(self, logits, last_hidden_state):
                self.logits = logits
                self.last_hidden_state = last_hidden_state
                self.aux_losses = []
                self.aux_loss = None
                self.halting = None
                self.loop_metrics = None
                self.n_loops_executed = 1
        return _Out(self.lm_head(h), h)


# ----------------------------------------------------------------------
# Convenience constructors aligned to RADIANT's parameter / compute budget.
# ----------------------------------------------------------------------
def same_param_baseline(cfg: RadiantConfig, *, n_loops_target: int) -> StackedTransformer:
    """Untied stack with the same total trainable parameters as RADIANT.

    RADIANT has ``stem(s) + refinement(r) + exit(e)`` unique block-tensors
    (the loop reuses ``r`` -- it does not introduce new parameters). A
    transformer of depth ``s + r + e`` with no weight tying matches that
    parameter count exactly.
    """
    depth = cfg.n_stem_blocks + cfg.n_refinement_blocks + cfg.n_exit_blocks
    return StackedTransformer(cfg, depth=depth, share_blocks=False)


def same_compute_baseline(cfg: RadiantConfig, *, n_loops_target: int) -> StackedTransformer:
    """Untied stack with the same FLOPs/forward as RADIANT unrolled to ``n_loops_target``.

    RADIANT applies ``stem (s) + n_loops_target * refinement (r) + exit (e)``
    blocks per forward; this baseline does the same with all blocks untied.
    Same compute, more parameters.
    """
    depth = cfg.n_stem_blocks + n_loops_target * cfg.n_refinement_blocks + cfg.n_exit_blocks
    return StackedTransformer(cfg, depth=depth, share_blocks=False)


def shared_depth_baseline(cfg: RadiantConfig, *, n_loops_target: int) -> StackedTransformer:
    """Single shared block applied for the full unrolled depth (no input re-injection).

    This is the closest baseline to RADIANT that *removes* the
    StateAnchorUpdate / IterationSignal / IterationAdapter machinery -- it
    is simply a weight-tied recurrent transformer (``Universal-Transformer``
    style without ACT). Comparing RADIANT to this isolates the
    contribution of our recurrence machinery from the basic
    "shared-depth recurrence" effect.
    """
    depth = cfg.n_stem_blocks + n_loops_target * cfg.n_refinement_blocks + cfg.n_exit_blocks
    return StackedTransformer(cfg, depth=depth, share_blocks=True)


# ----------------------------------------------------------------------
# Synthetic task + training driver
# ----------------------------------------------------------------------
class _Shift(Dataset):
    """Predict the previous token (input shifted right by one)."""

    def __init__(self, n=256, seq_len=16, vocab_size=8):
        self.x = torch.randint(0, vocab_size, (n, seq_len))
        self.y = torch.cat([torch.zeros((n, 1), dtype=torch.long), self.x[:, :-1]], dim=1)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {"input_ids": self.x[idx], "labels": self.y[idx]}


def lm_loss(out, batch):
    return F.cross_entropy(
        out.logits.reshape(-1, out.logits.size(-1)),
        batch["labels"].reshape(-1),
    )


def _train(model, train_loader, val_loader, *, n_loops):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rec = MetricsRecorder()
    Trainer(
        model, opt, lm_loss,
        loop_schedule=FixedLoopSchedule(n_loops),
        callbacks=[rec],
        grad_clip=1.0,
    ).fit(train_loader, val_loader=val_loader, epochs=2)
    return rec


def main() -> None:
    cfg = tiny_config(vocab_size=16, max_seq_len=32, iteration_signal_kind="sinusoidal")
    n_loops = cfg.n_loops_train
    train_ds = _Shift(n=256, seq_len=16, vocab_size=cfg.vocab_size)
    val_ds = _Shift(n=64, seq_len=16, vocab_size=cfg.vocab_size)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    rows: list[tuple[str, int, float]] = []

    print("=== RADIANT ===")
    lf = RadiantModel(cfg)
    rec = _train(lf, train_loader, val_loader, n_loops=n_loops)
    rows.append(("radiant", lf.num_params(), rec.epochs[-1]["val_loss"]))

    print("=== same-parameter baseline ===")
    sp = same_param_baseline(cfg, n_loops_target=n_loops)
    rec = _train(sp, train_loader, val_loader, n_loops=1)
    rows.append(("same_param_baseline", sp.num_params(), rec.epochs[-1]["val_loss"]))

    print("=== same-compute baseline ===")
    sc = same_compute_baseline(cfg, n_loops_target=n_loops)
    rec = _train(sc, train_loader, val_loader, n_loops=1)
    rows.append(("same_compute_baseline", sc.num_params(), rec.epochs[-1]["val_loss"]))

    print("=== shared-depth baseline (universal-transformer style) ===")
    sd = shared_depth_baseline(cfg, n_loops_target=n_loops)
    rec = _train(sd, train_loader, val_loader, n_loops=1)
    rows.append(("shared_depth_baseline", sd.num_params(), rec.epochs[-1]["val_loss"]))

    print("\nmodel,params,val_loss")
    for name, p, l in rows:
        print(f"{name},{p},{l:.4f}")


if __name__ == "__main__":
    main()
