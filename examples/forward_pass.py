"""Minimal forward pass: instantiate a tiny RADIANT, run it, print shapes.

Usage::

    python examples/forward_pass.py
"""

from __future__ import annotations

import torch

from radiant import RadiantModel, tiny_config


def main() -> None:
    cfg = tiny_config()
    model = RadiantModel(cfg).eval()

    print(f"params:           {model.num_params():,}")
    print(f"recurrent params: {model.num_recurrent_params():,}")

    x = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        out = model(x, n_loops=4, return_loop_metrics=True)

    print(f"input_ids:          {tuple(x.shape)}")
    print(f"logits:             {tuple(out.logits.shape)}")
    print(f"last_hidden_state:  {tuple(out.last_hidden_state.shape)}")
    print(f"n_loops_executed:   {out.n_loops_executed}")
    print(f"per-loop hidden norms: {[round(v, 3) for v in out.loop_metrics.norms]}")
    print(f"cosine to previous:    {[round(v, 4) for v in out.loop_metrics.cos_to_prev]}")


if __name__ == "__main__":
    main()
