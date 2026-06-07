"""Generation demo: same checkpoint sampled at different n_loops.

Usage::

    python examples/generate.py
"""

from __future__ import annotations

import torch

from radiant import RadiantModel, tiny_config


def main() -> None:
    cfg = tiny_config(iteration_signal_kind="sinusoidal")  # extrapolates safely
    model = RadiantModel(cfg).eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    print(f"prompt ids: {prompt.tolist()}")

    for n_loops in (1, 2, 4, cfg.max_loops, cfg.max_loops + 4):
        gen = model.generate(prompt, max_new_tokens=8, n_loops=n_loops, temperature=1.0, top_k=10)
        print(f"  n_loops={n_loops:>2}: {gen.tolist()}")


if __name__ == "__main__":
    main()
