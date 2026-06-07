"""Extract per-row halting signals from a RADIANT chem forward pass.

Phase G analyses need the same set of per-molecule halting summaries
emitted by *every* RADIANT inference path (fine-tune end-of-training
eval, the standalone re-emitter, the loop-depth sweep). Centralizing
the extraction here keeps those callers consistent.

The five columns we emit on each row of `predictions.csv`:

* ``halt_step``        — JSON-encoded list[int], per-token halt step
                         (0-indexed loop at which each token first
                         crossed the confidence threshold; defaults to
                         ``n_loops_executed - 1`` if never crossed).
                         Pad tokens are excluded.
* ``effective_depth``  — float, ``mean(halt_step + 1)`` over non-pad
                         tokens. Same scalar Phase G.1 / G.4 expect.
* ``confidence_var``   — float, variance across loop steps of the
                         per-loop mean confidence over the row's
                         non-pad tokens. A proxy for "how much did the
                         halting head hesitate". Phase G.3 maps this
                         to a sigma via the upstream `halt_var_to_sigma`.
* ``tokens``           — JSON-encoded list[str], decoded token strings
                         in the same order as halt_step. Lets G.5 do
                         token -> atom mapping with RDKit at analysis
                         time without re-tokenizing.
* ``per_atom_halt``    — JSON-encoded list[float] (optional, emitted
                         only when ``include_per_atom=True``). Mean
                         halt-step value aggregated per heavy atom in
                         the RDKit molecule. Heavier path because it
                         imports rdkit; off by default in the
                         training loop, on for the standalone re-emit.

Schema is stable; downstream analyses key on these names.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


HALTING_EXTRA_COLUMNS: tuple[str, ...] = (
    "halt_step",
    "effective_depth",
    "soft_effective_depth",
    "confidence_var",
    "tokens",
)
PER_ATOM_EXTRA_COLUMN: str = "per_atom_halt"


# ---------------------------------------------------------------------------
# Per-batch extraction
# ---------------------------------------------------------------------------

def extract_halting_extras(
    *,
    halting,                    # radiant.confidence_halting.HaltingTrace | None
    input_ids,                  # torch.LongTensor (B, S)
    attention_mask,             # torch.LongTensor (B, S)
    pad_id: int,
    id_to_token: dict[int, str],
    smiles_batch: Sequence[str] | None = None,
    include_per_atom: bool = False,
) -> dict[str, list]:
    """Return per-row halting summaries for one batch.

    All returned lists have length B. Each ``halt_step[i]`` /
    ``tokens[i]`` is JSON-encoded so it round-trips through CSV without
    custom parsing.

    When ``include_per_atom=True`` we also return JSON-encoded per-atom
    halt arrays via the same token->atom heuristic that
    :mod:`radiant_qsar.analyses.g5_atom_attribution` uses at analysis
    time. SMILES is taken from ``smiles_batch`` (required when
    per-atom is on).
    """
    import torch

    B, S = input_ids.shape
    n_rows = int(B)

    out: dict[str, list] = {
        "halt_step": [None] * n_rows,
        "effective_depth": [float("nan")] * n_rows,
        "soft_effective_depth": [float("nan")] * n_rows,
        "confidence_var": [float("nan")] * n_rows,
        "tokens": [None] * n_rows,
    }
    if include_per_atom:
        out[PER_ATOM_EXTRA_COLUMN] = [None] * n_rows

    if halting is None or halting.halt_step is None:
        # Halting was disabled; emit decoded tokens but leave halt
        # columns blank so the downstream loaders know to treat them
        # as missing.
        for b in range(n_rows):
            mask = attention_mask[b].bool() & (input_ids[b] != pad_id)
            ids = input_ids[b][mask].cpu().tolist()
            out["tokens"][b] = json.dumps([id_to_token.get(int(i), "[UNK]") for i in ids])
        return out

    halt_step = halting.halt_step.detach().cpu()           # (B, S) int
    confidences = (torch.stack(halting.confidences, dim=0).detach().cpu()
                   if halting.confidences else None)        # (T, B, S) float | None

    # Per-token unconditional halt-probability mass `p_halt(t)`, derived
    # from the same tail-absorbed model distribution as
    # `halting_kl_loss` / `pondernet_task_loss`. Captures the continuous
    # behavior of the head before the binary threshold flattens it.
    p_halt: torch.Tensor | None = None
    if confidences is not None and confidences.size(0) >= 2:
        from radiant.confidence_halting import _per_step_halt_probabilities
        # _per_step_halt_probabilities expects a list of (B, S) tensors.
        p_halt = _per_step_halt_probabilities(
            [confidences[t] for t in range(confidences.size(0))]
        )                                                        # (T, B, S)

    for b in range(n_rows):
        mask_row = (attention_mask[b].bool() & (input_ids[b] != pad_id)).cpu()
        if not bool(mask_row.any()):
            out["tokens"][b] = json.dumps([])
            continue
        ids = input_ids[b][mask_row].cpu().tolist()
        tok_strs = [id_to_token.get(int(i), "[UNK]") for i in ids]
        out["tokens"][b] = json.dumps(tok_strs)

        halt_row = halt_step[b][mask_row].numpy().astype(int)
        out["halt_step"][b] = json.dumps(halt_row.tolist())
        out["effective_depth"][b] = float(halt_row.mean() + 1.0)

        if p_halt is not None:
            # soft_effective_depth: E[halt step + 1] over the continuous
            # halt distribution. With T loops, steps run 0..T-1, so the
            # expected depth = sum_t (t + 1) * p_halt(t). Mean over valid
            # tokens then a single scalar per molecule.
            T = p_halt.size(0)
            steps_1based = torch.arange(1, T + 1, dtype=p_halt.dtype)
            p_b = p_halt[:, b, :][:, mask_row]                    # (T, n_valid)
            ed_per_token = (p_b * steps_1based.view(T, 1)).sum(dim=0)  # (n_valid,)
            out["soft_effective_depth"][b] = float(ed_per_token.mean().item())

        if confidences is not None:
            # Mean per-loop confidence over the row's valid tokens, then
            # variance across loops. Scalar in [0, 0.25] for sigmoid-bounded
            # signals; we report the raw variance.
            mean_per_loop = confidences[:, b, :][:, mask_row].mean(dim=-1).numpy()
            out["confidence_var"][b] = float(np.var(mean_per_loop))

        if include_per_atom:
            if smiles_batch is None:
                raise ValueError("include_per_atom=True requires smiles_batch")
            out[PER_ATOM_EXTRA_COLUMN][b] = _per_atom_halt_json(
                smiles_batch[b], tok_strs, halt_row,
            )

    return out


def _per_atom_halt_json(smiles: str, tokens: Sequence[str], halt_per_token: np.ndarray) -> str:
    """Aggregate per-token halt values onto heavy atoms via the g5 mapping."""
    try:
        from rdkit import Chem
        from radiant_qsar.analyses.g5_atom_attribution import (
            aggregate_token_to_atom,
            smiles_atom_index_map,
        )
    except Exception as exc:
        logger.debug("per-atom halt skipped for %s: %s", smiles, exc)
        return json.dumps(None)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return json.dumps(None)
    amap = smiles_atom_index_map(smiles, list(tokens))
    per_atom = aggregate_token_to_atom(halt_per_token.tolist(), amap, mol.GetNumAtoms())
    # NaN -> None so JSON survives a roundtrip.
    return json.dumps([None if not np.isfinite(v) else float(v) for v in per_atom.tolist()])


# ---------------------------------------------------------------------------
# Stream accumulator -- used by callers that iterate the test DataLoader
# ---------------------------------------------------------------------------

class HaltingExtrasAccumulator:
    """Helper that gathers per-row halting extras across many batches.

    Usage::

        acc = HaltingExtrasAccumulator(include_per_atom=False)
        for batch in test_loader:
            out = model(...)
            acc.add(
                halting=out.base.halting,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pad_id=pad,
                id_to_token=tok.id_to_token,
            )
        extras = acc.finalize()
        write_predictions(..., extra_columns=extras)
    """

    def __init__(self, *, include_per_atom: bool = False) -> None:
        self.include_per_atom = include_per_atom
        cols = list(HALTING_EXTRA_COLUMNS)
        if include_per_atom:
            cols.append(PER_ATOM_EXTRA_COLUMN)
        self._cols = cols
        self._rows: dict[str, list] = {c: [] for c in cols}

    def add(
        self,
        *,
        halting,
        input_ids,
        attention_mask,
        pad_id: int,
        id_to_token: dict[int, str],
        smiles_batch: Sequence[str] | None = None,
    ) -> None:
        extras = extract_halting_extras(
            halting=halting,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_id=pad_id,
            id_to_token=id_to_token,
            smiles_batch=smiles_batch,
            include_per_atom=self.include_per_atom,
        )
        for c in self._cols:
            self._rows[c].extend(extras[c])

    def finalize(self) -> dict[str, list]:
        return dict(self._rows)

    def __len__(self) -> int:
        # All columns have the same length; pick one.
        return len(self._rows[self._cols[0]])
