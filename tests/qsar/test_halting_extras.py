"""Unit tests for the halting-extras accumulator.

These tests don't load a real RADIANT checkpoint -- they hand-craft a
minimal HaltingTrace plus tokenizer/input_ids and verify the accumulator
produces correctly-shaped, correctly-typed, JSON-round-trippable rows.
That keeps the test cheap (~0.1 s) while still pinning the schema.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from radiant.confidence_halting import HaltingTrace
from radiant_qsar.eval.halting_extras import (
    HALTING_EXTRA_COLUMNS,
    HaltingExtrasAccumulator,
    extract_halting_extras,
)


def _build_trace(B: int, S: int, T: int, *, seed: int = 0) -> HaltingTrace:
    """Synthesize a HaltingTrace with T loop steps."""
    g = torch.Generator().manual_seed(seed)
    trace = HaltingTrace()
    for t in range(T):
        trace.append(torch.rand((B, S), generator=g) * 0.3)   # per-loop conf ∈ (0, 0.3)
    trace.finalize(threshold=0.5)
    assert trace.halt_step is not None
    return trace


def _id_to_token(vocab_size: int) -> dict[int, str]:
    return {i: f"tok{i}" for i in range(vocab_size)}


def test_extract_halting_extras_returns_canonical_schema():
    B, S, T = 3, 6, 4
    trace = _build_trace(B, S, T)
    input_ids = torch.randint(low=1, high=10, size=(B, S))
    # Pad the tail of each row to test that padding tokens are excluded.
    attention_mask = torch.ones_like(input_ids)
    input_ids[:, -1] = 0
    attention_mask[:, -1] = 0

    out = extract_halting_extras(
        halting=trace,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_id=0,
        id_to_token=_id_to_token(20),
    )

    # All four columns present, each list of length B.
    for col in HALTING_EXTRA_COLUMNS:
        assert col in out, f"missing column {col}"
        assert len(out[col]) == B

    # halt_step and tokens are JSON-encoded; round-trip should give same length.
    for i in range(B):
        halt = json.loads(out["halt_step"][i])
        toks = json.loads(out["tokens"][i])
        assert isinstance(halt, list) and isinstance(toks, list)
        assert len(halt) == len(toks) == S - 1, "padded token should be excluded"
        for v in halt:
            assert 0 <= int(v) < T

    # effective_depth is the mean halt_step + 1 over valid tokens.
    for i in range(B):
        halt = json.loads(out["halt_step"][i])
        assert out["effective_depth"][i] == pytest.approx(np.mean(halt) + 1.0)

    # confidence_var is a non-negative scalar.
    for i in range(B):
        assert out["confidence_var"][i] >= 0.0

    # soft_effective_depth is the expected halt step (1-indexed) over the
    # continuous halt distribution; must be in [1, T] and finite.
    for i in range(B):
        s = out["soft_effective_depth"][i]
        assert 1.0 - 1e-4 <= s <= T + 1e-4, f"soft_effective_depth out of range: {s}"


def test_extract_halting_extras_returns_blank_when_halting_off():
    B, S = 2, 4
    input_ids = torch.randint(low=1, high=10, size=(B, S))
    attention_mask = torch.ones_like(input_ids)
    out = extract_halting_extras(
        halting=None,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_id=0,
        id_to_token=_id_to_token(20),
    )
    # tokens are still decoded; halt arrays are None.
    assert all(out["halt_step"][i] is None for i in range(B))
    assert all(out["tokens"][i] is not None for i in range(B))
    assert all(np.isnan(out["effective_depth"][i]) for i in range(B))


def test_accumulator_concatenates_batches():
    B1, B2, S, T = 2, 3, 5, 4
    acc = HaltingExtrasAccumulator()
    for B in (B1, B2):
        trace = _build_trace(B, S, T)
        input_ids = torch.randint(low=1, high=10, size=(B, S))
        attention_mask = torch.ones_like(input_ids)
        acc.add(
            halting=trace,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_id=0,
            id_to_token=_id_to_token(20),
        )
    extras = acc.finalize()
    assert len(extras["halt_step"]) == B1 + B2
    assert len(extras["effective_depth"]) == B1 + B2
    assert len(acc) == B1 + B2


def test_extras_survive_csv_roundtrip(tmp_path):
    """write_predictions + read_csv must preserve halt_step JSON strings."""
    import csv
    from radiant_qsar.eval.predictions import write_predictions

    B, S, T = 2, 3, 3
    trace = _build_trace(B, S, T)
    input_ids = torch.randint(low=1, high=10, size=(B, S))
    attention_mask = torch.ones_like(input_ids)
    extras = extract_halting_extras(
        halting=trace, input_ids=input_ids, attention_mask=attention_mask,
        pad_id=0, id_to_token=_id_to_token(20),
    )

    p = write_predictions(
        tmp_path,
        indices=[0, 1],
        inchikeys=["AAA", "BBB"],
        smiles=["CCO", "c1ccccc1"],
        true_pchembl=[6.0, 7.5],
        pred_pchembl=[6.1, 7.4],
        target_chembl_id="CHEMBL999",
        split_kind="scaffold",
        extra_columns=extras,
    )
    assert p.exists()
    with p.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    # JSON column survives.
    halt = json.loads(rows[0]["halt_step"])
    assert isinstance(halt, list)
    # effective_depth is parseable as float.
    float(rows[0]["effective_depth"])
