"""Preflight architecture-match check between pretrain ckpt and fine-tune config.

This is the regression that catches the silent ``strict=False`` half-load:
if the user picks the wrong --config for a given checkpoint, the fine-tune
should fail loudly rather than train from partially-random init.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from radiant import RadiantConfig, RadiantModel  # noqa: E402
from radiant_chem import RadiantChemConfig, RadiantChemModel  # noqa: E402
from radiant_qsar.finetune.single_task import _preflight_state_dict_match  # noqa: E402


def _make_model(d_model: int):
    cfg = RadiantConfig(
        vocab_size=64, d_model=d_model, n_query_heads=4, n_kv_heads=2,
        head_dim=16, d_ff=64, max_seq_len=32,
        n_stem_blocks=1, n_refinement_blocks=1, n_exit_blocks=1,
    )
    chem_cfg = RadiantChemConfig(base=cfg)
    return RadiantChemModel(chem_cfg)


def test_preflight_passes_when_configs_match(tmp_path: Path):
    """Same architecture on both sides -> 100% match -> no raise."""
    model = _make_model(d_model=64)
    state = model.state_dict()
    # Should NOT raise.
    _preflight_state_dict_match(
        model, state,
        ckpt_path=tmp_path / "fake.pt", config_path=tmp_path / "cfg.json",
    )


def test_preflight_raises_on_dim_mismatch(tmp_path: Path):
    """Pretrain at d_model=64, fine-tune config asks for d_model=128 -> raise SystemExit."""
    pretrain_model = _make_model(d_model=64)
    finetune_model = _make_model(d_model=128)
    with pytest.raises(SystemExit) as exc:
        _preflight_state_dict_match(
            finetune_model, pretrain_model.state_dict(),
            ckpt_path=tmp_path / "pretrain.pt", config_path=tmp_path / "wrong.json",
        )
    msg = str(exc.value)
    assert "ARCHITECTURE MISMATCH" in msg
    assert "config" in msg.lower()
    # Should surface the most likely fix.
    assert "pretrain config = fine-tune config" in msg


def test_preflight_raises_on_almost_empty_overlap(tmp_path: Path):
    """No common tensors at all (e.g. a random ckpt from another model) -> raise."""
    finetune_model = _make_model(d_model=64)
    state = {"completely.unrelated.key": torch.zeros(8, 8)}
    with pytest.raises(SystemExit) as exc:
        _preflight_state_dict_match(
            finetune_model, state,
            ckpt_path=tmp_path / "wrong.pt", config_path=tmp_path / "cfg.json",
        )
    assert "ARCHITECTURE MISMATCH" in str(exc.value)


def test_preflight_threshold_passes_with_extra_keys(tmp_path: Path):
    """Pretrain ckpt with EXTRA keys (e.g. confidence_halting head not in fine-tune model)
    should pass -- those go to ``unexpected``, not mismatch."""
    model = _make_model(d_model=64)
    state = dict(model.state_dict())
    state["some_extra_key_from_pretrain.weight"] = torch.zeros(4, 4)
    # 100% of the model's expected tensors match -> still passes.
    _preflight_state_dict_match(
        model, state,
        ckpt_path=tmp_path / "ok.pt", config_path=tmp_path / "cfg.json",
    )
