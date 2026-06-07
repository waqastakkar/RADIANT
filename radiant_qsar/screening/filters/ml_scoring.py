"""ML-based scoring filters.

Three flavors:

* :class:`RADIANTPotency` -- loads a trained RADIANT-Chem checkpoint
  and scores each candidate with the model's predicted pXC50.
* :class:`MorganRFPotency` -- the same interface for a Morgan-RF
  baseline saved by :mod:`radiant_qsar.baselines.morgan_rf`. Lets you
  run the *same* filter pipeline against an RF baseline so the only
  delta in a screening campaign is the scoring model, not the recipe.
* :class:`GenericMlScorer` -- thin wrapper around any
  ``Callable[[Mol], float]`` for plugging in TDC, ADMETlab, or a custom
  model without writing a new filter class.

None of these filters are registered by default -- callers instantiate
them explicitly because each requires a checkpoint or scorer function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from radiant_qsar.screening.base import Filter, FilterContext, FilterResult


class GenericMlScorer(Filter):
    """Wrap any ``Callable[[Mol], float]`` as a filter.

    Args:
        name:        unique identifier for the audit log.
        scorer:      callable returning a single float per molecule.
        min_score:   keep when score >= min_score (set to ``None`` to disable).
        max_score:   keep when score <= max_score (set to ``None`` to disable).
    """

    def __init__(
        self,
        name: str,
        scorer: Callable[[object], float],
        *,
        min_score: float | None = None,
        max_score: float | None = None,
    ) -> None:
        if min_score is None and max_score is None:
            raise ValueError("Provide at least one of min_score / max_score.")
        self.name = name
        self._scorer = scorer
        self.min_score = min_score
        self.max_score = max_score

    def apply(self, mol, ctx) -> FilterResult:
        try:
            s = float(self._scorer(mol))
        except Exception as exc:
            return FilterResult(self.name, False, f"scorer error: {exc}")
        if self.min_score is not None and s < self.min_score:
            return FilterResult(self.name, False, f"{self.name} score {s:.3f} < {self.min_score}", s)
        if self.max_score is not None and s > self.max_score:
            return FilterResult(self.name, False, f"{self.name} score {s:.3f} > {self.max_score}", s)
        return FilterResult(self.name, True, score=s)


class RADIANTPotency(Filter):
    """Predicted pXC50 from a trained RADIANT-Chem checkpoint.

    Loads the checkpoint lazily on first call (so importing this module
    doesn't force the model into memory). Single-molecule scoring is
    intentionally simple here; for batched scoring of large libraries see
    :func:`score_library_batched`.
    """

    name = "radiant_potency"

    def __init__(
        self,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        *,
        min_pchembl: float = 6.0,
        n_loops: int | None = None,
        device: str = "cpu",
        max_len: int = 192,
        task_name: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)
        self.min_pchembl = min_pchembl
        self.n_loops = n_loops
        self.device = device
        self.max_len = max_len
        self.task_name = task_name
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch

        from radiant_chem import RadiantChemModel, SmilesTokenizer

        self._tokenizer = SmilesTokenizer.load(self.vocab_path)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        # Expect a dict with 'model_state' and 'chem_config' (or similar).
        # The pretrain driver writes ``chem_config.json`` next to the .pt,
        # which we use to instantiate the architecture.
        cfg_path = self.checkpoint_path.parent / "chem_config.json"
        from radiant_chem.config import RadiantChemConfig
        from radiant_chem.tasks import TaskRegistry, TaskSpec

        chem_cfg = RadiantChemConfig.from_json(cfg_path)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        task_name = self.task_name
        if task_name is None:
            for key in state:
                if key.startswith("task_heads."):
                    parts = key.split(".")
                    if len(parts) >= 3:
                        task_name = parts[1]
                        self.task_name = task_name
                        break
        tasks = None
        if task_name is not None:
            tasks = TaskRegistry([TaskSpec(task_name, "regression", task_name, num_outputs=1)])
        model = RadiantChemModel(chem_cfg, tasks)
        model.load_state_dict(state, strict=False)
        model.eval().to(self.device)
        self._model = model

    def apply(self, mol, ctx) -> FilterResult:
        import torch
        from rdkit import Chem

        self._ensure_loaded()
        smi = Chem.MolToSmiles(mol)
        ids, attn = self._tokenizer.encode_batch([smi], max_len=self.max_len)
        ids = ids.to(self.device)
        attn = attn.to(self.device)
        with torch.no_grad():
            out = self._model(ids, attention_mask=attn, n_loops=self.n_loops)
        if self.task_name and self.task_name in out.task_outputs:
            score = float(out.task_outputs[self.task_name].squeeze().item())
        else:
            # Default: mean pooled embedding norm; not a real prediction --
            # callers should specify ``task_name`` when running production
            # screens.
            return FilterResult(
                self.name, False,
                "task_name not specified or task head not in checkpoint",
            )
        ok = score >= self.min_pchembl
        return FilterResult(self.name, ok,
                            f"predicted pchembl {score:.2f} < {self.min_pchembl}" if not ok else "",
                            score)


# ---------------------------------------------------------------------------
class MorganRFPotency(Filter):
    """Predicted pXC50 from a Morgan/RF baseline saved by
    :mod:`radiant_qsar.baselines.morgan_rf`.

    Loads the joblib bundle lazily on first call. Single-molecule
    scoring; for batched scoring of large libraries the screening
    Pipeline already streams molecule-by-molecule with negligible
    per-call overhead because the FP transform is vectorized.
    """

    name = "morgan_rf_potency"

    def __init__(
        self,
        bundle_path: str | "Path",
        *,
        min_pchembl: float = 6.0,
    ) -> None:
        from pathlib import Path as _P
        self.bundle_path = _P(bundle_path)
        self.min_pchembl = min_pchembl
        self._bundle = None

    def _ensure_loaded(self):
        if self._bundle is None:
            from radiant_qsar.baselines.morgan_rf import load_bundle
            self._bundle = load_bundle(self.bundle_path)

    def apply(self, mol, ctx) -> FilterResult:
        import numpy as np
        from rdkit import Chem

        self._ensure_loaded()
        smi = Chem.MolToSmiles(mol) if mol is not None else None
        if not smi:
            return FilterResult(self.name, False, "unparseable molecule")
        # Score using the cached bundle directly instead of reloading from disk
        # on every call (predict_smiles_from_ckpt deserializes the joblib each time).
        from radiant_qsar.baselines.morgan_rf import _morgan_fp_matrix

        X = _morgan_fp_matrix([smi], self._bundle["fp_radius"], self._bundle["fp_n_bits"])
        preds = self._bundle["model"].predict(X)
        # Mark unparseable inputs (all-zero FP) as NaN
        if np.all(X[0] == 0):
            preds[0] = np.nan
        if not preds.size or not float("-inf") < float(preds[0]) < float("inf"):
            return FilterResult(self.name, False, "scorer returned NaN/inf")
        score = float(preds[0])
        ok = score >= self.min_pchembl
        return FilterResult(
            self.name, ok,
            f"RF predicted pchembl {score:.2f} < {self.min_pchembl}" if not ok else "",
            score,
        )
