import json
from pathlib import Path

from radiant_qsar.finetune.select_checkpoint import select_checkpoint


def _write_cell(root: Path, target: str, split: str, pearson: float) -> None:
    cell = root / "radiant" / target / split
    cell.mkdir(parents=True)
    (cell / "best.pt").write_bytes(b"checkpoint")
    (cell / "chem_config.json").write_text("{}", encoding="utf-8")
    (cell / "result.json").write_text(
        json.dumps({
            "val": {"pearson": pearson},
            "model_path": "best.pt",
            "predictions_path": "predictions.csv",
        }),
        encoding="utf-8",
    )


def test_select_checkpoint_writes_best_radiant_manifest(tmp_path: Path):
    panel_root = tmp_path / "panel"
    _write_cell(panel_root, "CHEMBL1", "scaffold", 0.41)
    _write_cell(panel_root, "CHEMBL2", "scaffold", 0.52)

    out = tmp_path / "selected.json"
    selected = select_checkpoint(
        panel_root,
        out=out,
        split="scaffold",
        metric="pearson",
        vocab=tmp_path / "vocab.json",
    )

    assert selected["target_chembl_id"] == "CHEMBL2"
    assert selected["val_pearson"] == 0.52
    assert selected["task_name"] == "pchembl"
    assert json.loads(out.read_text(encoding="utf-8"))["checkpoint_path"].endswith("best.pt")
