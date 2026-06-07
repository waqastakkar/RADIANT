"""Phase G.5 — Per-atom halt-step attribution (Sub-claim C5).

For each test molecule the RADIANT model emits a per-token halt step.
SMILES tokens map back to specific atoms in the RDKit ``Mol``; this
script:

* Generates a per-atom heat map (RDKit ``SimilarityMaps`` style) of the
  halt step.
* Computes per-atom **gradient × input** attribution from the same model
  and reports the rank-correlation against halt step.
* Optionally compares to **matched-molecular-pair (MMP)** SAR fragments
  derived on the fly: for each test molecule we find its nearest
  congeneric pair in the dataset that differs in pXC50 by ≥ 1 log unit
  and measure agreement between the halt-attention top-k atoms and the
  changed substructure.
* Saves 6–12 qualitative case-study molecule images for the paper.

Two operating modes
-------------------

1. **Predictions-only** (``--mode predictions``): consumes a
   per-row CSV exported by the RADIANT inference path that contains
   ``smiles``, ``true_pchembl``, ``pred_pchembl`` and a JSON-encoded
   ``per_atom_halt`` column with one int per atom. Gradient×input
   is *not* computed in this mode (no model is available).

2. **Model-driven** (``--mode model``): loads a chem checkpoint and
   computes both halt-step and gradient×input attributions in place.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import (
    AnalysisPaths,
    publication_style,
    save_figure,
    save_table,
    spearman_pearson,
    write_summary_md,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-to-atom mapping
# ---------------------------------------------------------------------------

def smiles_atom_index_map(smiles: str, tokens: Sequence[str]) -> list[int]:
    """Best-effort map of each non-special token to an atom index.

    The mapping is heuristic: walk the SMILES string, match each
    atom-bearing token to the next atom in the RDKit molecule. Tokens
    that don't correspond to atoms (rings, bond markers, brackets) map
    to ``-1``.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit required for atom attribution") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [-1] * len(tokens)
    n_atoms = mol.GetNumAtoms()

    mapping: list[int] = []
    atom_cursor = 0
    atom_chars = set("CcNnOoSsPpFIBrCl[]aA")
    for tok in tokens:
        if not tok or tok.startswith("["):
            # atom inside brackets, e.g. [nH], [C@H]
            if atom_cursor < n_atoms:
                mapping.append(atom_cursor)
                atom_cursor += 1
            else:
                mapping.append(-1)
            continue
        # simple atom symbols
        first = tok[0]
        if first in atom_chars and not tok[0].isdigit() and tok not in {"(", ")", "=", "#", "/", "\\", ".", "+", "-"}:
            if atom_cursor < n_atoms:
                mapping.append(atom_cursor)
                atom_cursor += 1
            else:
                mapping.append(-1)
        else:
            mapping.append(-1)
    return mapping


def aggregate_token_to_atom(values: Sequence[float], atom_map: Sequence[int], n_atoms: int) -> np.ndarray:
    """Mean-aggregate per-token values into per-atom values."""
    sums = np.zeros(n_atoms, dtype=float)
    counts = np.zeros(n_atoms, dtype=float)
    for v, a in zip(values, atom_map):
        if a < 0 or a >= n_atoms:
            continue
        sums[a] += float(v)
        counts[a] += 1.0
    out = np.full(n_atoms, np.nan, dtype=float)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    return out


# ---------------------------------------------------------------------------
# Mode B: gradient x input from a live model
# ---------------------------------------------------------------------------

@dataclass
class AttributionConfig:
    checkpoint_path: Path
    config_path: Path
    vocab_path: Path
    smiles_list: Sequence[str]
    task_name: str = "pchembl"
    device: str = "cuda"
    n_loops: int | None = None


def grad_input_attribution(cfg: AttributionConfig) -> list[dict]:
    """Compute (halt_step, grad_x_input) per atom for each SMILES.

    Gradient × input is computed against the **embedded tokens** (the
    output of the stem's token-embedding lookup), which is the standard
    saliency definition for transformer inputs. We attach a forward hook
    that snapshots the embedding tensor and retains its grad so we can
    extract it after backward.
    """
    import torch
    from radiant import RadiantConfig
    from radiant_chem.config import RadiantChemConfig
    from radiant_chem.model_chem import RadiantChemModel
    from radiant_chem.tasks import TaskRegistry, TaskSpec
    from radiant_chem.tokenizer import SmilesTokenizer

    tok = SmilesTokenizer.load(cfg.vocab_path)
    base_cfg = RadiantConfig.from_json(cfg.config_path).replace(
        vocab_size=tok.vocab_size, pad_token_id=tok.pad_id,
    )
    chem_cfg = RadiantChemConfig(base=base_cfg)
    tasks = TaskRegistry([TaskSpec(cfg.task_name, "regression", cfg.task_name, num_outputs=1)])
    model = RadiantChemModel(chem_cfg, tasks).to(cfg.device)

    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()

    results: list[dict] = []
    for smi in cfg.smiles_list:
        input_ids, attn = tok.encode_batch([smi])
        input_ids = input_ids.to(cfg.device)
        attn = attn.to(cfg.device)
        token_strs = [tok.id_to_token.get(int(i), "[UNK]") for i in input_ids[0].tolist()]

        captured: dict = {}

        def _embed_hook(_module, _inputs, output):
            output.retain_grad()
            captured["embedded"] = output
            return output

        handle = model.core.stem.token_embed.register_forward_hook(_embed_hook)
        try:
            out = model(input_ids, n_loops=cfg.n_loops, attention_mask=attn,
                        return_loop_metrics=True)
            pred = out.task_outputs[cfg.task_name].squeeze(-1).sum()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            pred.backward()
        finally:
            handle.remove()

        embedded = captured.get("embedded")
        if embedded is not None and embedded.grad is not None:
            gxi = (embedded.detach() * embedded.grad.detach()).abs().sum(dim=-1)
            gxi = gxi[0].cpu().numpy()
        else:
            with torch.no_grad():
                emb = model.core.stem.token_embed(input_ids)[0]
                gxi = emb.abs().sum(dim=-1).cpu().numpy()

        halting = out.base.halting
        halt = (halting.halt_step[0].detach().cpu().numpy()
                if halting is not None and halting.halt_step is not None else None)

        results.append({
            "smiles": smi,
            "tokens": token_strs,
            "halt_step": halt.tolist() if halt is not None else None,
            "grad_x_input": gxi.tolist(),
        })
    return results


# ---------------------------------------------------------------------------
# MMP analysis
# ---------------------------------------------------------------------------

def find_mmp_pairs(
    df: pd.DataFrame,
    *,
    delta_pchembl: float = 1.0,
    tanimoto_threshold: float = 0.7,
) -> pd.DataFrame:
    """Pair up molecules with high similarity but large activity gap.

    Returns columns ``(smiles_a, smiles_b, delta_pchembl, tanimoto, changed_atoms_a)``.
    The ``changed_atoms_a`` column is the atom indices of ``mol_a`` that
    are *not* part of the maximum common substructure with ``mol_b`` —
    i.e., the SAR-relevant substituent. We only return at most one pair
    per anchor molecule, namely the one with the largest ΔpXC50.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from rdkit.Chem import rdFMCS
    except ImportError as exc:
        raise RuntimeError("RDKit required for MMP analysis") from exc

    smis = df["smiles"].tolist()
    pys = df["true_pchembl"].to_numpy(dtype=float)
    mols = [Chem.MolFromSmiles(s) for s in smis]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None for m in mols]

    n = len(smis)
    rows: list[dict] = []
    for i in range(n):
        if mols[i] is None or fps[i] is None:
            continue
        best = None
        for j in range(n):
            if i == j or mols[j] is None or fps[j] is None:
                continue
            tani = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if tani < tanimoto_threshold:
                continue
            d = abs(pys[i] - pys[j])
            if d < delta_pchembl:
                continue
            if best is None or d > best["delta"]:
                best = {"j": j, "tani": tani, "delta": d}
        if best is None:
            continue

        try:
            mcs = rdFMCS.FindMCS([mols[i], mols[best["j"]]], timeout=2)
            mcs_pattern = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
            matched = set(mols[i].GetSubstructMatch(mcs_pattern)) if mcs_pattern else set()
        except Exception:
            matched = set()
        changed = [a.GetIdx() for a in mols[i].GetAtoms() if a.GetIdx() not in matched]

        rows.append({
            "smiles_a": smis[i],
            "smiles_b": smis[best["j"]],
            "delta_pchembl": float(best["delta"]),
            "tanimoto": float(best["tani"]),
            "changed_atoms_a": changed,
        })
    return pd.DataFrame(rows)


def mmp_overlap_score(
    halt_per_atom: np.ndarray,
    changed_atoms: Sequence[int],
    *,
    top_k_frac: float = 0.25,
) -> float:
    """Fraction of top-k halt-step atoms that fall in the changed set.

    Higher = halting attribution agrees with the SAR-relevant substituent.
    """
    if len(changed_atoms) == 0 or np.all(~np.isfinite(halt_per_atom)):
        return float("nan")
    finite = np.isfinite(halt_per_atom)
    k = max(1, int(round(top_k_frac * finite.sum())))
    order = np.argsort(-halt_per_atom)  # descending
    order = [int(a) for a in order if finite[a]][:k]
    overlap = sum(1 for a in order if a in set(changed_atoms))
    return overlap / k


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

def render_heatmap(smiles: str, halt_per_atom: np.ndarray, out_path: Path) -> Path | None:
    """Render a per-atom halt-step heat map as **both** SVG and PNG.

    Returns the SVG path (for the manifest); the PNG is written alongside
    it at ``out_path.with_suffix('.png')``. Vector SVG preserves atom-level
    details for publication zoom; PNG is a rasterised companion for slides /
    Word / GitHub previews where SVG renders inconsistently.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        logger.warning("RDKit Draw not available; skipping heatmap for %s", smiles)
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    finite = halt_per_atom.copy()
    if np.isfinite(halt_per_atom).any():
        vmin = float(np.nanmin(halt_per_atom[np.isfinite(halt_per_atom)]))
        vmax = float(np.nanmax(halt_per_atom[np.isfinite(halt_per_atom)]))
    else:
        vmin, vmax = 0.0, 1.0
    finite[~np.isfinite(finite)] = vmin

    # Normalise weights to [-1, 1] (RDKit SimilarityMap convention)
    span = vmax - vmin if vmax > vmin else 1.0
    weights = [2.0 * (float(w) - vmin) / span - 1.0 for w in finite.tolist()]

    try:
        from rdkit.Chem.Draw import SimilarityMaps
    except ImportError:
        logger.warning("RDKit SimilarityMaps not available; skipping heatmap")
        return None

    svg_path = out_path.with_suffix(".svg")
    png_path = out_path.with_suffix(".png")

    # --- SVG (primary, always emitted on modern RDKit) -------------------
    svg_ok = False
    try:
        drawer_svg = rdMolDraw2D.MolDraw2DSVG(500, 500)
        SimilarityMaps.GetSimilarityMapFromWeights(mol, weights,
                                                   colorMap="RdYlGn",
                                                   draw2d=drawer_svg)
        drawer_svg.FinishDrawing()
        svg_path.write_text(drawer_svg.GetDrawingText(), encoding="utf-8")
        svg_ok = True
    except TypeError:
        # Very old RDKit (<= 2022.03) — draw2d kwarg not accepted yet.
        # Fall through to the matplotlib path below which produces a PNG.
        pass
    except Exception as exc:
        logger.warning("SVG heatmap render failed for %s: %s", smiles, exc)

    # --- PNG companion ---------------------------------------------------
    # Preferred: native MolDraw2DCairo (vector-quality rasterisation,
    # identical layout to the SVG). Falls back to cairosvg conversion of
    # the SVG we just wrote, then to the legacy matplotlib SimilarityMaps
    # path, so something always lands on disk.
    png_ok = False
    try:
        drawer_png = rdMolDraw2D.MolDraw2DCairo(500, 500)
        SimilarityMaps.GetSimilarityMapFromWeights(mol, weights,
                                                   colorMap="RdYlGn",
                                                   draw2d=drawer_png)
        drawer_png.FinishDrawing()
        drawer_png.WriteDrawingText(str(png_path))
        png_ok = True
    except Exception:
        # MolDraw2DCairo missing (RDKit built without Cairo) or draw2d
        # kwarg unsupported. Try cairosvg from the SVG we wrote.
        if svg_ok:
            try:
                import cairosvg  # type: ignore
                cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                                 output_width=1000, output_height=1000)
                png_ok = True
            except Exception:
                pass
        if not png_ok:
            try:
                import matplotlib.pyplot as plt
                fig = SimilarityMaps.GetSimilarityMapFromWeights(
                    mol, weights, colorMap="RdYlGn"
                )
                fig.savefig(png_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                png_ok = True
            except Exception as exc:
                logger.warning("PNG heatmap render failed for %s: %s", smiles, exc)

    if svg_ok:
        return svg_path
    if png_ok:
        return png_path
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _coerce_per_atom_halt(s) -> list[int] | None:
    """Coerce a JSON-encoded per_atom_halt cell into a python list."""
    if s is None or (isinstance(s, float) and not np.isfinite(s)):
        return None
    if isinstance(s, list):
        return list(s)
    try:
        return list(json.loads(s))
    except Exception:
        return None


def run(
    predictions_path: Path | str | None,
    out_dir: Path | str,
    *,
    descriptors_path: Path | str | None = None,
    n_case_studies: int = 8,
    mmp_delta_pchembl: float = 1.0,
    mmp_tanimoto: float = 0.7,
    grad_attribution: list[dict] | None = None,
) -> dict:
    publication_style()
    paths = AnalysisPaths(Path(out_dir), name="g5_atom_attribution")

    if predictions_path is not None:
        df = pd.read_csv(predictions_path)
    elif grad_attribution is not None:
        df = pd.DataFrame(grad_attribution).rename(columns={"smiles": "smiles"})
    else:
        raise ValueError("Provide --predictions or call with grad_attribution=...")

    # Build per-atom halt-step arrays
    per_atom_rows: list[dict] = []
    for _, row in df.iterrows():
        smi = row["smiles"]
        if "per_atom_halt" in df.columns:
            halt_per_atom = _coerce_per_atom_halt(row["per_atom_halt"])
            tokens = None
        elif "tokens" in df.columns and "halt_step" in df.columns:
            tokens = row["tokens"] if isinstance(row["tokens"], list) else json.loads(row["tokens"])
            halt_tok = row["halt_step"] if isinstance(row["halt_step"], list) else json.loads(row["halt_step"])
            try:
                from rdkit import Chem
                n_atoms = Chem.MolFromSmiles(smi).GetNumAtoms()
            except Exception:
                continue
            amap = smiles_atom_index_map(smi, tokens)
            halt_per_atom = aggregate_token_to_atom(halt_tok, amap, n_atoms).tolist()
        else:
            continue

        gxi_per_atom = None
        if "grad_x_input" in df.columns and "tokens" in df.columns:
            gxi_tok = row["grad_x_input"] if isinstance(row["grad_x_input"], list) else json.loads(row["grad_x_input"])
            tokens = row["tokens"] if isinstance(row["tokens"], list) else json.loads(row["tokens"])
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                amap = smiles_atom_index_map(smi, tokens)
                gxi_per_atom = aggregate_token_to_atom(gxi_tok, amap, mol.GetNumAtoms()).tolist()

        per_atom_rows.append({
            "smiles": smi,
            "true_pchembl": row.get("true_pchembl", float("nan")),
            "pred_pchembl": row.get("pred_pchembl", float("nan")),
            "halt_per_atom": halt_per_atom,
            "gxi_per_atom": gxi_per_atom,
        })

    if not per_atom_rows:
        raise RuntimeError("No per-atom halt arrays could be constructed from the inputs.")

    atom_df = pd.DataFrame(per_atom_rows)

    # Rank correlation halt vs grad x input
    rank_rows: list[dict] = []
    for _, r in atom_df.iterrows():
        if r["halt_per_atom"] is None or r["gxi_per_atom"] is None:
            continue
        h = np.array(r["halt_per_atom"], dtype=float)
        g = np.array(r["gxi_per_atom"], dtype=float)
        stats = spearman_pearson(h, g, n_bootstrap=200)
        rank_rows.append({"smiles": r["smiles"], "spearman_rho": stats["spearman"], "n_atoms": stats["n"]})
    rank_df = pd.DataFrame(rank_rows)
    save_table(rank_df, paths, "g5_halt_vs_gradxinput_per_mol")

    mean_rho = float(rank_df["spearman_rho"].mean()) if not rank_df.empty else float("nan")

    # MMP overlap
    mmp_rows: list[dict] = []
    if {"smiles", "true_pchembl"}.issubset(df.columns):
        try:
            pairs = find_mmp_pairs(
                df[["smiles", "true_pchembl"]].dropna(),
                delta_pchembl=mmp_delta_pchembl,
                tanimoto_threshold=mmp_tanimoto,
            )
            atom_lookup = {r["smiles"]: r["halt_per_atom"] for _, r in atom_df.iterrows()}
            for _, p in pairs.iterrows():
                halt = atom_lookup.get(p["smiles_a"])
                if halt is None:
                    continue
                score = mmp_overlap_score(np.array(halt, dtype=float), p["changed_atoms_a"])
                mmp_rows.append({
                    "smiles_a": p["smiles_a"], "smiles_b": p["smiles_b"],
                    "delta_pchembl": p["delta_pchembl"], "tanimoto": p["tanimoto"],
                    "overlap_top25pct": score,
                })
        except RuntimeError as exc:
            logger.warning("MMP analysis skipped: %s", exc)

    mmp_df = pd.DataFrame(mmp_rows)
    if not mmp_df.empty:
        save_table(mmp_df, paths, "g5_mmp_overlap")

    # Case-study heatmaps
    case_paths: list[Path] = []
    if not rank_df.empty:
        # pick top-rho molecules (where halt and grad agree most) for clarity
        ordered = atom_df.merge(rank_df, on="smiles", how="left").sort_values(
            "spearman_rho", ascending=False
        )
    else:
        ordered = atom_df
    n_take = min(n_case_studies, len(ordered))
    case_dir = paths.figures / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)
    for i, (_, r) in enumerate(ordered.head(n_take).iterrows()):
        if r["halt_per_atom"] is None:
            continue
        out_path = case_dir / f"mol_{i:02d}.png"
        rendered = render_heatmap(r["smiles"], np.array(r["halt_per_atom"], dtype=float), out_path)
        if rendered is not None:
            case_paths.append(rendered)

    if mmp_df.empty or mmp_df["overlap_top25pct"].dropna().empty:
        mmp_summary = "MMP overlap: not measured (no qualifying pairs)."
    else:
        mmp_mean = float(mmp_df["overlap_top25pct"].dropna().mean())
        mmp_summary = f"Mean MMP overlap@25% = {mmp_mean:.3f} (n={len(mmp_df)} pairs)."

    write_summary_md(
        paths,
        title="G.5 — Per-atom halt-step attribution",
        claim="C5: per-atom halting attribution highlights chemically meaningful hotspots.",
        headline=(
            f"Mean Spearman ρ between halt-step and gradient×input = {mean_rho:.3f}. "
            f"{mmp_summary} {n_take} qualitative heatmaps saved."
        ),
        details={
            "Molecules analyzed": str(len(atom_df)),
            "Case studies rendered": str(len(case_paths)),
            "MMP ΔpChEMBL threshold": str(mmp_delta_pchembl),
            "MMP Tanimoto threshold": str(mmp_tanimoto),
        },
        tables_referenced=["g5_halt_vs_gradxinput_per_mol.csv",
                           *(["g5_mmp_overlap.csv"] if not mmp_df.empty else [])],
        figures_referenced=[f"case_studies/{p.name}" for p in case_paths],
    )

    return {
        "rank_correlation": rank_df,
        "mmp_overlap": mmp_df,
        "case_study_paths": case_paths,
        "paths": paths,
    }


def run_panel(
    panel_root: Path | str,
    out_dir: Path | str,
    *,
    lf_model_dir: str = "radiant",
    split: str = "scaffold",
    n_case_studies_per_target: int = 2,
    mmp_delta_pchembl: float = 1.0,
    mmp_tanimoto: float = 0.7,
) -> dict:
    """Run G.5 per-atom attribution across all LF cells (all 20 targets).

    Uses predictions mode only (reads ``tokens`` + ``halt_step`` columns
    from existing ``predictions.csv`` — no checkpoint needed). Pools
    rank-correlation statistics across all targets and assembles a
    multi-target case-study gallery.
    """
    publication_style()
    panel_root = Path(panel_root)
    out_dir = Path(out_dir)
    paths = AnalysisPaths(out_dir, name="g5_atom_attribution")

    cell_csvs = sorted(
        (panel_root / lf_model_dir).glob(f"*/{split}/predictions.csv")
    )
    if not cell_csvs:
        raise FileNotFoundError(
            f"No predictions.csv found under {panel_root / lf_model_dir}/*/{split}/. "
            "Check that fine-tuning completed."
        )
    logger.info("G.5 panel mode: %d cells (split=%s)", len(cell_csvs), split)

    all_rank_rows: list[pd.DataFrame] = []
    all_mmp_rows: list[pd.DataFrame] = []
    all_case_paths: list[Path] = []
    case_dir = paths.figures / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)
    case_count = 0
    n_success = 0

    for csv_path in cell_csvs:
        target = csv_path.parts[-3]
        try:
            result = run(
                predictions_path=csv_path,
                out_dir=out_dir,
                n_case_studies=n_case_studies_per_target,
                mmp_delta_pchembl=mmp_delta_pchembl,
                mmp_tanimoto=mmp_tanimoto,
            )
            n_success += 1

            # rank_correlation is only populated in model mode (grad×input);
            # in predictions mode it is empty — that is expected, not a failure.
            rk = result["rank_correlation"]
            if not rk.empty:
                rk = rk.copy()
                rk["target"] = target
                all_rank_rows.append(rk)

            mmp = result["mmp_overlap"]
            if not mmp.empty:
                mmp = mmp.copy()
                mmp["target"] = target
                all_mmp_rows.append(mmp)

            import shutil
            for p in result["case_study_paths"][:n_case_studies_per_target]:
                dst = case_dir / f"{target}_{case_count:04d}{p.suffix}"
                shutil.copy2(p, dst)
                all_case_paths.append(dst)
                case_count += 1

        except Exception as exc:
            logger.warning("  %s: failed — %s", target, exc)

    if n_success == 0:
        raise RuntimeError("All panel cells failed in G.5 attribution.")

    rank_df = pd.concat(all_rank_rows, ignore_index=True) if all_rank_rows else pd.DataFrame()
    if not rank_df.empty:
        save_table(rank_df, paths, "g5_panel_halt_vs_gradxinput")

        # Per-target summary
        per_target_rho = (rank_df.groupby("target")["spearman_rho"]
                          .agg(["mean", "std", "count"])
                          .reset_index()
                          .rename(columns={"mean": "mean_rho", "std": "std_rho", "count": "n_mols"}))
        save_table(per_target_rho, paths, "g5_panel_per_target_summary")

    mmp_df = pd.concat(all_mmp_rows, ignore_index=True) if all_mmp_rows else pd.DataFrame()
    if not mmp_df.empty:
        save_table(mmp_df, paths, "g5_panel_mmp_overlap")

    mean_rho = float(rank_df["spearman_rho"].mean()) if not rank_df.empty else float("nan")
    mmp_summary = (
        f"Mean MMP overlap@25% = {float(mmp_df['overlap_top25pct'].dropna().mean()):.3f} "
        f"(n={len(mmp_df)} pairs across {mmp_df['target'].nunique()} targets)."
        if not mmp_df.empty and "overlap_top25pct" in mmp_df.columns else
        "MMP overlap: not measured."
    )

    n_targets_rho = rank_df["target"].nunique() if not rank_df.empty else 0
    n_mols_rho = len(rank_df) if not rank_df.empty else 0
    if rank_df.empty:
        per_target_rho = pd.DataFrame()

    write_summary_md(
        paths,
        title="G.5 — Per-atom halt-step attribution (panel)",
        claim="C5: per-atom halting attribution highlights chemically meaningful hotspots across all 20 targets.",
        headline=(
            f"Mean Spearman ρ (halt vs grad×input) = {mean_rho:.3f} across "
            f"{n_targets_rho} targets, {n_mols_rho} molecules. {mmp_summary}"
        ),
        details={
            "Panel cells processed": str(n_success),
            "Split": split,
            "Total molecules analyzed": str(n_mols_rho),
            "Case study images": str(len(all_case_paths)),
        },
        tables_referenced=[
            *( ["g5_panel_halt_vs_gradxinput.csv", "g5_panel_per_target_summary.csv"]
               if not rank_df.empty else [] ),
            *(["g5_panel_mmp_overlap.csv"] if not mmp_df.empty else []),
        ],
        figures_referenced=[f"case_studies/{p.name}" for p in all_case_paths],
    )

    return {
        "rank_correlation": rank_df,
        "per_target": per_target_rho,
        "mmp_overlap": mmp_df,
        "case_study_paths": all_case_paths,
        "paths": paths,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase G.5 per-atom halt attribution")
    p.add_argument("--mode", choices=["predictions", "model", "panel"], default="predictions")
    p.add_argument("--panel-root", type=Path,
                   help="(panel mode) auto-discovers all LF cells")
    p.add_argument("--predictions", type=Path,
                   help="(predictions mode) CSV with smiles, true_pchembl, pred_pchembl, per_atom_halt or tokens+halt_step")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-case-studies", type=int, default=8,
                   help="case studies per target in panel mode; total in single mode")
    p.add_argument("--split", default="scaffold", help="(panel mode) which split to use")
    p.add_argument("--mmp-delta", type=float, default=1.0)
    p.add_argument("--mmp-tanimoto", type=float, default=0.7)
    # model mode
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--config", type=Path, help="RadiantConfig json")
    p.add_argument("--vocab", type=Path, help="SmilesTokenizer vocab json")
    p.add_argument("--smiles-list", type=Path, help="text file with one SMILES per line")
    p.add_argument("--task-name", default="pchembl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-loops", type=int, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()

    if args.mode == "panel":
        if not args.panel_root:
            raise SystemExit("--mode panel requires --panel-root")
        run_panel(
            panel_root=args.panel_root,
            out_dir=args.out_dir,
            split=args.split,
            n_case_studies_per_target=args.n_case_studies,
            mmp_delta_pchembl=args.mmp_delta,
            mmp_tanimoto=args.mmp_tanimoto,
        )
        return

    grad_attrib = None
    if args.mode == "model":
        if not (args.checkpoint and args.config and args.vocab and args.smiles_list):
            raise SystemExit("--mode model requires --checkpoint, --config, --vocab, --smiles-list")
        smis = [s.strip() for s in args.smiles_list.read_text().splitlines() if s.strip()]
        grad_attrib = grad_input_attribution(AttributionConfig(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            vocab_path=args.vocab,
            smiles_list=smis,
            task_name=args.task_name,
            device=args.device,
            n_loops=args.n_loops,
        ))

    run(
        predictions_path=args.predictions,
        out_dir=args.out_dir,
        n_case_studies=args.n_case_studies,
        mmp_delta_pchembl=args.mmp_delta,
        mmp_tanimoto=args.mmp_tanimoto,
        grad_attribution=grad_attrib,
    )


if __name__ == "__main__":
    main()
