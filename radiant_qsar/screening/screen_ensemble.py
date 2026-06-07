"""Ensemble virtual screening with multiple RADIANT-Chem checkpoints.

Designed for the natural-product + PTP1B + BBB workflow but generic to
any (library, target, model-set) triplet.

Pipeline
--------
Stage 1 (external, not in this module): call
``radiant_qsar.screening.prepare_library`` with a CNS / BBB profile to
get a filtered SMI of brain-penetrant compounds + funnel JSON.

Stage 2 (this module, mode=score): load each split model in turn, batch
score every SMILES in the filtered SMI, write a CSV with one column per
split model and ``ensemble_mean / ensemble_std``.

Stage 3 (this module, mode=analyze): from the scored CSV + funnel JSON,
emit publication-style figures and tables:
* Filter funnel
* Predicted pChEMBL distribution
* Ensemble agreement (mean vs stdev scatter)
* BBB metric scatter (TPSA vs MW, coloured by predicted potency)
* Top-K structure thumbnails (PNG + SVG) and a combined grid
* Per-model agreement matrix
* ``summary.md`` (NOT stitched into Phase G -- this is the screening report)

Usage
-----
::

    python -m radiant_qsar.screening.screen_ensemble \\
        --mode all \\
        --filtered-smi runs/screening/PTP1B_NP_BBB/01_filtered.smi \\
        --funnel-json  runs/screening/PTP1B_NP_BBB/funnel.json \\
        --models       runs/screening_models/PTP1B_CHEMBL335/scaffold/best.pt \\
                       runs/screening_models/PTP1B_CHEMBL335/random/best.pt \\
                       runs/screening_models/PTP1B_CHEMBL335/time/best.pt \\
                       runs/screening_models/PTP1B_CHEMBL335/cluster/best.pt \\
                       runs/screening_models/PTP1B_CHEMBL335/activity_cliff/best.pt \\
        --model-names  scaffold random time cluster activity_cliff \\
        --vocab        data/zinc20/smiles_vocab.json \\
        --config       configs/radiant_75m.json \\
        --out-dir      runs/screening/PTP1B_NP_BBB \\
        --top-k        20 \\
        --device       cuda
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 2: score
# ---------------------------------------------------------------------------

@dataclass
class _ScoreCfg:
    config_path: Path
    vocab_path: Path
    device: str = "cuda"
    batch_size: int = 64
    max_seq_len: int | None = None  # None -> use config value
    task_name: str = "pchembl"


def _load_model(checkpoint_path: Path, cfg: _ScoreCfg):
    import torch
    from radiant import RadiantConfig
    from radiant_chem.config import RadiantChemConfig
    from radiant_chem.model_chem import RadiantChemModel
    from radiant_chem.tasks import TaskRegistry, TaskSpec
    from radiant_chem.tokenizer import SmilesTokenizer

    tok = SmilesTokenizer.load(cfg.vocab_path)
    base = RadiantConfig.from_json(cfg.config_path).replace(
        vocab_size=tok.vocab_size, pad_token_id=tok.pad_id,
    )
    cell_chem_cfg_path = checkpoint_path.parent / "chem_config.json"
    if cell_chem_cfg_path.exists():
        cc = json.loads(cell_chem_cfg_path.read_text(encoding="utf-8"))
        chem_kwargs = {k: v for k, v in cc.items() if k != "base"}
        chem_cfg = RadiantChemConfig(base=base, **chem_kwargs)
    else:
        chem_cfg = RadiantChemConfig(base=base)
    tasks = TaskRegistry([TaskSpec(cfg.task_name, "regression",
                                   cfg.task_name, num_outputs=1)])
    model = RadiantChemModel(chem_cfg, tasks).to(cfg.device)
    ckpt = torch.load(checkpoint_path, map_location=cfg.device,
                      weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    max_seq = cfg.max_seq_len or base.max_seq_len
    return model, tok, max_seq


def _score_one_model(model, tok, max_seq: int, smiles: Sequence[str],
                     batch_size: int, device: str, task_name: str) -> np.ndarray:
    import torch
    preds = np.full(len(smiles), np.nan, dtype=float)
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            chunk = list(smiles[start:start + batch_size])
            input_ids, attn = tok.encode_batch(chunk)
            if input_ids.shape[1] > max_seq:
                input_ids = input_ids[:, :max_seq]
                attn = attn[:, :max_seq]
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            out = model(input_ids, attention_mask=attn,
                        return_loop_metrics=True)
            v = out.task_outputs[task_name].squeeze(-1).cpu().numpy()
            preds[start:start + len(chunk)] = v
    return preds


def _load_filtered_smi(path: Path) -> pd.DataFrame:
    """Read a .smi produced by prepare_library: ``smiles<TAB>id`` per line."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            smi = parts[0]
            mol_id = parts[1] if len(parts) > 1 else f"mol_{len(rows):06d}"
            rows.append({"id": mol_id, "smiles": smi})
    return pd.DataFrame(rows)


def score(*, filtered_smi: Path, model_paths: list[Path],
          model_names: list[str], cfg: _ScoreCfg, out_csv: Path) -> pd.DataFrame:
    """Run every model on every smiles, return a wide DF with ensemble stats."""
    df = _load_filtered_smi(filtered_smi)
    if df.empty:
        raise RuntimeError(f"no smiles in {filtered_smi}")
    logger.info("scoring %d compounds with %d models on %s",
                len(df), len(model_paths), cfg.device)

    smiles = df["smiles"].astype(str).tolist()
    for ckpt, name in zip(model_paths, model_names):
        logger.info("  loading %s (%s)", name, ckpt)
        model, tok, max_seq = _load_model(ckpt, cfg)
        preds = _score_one_model(model, tok, max_seq, smiles,
                                 batch_size=cfg.batch_size,
                                 device=cfg.device, task_name=cfg.task_name)
        df[f"pred_{name}"] = preds
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    pred_cols = [f"pred_{n}" for n in model_names]
    df["ensemble_mean"] = df[pred_cols].mean(axis=1)
    df["ensemble_std"] = df[pred_cols].std(axis=1, ddof=0)
    df["ensemble_min"] = df[pred_cols].min(axis=1)
    df["ensemble_max"] = df[pred_cols].max(axis=1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("wrote %s (%d rows, %d models)", out_csv, len(df), len(model_names))
    return df


# ---------------------------------------------------------------------------
# Stage 3: analyze
# ---------------------------------------------------------------------------

NATURE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442",
          "#56B4E9", "#E69F00", "#000000"]


def _apply_publication_style() -> None:
    """Apply the same Times-compatible + bold style used by Phase G."""
    from radiant_qsar.analyses.common import publication_style
    publication_style()


def _physchem(smiles_list: Sequence[str]) -> pd.DataFrame:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski
    rows = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            rows.append({"MW": np.nan, "LogP": np.nan, "TPSA": np.nan,
                         "HBD": np.nan, "HBA": np.nan, "RotB": np.nan})
            continue
        rows.append({
            "MW": float(Descriptors.MolWt(m)),
            "LogP": float(Crippen.MolLogP(m)),
            "TPSA": float(Descriptors.TPSA(m)),
            "HBD": int(Lipinski.NumHDonors(m)),
            "HBA": int(Lipinski.NumHAcceptors(m)),
            "RotB": int(Lipinski.NumRotatableBonds(m)),
        })
    return pd.DataFrame(rows)


def _save_figs(fig, paths_no_ext: Path) -> None:
    paths_no_ext.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(str(paths_no_ext) + f".{ext}",
                    dpi=200 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")


def _plot_funnel(funnel_json: Path, out_path: Path) -> None:
    """Render the prepare_library funnel.

    Real schema (prepare_library): ``{n_input, n_passed, n_failed,
    pass_rate, elapsed_s, rejects_by_filter: {filter_name: count}, profile}``.
    We draw a two-panel figure: (left) input → passed waterfall;
    (right) rejects-by-filter bar.
    """
    import matplotlib.pyplot as plt
    if not funnel_json.exists():
        return
    summary = json.loads(funnel_json.read_text(encoding="utf-8"))

    n_input = int(summary.get("n_input", 0))
    n_passed = int(summary.get("n_passed", 0))
    n_failed = int(summary.get("n_failed", n_input - n_passed))
    rej = summary.get("rejects_by_filter", {})
    profile = summary.get("profile", "")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6),
                             gridspec_kw={"width_ratios": [1.0, 1.6]})

    # Left panel: input -> passed waterfall
    ax = axes[0]
    bars = ax.bar(["input", "passed"], [n_input, n_passed],
                  color=[NATURE[7], NATURE[2]], edgecolor="none")
    for b, v in zip(bars, [n_input, n_passed]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Compounds", fontweight="bold")
    pct = (n_passed / n_input * 100.0) if n_input else 0.0
    ax.set_title(f"BBB+ pass rate: {pct:.1f}%\n(profile: {profile})",
                 fontweight="bold", fontsize=10)

    # Right panel: per-filter rejection counts (descending)
    ax = axes[1]
    if rej:
        items = sorted(rej.items(), key=lambda kv: -int(kv[1]))
        names = [k for k, _ in items]
        cnts = [int(v) for _, v in items]
        ax.barh(names[::-1], cnts[::-1], color=NATURE[1], edgecolor="none")
        for i, v in enumerate(cnts[::-1]):
            ax.text(v, i, f" {v:,}", va="center", fontsize=8, fontweight="bold")
        ax.set_xlabel("Compounds rejected (first-failing filter)",
                      fontweight="bold")
        ax.set_title("Rejection counts by filter", fontweight="bold",
                     fontsize=10)
    else:
        ax.text(0.5, 0.5, "no per-filter rejection data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle("Filter funnel: natural-product library → BBB+ candidates",
                 fontweight="bold", fontsize=11, y=1.03)
    _save_figs(fig, out_path)
    plt.close(fig)


def _plot_pred_distribution(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.hist(df["ensemble_mean"].dropna(), bins=40,
            color=NATURE[0], edgecolor="black", linewidth=0.3)
    med = float(df["ensemble_mean"].median())
    ax.axvline(med, color="black", ls="--", lw=0.8,
               label=f"median = {med:.2f}")
    for thr in (6.0, 7.0):
        ax.axvline(thr, color=NATURE[1], ls=":", lw=0.8, alpha=0.7,
                   label=f"pChEMBL ≥ {thr}")
    ax.set_xlabel("Ensemble mean predicted pChEMBL", fontweight="bold")
    ax.set_ylabel("Compounds", fontweight="bold")
    ax.set_title("Predicted potency distribution", fontweight="bold")
    ax.legend(fontsize=7, frameon=False)
    _save_figs(fig, out_path)
    plt.close(fig)


def _plot_agreement(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    sc = ax.scatter(df["ensemble_mean"], df["ensemble_std"],
                    s=12, alpha=0.5, color=NATURE[0])
    ax.set_xlabel("Ensemble mean pred pChEMBL", fontweight="bold")
    ax.set_ylabel("Ensemble stdev across 5 splits", fontweight="bold")
    ax.set_title("Ensemble agreement", fontweight="bold")
    ax.grid(True, alpha=0.3, lw=0.4)
    _save_figs(fig, out_path)
    plt.close(fig)


def _plot_bbb_scatter(df: pd.DataFrame, phys: pd.DataFrame,
                      out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    sc = ax.scatter(phys["MW"], phys["TPSA"],
                    c=df["ensemble_mean"], cmap="viridis",
                    s=14, alpha=0.8, edgecolor="none")
    # Egan BBB envelope: TPSA <= 132, LogP <= 5.88 (informally TPSA <90 = strong)
    ax.axhline(90, color=NATURE[1], ls=":", lw=0.7,
               label="TPSA = 90 (BBB likely)")
    ax.set_xlabel("Molecular weight (Da)", fontweight="bold")
    ax.set_ylabel("TPSA (Å²)", fontweight="bold")
    ax.set_title("BBB filters coloured by predicted pChEMBL",
                 fontweight="bold")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.colorbar(sc, ax=ax, label="ensemble pChEMBL")
    _save_figs(fig, out_path)
    plt.close(fig)


def _plot_per_model_corr(df: pd.DataFrame, model_names: list[str],
                         out_path: Path) -> None:
    import matplotlib.pyplot as plt
    cols = [f"pred_{n}" for n in model_names]
    sub = df[cols].dropna()
    if sub.empty:
        return
    corr = sub.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=7, fontweight="bold")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Per-model Pearson r\n(natural-product predictions)",
                 fontweight="bold", fontsize=9)
    _save_figs(fig, out_path)
    plt.close(fig)


def _render_top_k_structures(df: pd.DataFrame, top_k: int,
                             out_dir: Path) -> list[Path]:
    """Render top-K compounds (by ensemble_mean) as PNG + SVG + combined grid."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    sub = (df.dropna(subset=["ensemble_mean"])
             .sort_values("ensemble_mean", ascending=False)
             .head(top_k).reset_index(drop=True))
    paths = []
    for i, row in sub.iterrows():
        smi = str(row["smiles"])
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        name = row.get("Name", "") if "Name" in row.index else ""
        name_str = f"  {name}" if isinstance(name, str) and name else ""
        legend = (f"#{i+1}  {row['id']}{name_str}\n"
                  f"pred={row['ensemble_mean']:.2f} ± {row['ensemble_std']:.2f}")
        d_svg = rdMolDraw2D.MolDraw2DSVG(460, 460)
        d_svg.drawOptions().legendFontSize = 16
        d_svg.DrawMolecule(m, legend=legend)
        d_svg.FinishDrawing()
        svg_path = out_dir / f"top_{i+1:02d}.svg"
        svg_path.write_text(d_svg.GetDrawingText(), encoding="utf-8")
        png_path = out_dir / f"top_{i+1:02d}.png"
        try:
            d_png = rdMolDraw2D.MolDraw2DCairo(460, 460)
            d_png.drawOptions().legendFontSize = 16
            d_png.DrawMolecule(m, legend=legend)
            d_png.FinishDrawing()
            d_png.WriteDrawingText(str(png_path))
        except Exception:
            pass
        paths.append(svg_path)

    # combined grid
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        ncols = 4
        n = len(sub)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.0, nrows * 3.3),
                                 squeeze=False)
        for i in range(nrows * ncols):
            ax = axes[i // ncols][i % ncols]
            png = out_dir / f"top_{i+1:02d}.png"
            if i < n and png.exists():
                ax.imshow(mpimg.imread(str(png)))
            ax.set_axis_off()
        fig.suptitle(f"Top {n} predicted PTP1B hits (BBB+ natural products)",
                     fontweight="bold", fontsize=12, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        _save_figs(fig, out_dir / "top_hits_grid")
        plt.close(fig)
    except Exception:
        pass
    return paths


def analyze(*, scored_csv: Path, funnel_json: Path, out_dir: Path,
            top_k: int, model_names: list[str],
            sdf_meta_csv: Path | None = None) -> None:
    _apply_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(scored_csv)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir = out_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # Join SDF metadata (compound names, targets, source type) when available.
    # The HY-Selleck Natural Product Library SDF exposes Catalog_NO + Name +
    # Target etc. as SDF properties; we extract them in Step 0 of the
    # orchestrator and join here.
    if sdf_meta_csv is not None and Path(sdf_meta_csv).exists():
        meta = pd.read_csv(sdf_meta_csv)
        if "Catalog_NO" in meta.columns:
            df = df.merge(meta, left_on="id", right_on="Catalog_NO", how="left")

    # Physchem (TPSA, MW) for BBB plot
    phys = _physchem(df["smiles"].astype(str).tolist())
    df_with_phys = pd.concat([df.reset_index(drop=True), phys], axis=1)
    df_with_phys.to_csv(tbl_dir / "scored_with_physchem.csv", index=False)

    # Plots
    _plot_funnel(funnel_json, fig_dir / "01_filter_funnel")
    _plot_pred_distribution(df, fig_dir / "02_pred_distribution")
    _plot_agreement(df, fig_dir / "03_ensemble_agreement")
    _plot_bbb_scatter(df, phys, fig_dir / "04_bbb_scatter")
    _plot_per_model_corr(df, model_names, fig_dir / "05_per_model_corr")
    _render_top_k_structures(df_with_phys, top_k, fig_dir / "top_structures")

    # Top-K table
    top = (df_with_phys.dropna(subset=["ensemble_mean"])
           .sort_values("ensemble_mean", ascending=False)
           .head(top_k))
    top.to_csv(tbl_dir / f"top_{top_k}_hits.csv", index=False)

    # Summary md
    n_total = len(df)
    n_above_6 = int((df["ensemble_mean"] >= 6.0).sum())
    n_above_7 = int((df["ensemble_mean"] >= 7.0).sum())
    median_pred = float(df["ensemble_mean"].median())
    mean_std = float(df["ensemble_std"].mean())
    (out_dir / "summary.md").write_text(
        "# PTP1B virtual screen on BBB+ natural products\n\n"
        "Ensemble across 5 PTP1B split-trained RADIANT models "
        "({models}).\n\n".format(models=", ".join(model_names))
        + "## Headline\n\n"
        + f"- BBB-filtered compounds scored: **{n_total}**\n"
        + f"- Predicted pChEMBL >= 6.0: **{n_above_6}** ({n_above_6/n_total:.1%})\n"
        + f"- Predicted pChEMBL >= 7.0: **{n_above_7}** ({n_above_7/n_total:.1%})\n"
        + f"- Median predicted pChEMBL: {median_pred:.2f}\n"
        + f"- Mean ensemble stdev across 5 splits: {mean_std:.3f}\n\n"
        + "## Figures\n\n"
        + "- figures/01_filter_funnel.{png,svg}\n"
        + "- figures/02_pred_distribution.{png,svg}\n"
        + "- figures/03_ensemble_agreement.{png,svg}\n"
        + "- figures/04_bbb_scatter.{png,svg}\n"
        + "- figures/05_per_model_corr.{png,svg}\n"
        + "- figures/top_structures/top_hits_grid.{png,svg}\n"
        + f"- figures/top_structures/top_01..{top_k:02d}.{{png,svg}}\n\n"
        + "## Tables\n\n"
        + "- tables/scored_with_physchem.csv\n"
        + f"- tables/top_{top_k}_hits.csv\n"
        + "\n## Note\n\nThis screening report is intentionally NOT stitched into\n"
        + "`PHASE_G_REPORT.md`. Virtual screening is a downstream application,\n"
        + "not a benchmark module.\n",
        encoding="utf-8",
    )
    logger.info("analysis written to %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=("score", "analyze", "all"),
                   default="all")
    p.add_argument("--filtered-smi", type=Path, required=True)
    p.add_argument("--funnel-json", type=Path, default=None)
    p.add_argument("--models", nargs="+", type=Path, default=[])
    p.add_argument("--model-names", nargs="+", default=[])
    p.add_argument("--vocab", type=Path,
                   default=Path("data/zinc20/smiles_vocab.json"))
    p.add_argument("--config", type=Path,
                   default=Path("configs/radiant_75m.json"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--task-name", default="pchembl")
    p.add_argument("--sdf-meta-csv", type=Path, default=None,
                   help="optional CSV with Catalog_NO + Name + Target etc., "
                        "joined into the scored table for human-readable hits")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    out_dir = Path(args.out_dir)
    scored_csv = out_dir / "02_scored.csv"

    if args.mode in ("score", "all"):
        if not args.models:
            raise SystemExit("--models is required for score mode")
        if not args.model_names:
            args.model_names = [m.parent.name for m in args.models]
        if len(args.model_names) != len(args.models):
            raise SystemExit("--model-names must match --models")
        cfg = _ScoreCfg(config_path=args.config, vocab_path=args.vocab,
                        device=args.device, batch_size=args.batch_size,
                        task_name=args.task_name)
        score(filtered_smi=args.filtered_smi,
              model_paths=args.models, model_names=args.model_names,
              cfg=cfg, out_csv=scored_csv)

    if args.mode in ("analyze", "all"):
        if not args.model_names:
            df_probe = pd.read_csv(scored_csv, nrows=1)
            args.model_names = [c[5:] for c in df_probe.columns
                                if c.startswith("pred_")]
        analyze(scored_csv=scored_csv,
                funnel_json=args.funnel_json or out_dir / "funnel.json",
                out_dir=out_dir, top_k=args.top_k,
                model_names=args.model_names,
                sdf_meta_csv=args.sdf_meta_csv)


if __name__ == "__main__":
    main()
