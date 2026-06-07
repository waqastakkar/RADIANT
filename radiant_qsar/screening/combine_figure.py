"""Compose all screening panels into a single manuscript figure.

Reads the per-panel outputs produced by ``screen_ensemble`` and assembles
one multi-panel composite suitable for a manuscript figure (typically
Figure 8 or a screening-application figure). Outputs both PNG and SVG.

Layout (6 panels, 7.2 in two-column wide x ~10 in tall):

    +---------------------------------------------------+
    | (a) Filter funnel summary + per-filter rejections |
    +---------------------------+-----------------------+
    | (b) Predicted pChEMBL    | (c) BBB scatter        |
    |     distribution         |     (MW vs TPSA,      |
    |                          |      coloured by pred) |
    +---------------------------+-----------------------+
    | (d) Ensemble agreement   | (e) Per-model Pearson r|
    |     (mean vs stdev)      |     (5x5 heatmap)      |
    +---------------------------+-----------------------+
    | (f) Top 8 hits -- RDKit structure thumbnails      |
    |     with Name + Catalog + ensemble pred +/- std    |
    +---------------------------------------------------+

Inputs (all already on disk after ``screen_ensemble`` runs):
    runs/screening/<NAME>/funnel.json
    runs/screening/<NAME>/02_scored.csv
    runs/screening/<NAME>/tables/scored_with_physchem.csv
    runs/screening/<NAME>/tables/top_20_hits.csv
    runs/screening/<NAME>/figures/top_structures/top_NN.png

Outputs:
    runs/screening/<NAME>/figures/screening_composite.{png,svg}

CLI:
    python -m radiant_qsar.screening.combine_figure \\
        --screen-dir runs/screening/PTP1B_NP_BBB \\
        --title "PTP1B virtual screen on BBB+ natural products" \\
        --n-top 8
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _apply_style() -> None:
    from radiant_qsar.analyses.common import publication_style
    publication_style()


NATURE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442",
         "#56B4E9", "#E69F00", "#000000"]


def _annotate(ax, letter: str) -> None:
    ax.text(-0.10, 1.06, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="right")


def _panel_funnel(ax, funnel: dict, profile: str) -> None:
    """Panel (a): pass-rate + horizontal bar of per-filter rejections."""
    n_in = int(funnel.get("n_input", 0))
    n_pass = int(funnel.get("n_passed", 0))
    n_fail = int(funnel.get("n_failed", n_in - n_pass))
    rej = funnel.get("rejects_by_filter", {})
    pct = (n_pass / n_in * 100.0) if n_in else 0.0

    if rej:
        items = sorted(rej.items(), key=lambda kv: -int(kv[1]))
        names = [k for k, _ in items]
        cnts = [int(v) for _, v in items]
        y = np.arange(len(names))[::-1]
        ax.barh(y, cnts, color=NATURE[1], edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7, fontweight="bold")
        for yi, v in zip(y, cnts):
            ax.text(v, yi, f" {v:,}", va="center",
                    fontsize=6.5, fontweight="bold")
        ax.set_xlabel("Compounds rejected (first-failing filter)",
                      fontweight="bold")
    ax.set_title(
        f"input {n_in:,} -> BBB+ {n_pass:,}  ({pct:.1f}%)\n"
        f"profile: {profile}",
        fontweight="bold", fontsize=9)


def _panel_distribution(ax, df: pd.DataFrame) -> None:
    """Panel (b): histogram of ensemble mean predicted pChEMBL."""
    vals = df["ensemble_mean"].dropna()
    ax.hist(vals, bins=40, color=NATURE[0],
            edgecolor="black", linewidth=0.3)
    med = float(vals.median())
    ax.axvline(med, color="black", ls="--", lw=0.8,
               label=f"median = {med:.2f}")
    for thr in (6.0, 7.0):
        ax.axvline(thr, color=NATURE[1], ls=":", lw=0.8, alpha=0.7,
                   label=f"pChEMBL >= {thr}")
    ax.set_xlabel("Ensemble mean predicted pChEMBL", fontweight="bold")
    ax.set_ylabel("Compounds", fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_title("Predicted potency", fontweight="bold", fontsize=9)


def _panel_bbb(ax, df: pd.DataFrame) -> None:
    """Panel (c): MW vs TPSA coloured by predicted pChEMBL."""
    if not {"MW", "TPSA"}.issubset(df.columns):
        ax.text(0.5, 0.5, "missing physchem", ha="center", va="center",
                transform=ax.transAxes)
        return
    sc = ax.scatter(df["MW"], df["TPSA"], c=df["ensemble_mean"],
                    cmap="viridis", s=10, alpha=0.75, edgecolor="none")
    ax.axhline(90, color=NATURE[1], ls=":", lw=0.7,
               label="TPSA = 90")
    ax.set_xlabel("MW (Da)", fontweight="bold")
    ax.set_ylabel("TPSA (A^2)", fontweight="bold")
    cb = ax.figure.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("pred pChEMBL", fontweight="bold", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.set_title("BBB envelope (coloured by predicted pChEMBL)",
                 fontweight="bold", fontsize=9)


def _panel_agreement(ax, df: pd.DataFrame) -> None:
    """Panel (d): mean vs stdev across 5 split models."""
    ax.scatter(df["ensemble_mean"], df["ensemble_std"],
               s=10, alpha=0.5, color=NATURE[0], edgecolor="none")
    ax.set_xlabel("Ensemble mean pred pChEMBL", fontweight="bold")
    ax.set_ylabel("Ensemble stdev across 5 splits", fontweight="bold")
    ax.set_title("Ensemble agreement", fontweight="bold", fontsize=9)
    ax.grid(True, alpha=0.3, lw=0.4)


def _panel_corr(ax, df: pd.DataFrame, model_names: list[str]) -> None:
    """Panel (e): pairwise Pearson r between the 5 split models."""
    cols = [f"pred_{n}" for n in model_names if f"pred_{n}" in df.columns]
    if len(cols) < 2:
        ax.text(0.5, 0.5, "no per-model columns", ha="center", va="center",
                transform=ax.transAxes)
        return
    sub = df[cols].dropna()
    corr = sub.corr(method="pearson")
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    labels = [c.replace("pred_", "") for c in cols]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7, fontweight="bold")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center",
                    va="center", fontsize=7, fontweight="bold")
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Per-model Pearson r", fontweight="bold", fontsize=9)


def _panel_top_hits(axes_row, top_df: pd.DataFrame, struct_dir: Path,
                    n_top: int) -> None:
    """Panel (f): row of n_top RDKit structure thumbnails with labels."""
    import matplotlib.image as mpimg

    for i, ax in enumerate(axes_row):
        ax.set_axis_off()
        if i >= len(top_df) or i >= n_top:
            continue
        row = top_df.iloc[i]
        png = struct_dir / f"top_{i+1:02d}.png"
        if png.exists():
            img = mpimg.imread(str(png))
            ax.imshow(img)
        name = row.get("Name", "") if "Name" in row.index else ""
        cat = row.get("Catalog_NO", row.get("id", ""))
        pred = row.get("ensemble_mean", float("nan"))
        std = row.get("ensemble_std", float("nan"))
        title = (f"#{i+1} {name}" if isinstance(name, str) and name
                 else f"#{i+1} {cat}")
        if len(title) > 28:
            title = title[:25] + "..."
        ax.set_title(
            f"{title}\n{cat}  pred={pred:.2f}+/-{std:.2f}",
            fontsize=7, fontweight="bold", pad=3)


def compose(*, screen_dir: Path, title: str | None = None,
            n_top: int = 8, profile: str = "cns_brain_penetrant") -> Path:
    _apply_style()
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    screen_dir = Path(screen_dir)
    funnel = json.loads((screen_dir / "funnel.json").read_text(encoding="utf-8"))
    scored = pd.read_csv(screen_dir / "tables" / "scored_with_physchem.csv")
    top_df = pd.read_csv(screen_dir / "tables" / "top_20_hits.csv")
    struct_dir = screen_dir / "figures" / "top_structures"

    # Derive per-model names from columns
    model_names = [c.replace("pred_", "") for c in scored.columns
                   if c.startswith("pred_")]

    fig = plt.figure(figsize=(7.2, 10.2))
    gs = gridspec.GridSpec(
        4, max(n_top, 4),
        height_ratios=[1.7, 1.5, 1.5, 1.4],
        hspace=0.95, wspace=0.55,
        left=0.10, right=0.97, top=0.93, bottom=0.05,
    )

    # Row 1 (full width): funnel
    ax_a = fig.add_subplot(gs[0, :])
    _annotate(ax_a, "a")
    _panel_funnel(ax_a, funnel, funnel.get("profile", profile))

    # Row 2: distribution | BBB scatter
    ax_b = fig.add_subplot(gs[1, :max(n_top, 4) // 2])
    _annotate(ax_b, "b")
    _panel_distribution(ax_b, scored)

    ax_c = fig.add_subplot(gs[1, max(n_top, 4) // 2:])
    _annotate(ax_c, "c")
    _panel_bbb(ax_c, scored)

    # Row 3: agreement | correlation
    ax_d = fig.add_subplot(gs[2, :max(n_top, 4) // 2])
    _annotate(ax_d, "d")
    _panel_agreement(ax_d, scored)

    ax_e = fig.add_subplot(gs[2, max(n_top, 4) // 2:])
    _annotate(ax_e, "e")
    _panel_corr(ax_e, scored, model_names)

    # Row 4: top hits row (1 axis per hit, spanning full width)
    hit_axes = [fig.add_subplot(gs[3, i]) for i in range(min(n_top, max(n_top, 4)))]
    _annotate(hit_axes[0], "f")
    _panel_top_hits(hit_axes, top_df, struct_dir, n_top)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.985)

    fig_dir = screen_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / "screening_composite.png"
    out_svg = fig_dir / "screening_composite.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s and %s", out_png, out_svg)
    return out_png


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--screen-dir", required=True, type=Path)
    p.add_argument("--title", default=None)
    p.add_argument("--n-top", type=int, default=8)
    p.add_argument("--profile", default="cns_brain_penetrant")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    compose(screen_dir=args.screen_dir, title=args.title,
            n_top=args.n_top, profile=args.profile)


if __name__ == "__main__":
    main()
