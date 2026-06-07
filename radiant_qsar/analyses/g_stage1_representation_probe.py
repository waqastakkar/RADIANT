"""Stage-1 chemistry representation probe.

The production path consumes a precomputed embeddings CSV so the probe remains
independent of GPU availability. Expected columns are ``smiles`` plus either
``scaffold``/``rgroup`` or enough SMILES information to derive them, followed by
embedding columns named ``emb_0``, ``emb_1``, ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import (
    AnalysisPaths,
    publication_style,
    save_figure,
    save_table,
    write_summary_md,
)
from radiant_qsar.pretrain.activity_pretrain import murcko_rgroup_smiles
from radiant_qsar.pretrain.corpus import _murcko_scaffold_smiles


def _nearest_enrichment(X: np.ndarray, labels: pd.Series, *, k: int) -> tuple[float, pd.DataFrame]:
    labels = labels.astype(str).to_numpy()
    valid = labels != ""
    X = X[valid]
    labels = labels[valid]
    if len(X) <= k or len(np.unique(labels)) < 2:
        return float("nan"), pd.DataFrame()
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    hits = labels[nn] == labels[:, None]
    rows = []
    for i in range(len(labels)):
        rows.append({
            "query_index": i,
            "label": labels[i],
            "neighbor_hit_rate": float(hits[i].mean()),
            "nearest_similarity": float(sim[i, nn[i, 0]]),
        })
    return float(hits.mean()), pd.DataFrame(rows)


def _linear_probe(X: np.ndarray, labels: pd.Series, *, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    labels = labels.astype(str)
    mask = labels != ""
    y = labels[mask].to_numpy()
    X = X[mask.to_numpy()]
    counts = pd.Series(y).value_counts()
    keep = counts[counts >= 2].index
    use = np.isin(y, keep)
    y = y[use]
    X = X[use]
    if len(np.unique(y)) < 2 or len(y) < 6:
        return float("nan")
    min_count = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_count))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy").mean())


def run(
    *,
    embeddings_csv: Path | str,
    out_dir: Path | str,
    k: int = 5,
    seed: int = 0,
) -> dict:
    publication_style()
    paths = AnalysisPaths(Path(out_dir), "g_stage1_representation_probe")
    df = pd.read_csv(embeddings_csv)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("embeddings_csv must contain columns named emb_0, emb_1, ...")
    if "scaffold" not in df.columns:
        df["scaffold"] = df["smiles"].map(_murcko_scaffold_smiles)
    if "rgroup" not in df.columns:
        df["rgroup"] = df["smiles"].map(murcko_rgroup_smiles)
    X = df[emb_cols].to_numpy(dtype=float)

    scaffold_nn, scaffold_neighbors = _nearest_enrichment(X, df["scaffold"], k=k)
    rgroup_nn, rgroup_neighbors = _nearest_enrichment(X, df["rgroup"], k=k)
    scaffold_probe = _linear_probe(X, df["scaffold"], seed=seed)
    rgroup_probe = _linear_probe(X, df["rgroup"], seed=seed)
    metrics = pd.DataFrame([{
        "n": int(len(df)),
        "embedding_dim": int(len(emb_cols)),
        "k": int(k),
        "scaffold_nn_enrichment": scaffold_nn,
        "rgroup_nn_enrichment": rgroup_nn,
        "scaffold_linear_probe_balanced_accuracy": scaffold_probe,
        "rgroup_linear_probe_balanced_accuracy": rgroup_probe,
        "n_scaffolds": int(df["scaffold"].nunique()),
        "n_rgroups": int(df["rgroup"].nunique()),
    }])

    neighbors = pd.concat(
        [
            scaffold_neighbors.assign(label_type="scaffold"),
            rgroup_neighbors.assign(label_type="rgroup"),
        ],
        ignore_index=True,
    )
    save_table(metrics, paths, "stage1_probe_metrics")
    save_table(neighbors, paths, "stage1_probe_neighbors")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.46, 2.8))
    vals = [
        metrics.loc[0, "scaffold_nn_enrichment"],
        metrics.loc[0, "rgroup_nn_enrichment"],
        metrics.loc[0, "scaffold_linear_probe_balanced_accuracy"],
        metrics.loc[0, "rgroup_linear_probe_balanced_accuracy"],
    ]
    labels = ["Scaffold NN", "R-group NN", "Scaffold probe", "R-group probe"]
    ax.bar(np.arange(len(vals)), vals)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Stage-1 chemistry probe")
    save_figure(fig, paths, "stage1_probe_scores")
    plt.close(fig)

    headline = (
        f"Scaffold NN enrichment={scaffold_nn:.3f}; "
        f"R-group NN enrichment={rgroup_nn:.3f}."
    )
    write_summary_md(
        paths,
        title="Stage-1 Representation Probe",
        claim="Chemistry-aware Stage-1 pretraining should cluster scaffold and R-group information in latent space.",
        headline=headline,
        details={"Embeddings": str(embeddings_csv), "Neighbors k": str(k)},
        tables_referenced=["stage1_probe_metrics.csv", "stage1_probe_neighbors.csv"],
        figures_referenced=["stage1_probe_scores.png"],
    )
    return {"paths": paths, "metrics": metrics, "neighbors": neighbors}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe Stage-1 chemistry embeddings")
    p.add_argument("--embeddings-csv", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
