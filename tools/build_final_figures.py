from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final figures"
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


FIGURES = [
    {
        "file": "figure_1_model_training_story.svg",
        "title": "Figure 1. RADIANT model and training program",
        "subtitle": "Three-stage chemistry-to-activity learning with recurrent-depth training diagnostics.",
        "panels": [
            {
                "label": "A",
                "caption": "Three-stage pretraining and fine-tuning overview",
                "source": "runs/phase_g/g_pretrain_curves/figures/g_pretrain_three_stage_overview.svg",
                "x": 40,
                "y": 92,
                "w": 1040,
                "h": 390,
            },
            {
                "label": "B",
                "caption": "ZINC20/ChEMBL self-supervised chemistry pretraining",
                "source": "runs/phase_g/g_pretrain_curves/figures/g_pretrain_zinc20_loss.svg",
                "x": 40,
                "y": 540,
                "w": 505,
                "h": 320,
            },
            {
                "label": "C",
                "caption": "Panel fine-tuning convergence across QSAR targets",
                "source": "runs/phase_g/g_training_curves/figures/g_training_curves_aggregate.svg",
                "x": 585,
                "y": 540,
                "w": 495,
                "h": 320,
            },
        ],
    },
    {
        "file": "figure_2_predictive_evidence.svg",
        "title": "Figure 2. Predictive accuracy and robustness across the QSAR panel",
        "subtitle": "Benchmarks, split robustness, rank structure, and statistical evidence across the 20-target panel.",
        "panels": [
            {
                "label": "A",
                "caption": "MAE comparison against molecular baselines",
                "source": "runs/phase_g/g0_validation_metrics/figures/g0_model_comparison_mae.svg",
                "x": 38,
                "y": 96,
                "w": 515,
                "h": 300,
            },
            {
                "label": "B",
                "caption": "RADIANT parity across targets",
                "source": "runs/phase_g/g0_validation_metrics/figures/g0_parity_grid_radiant.svg",
                "x": 590,
                "y": 96,
                "w": 520,
                "h": 300,
            },
            {
                "label": "C",
                "caption": "Hard-split win rate versus reference",
                "source": "runs/phase_g/g_hard_splits/figures/g_hard_splits_winrate_vs_reference.svg",
                "x": 38,
                "y": 455,
                "w": 350,
                "h": 270,
            },
            {
                "label": "D",
                "caption": "MAE rank heatmap across split types",
                "source": "runs/phase_g/g_ranks/figures/g_ranks_split_heatmap_mae.svg",
                "x": 415,
                "y": 455,
                "w": 350,
                "h": 270,
            },
            {
                "label": "E",
                "caption": "Nemenyi post-hoc ranking for MAE",
                "source": "runs/phase_g/g_stat_tests/figures/g_stat_tests_nemenyi_mae.svg",
                "x": 790,
                "y": 455,
                "w": 320,
                "h": 270,
            },
        ],
    },
    {
        "file": "figure_3_adaptive_depth_interpretability_screening.svg",
        "title": "Figure 3. Adaptive inference and SAR sensitivity",
        "subtitle": "Compute-aware loop depth, halting behavior, applicability domain, and chemically grounded SAR delta preservation.",
        "panels": [
            {
                "label": "A",
                "caption": "Loop depth improves difficult-molecule MAE",
                "source": "runs/phase_g/g4_test_time_loop_sweep/figures/g4_mae_vs_nloops.svg",
                "x": 38,
                "y": 96,
                "w": 345,
                "h": 260,
            },
            {
                "label": "B",
                "caption": "Halting-depth ablation",
                "source": "runs/phase_g/g_halting_toggle/figures/g_halting_toggle_mae_vs_k.svg",
                "x": 410,
                "y": 96,
                "w": 345,
                "h": 260,
            },
            {
                "label": "C",
                "caption": "Applicability domain distance versus error",
                "source": "runs/phase_g/g_applicability_domain/figures/g_ad_mae_vs_distance.svg",
                "x": 782,
                "y": 96,
                "w": 345,
                "h": 260,
            },
            {
                "label": "D",
                "caption": "R-group SAR delta preservation",
                "source": "runs/phase_g/g_rgroup_sar/figures/rgroup_sar_delta.svg",
                "x": 125,
                "y": 455,
                "w": 460,
                "h": 315,
            },
            {
                "label": "E",
                "caption": "Activity-cliff delta preservation",
                "source": "runs/phase_g/g_activity_cliff_sar/figures/activity_cliff_delta.svg",
                "x": 625,
                "y": 455,
                "w": 460,
                "h": 315,
            },
        ],
    },
]


def read_svg(source: Path) -> tuple[ET.Element, tuple[float, float, float, float]]:
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"<!DOCTYPE[^>]*>\s*", "", text)
    root = ET.fromstring(text)
    view_box = root.get("viewBox")
    if view_box:
        parts = [float(p) for p in re.split(r"[\s,]+", view_box.strip())]
        if len(parts) == 4:
            return root, (parts[0], parts[1], parts[2], parts[3])
    width = _num(root.get("width", "0"))
    height = _num(root.get("height", "0"))
    return root, (0.0, 0.0, width, height)


def _num(value: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", value)
    return float(match.group(0)) if match else 0.0


def add_text(parent: ET.Element, text: str, x: float, y: float, size: int, weight: str = "700") -> None:
    node = ET.SubElement(parent, f"{{{SVG_NS}}}text")
    node.set("x", str(x))
    node.set("y", str(y))
    node.set("font-family", "Times New Roman, Times, serif")
    node.set("font-size", str(size))
    node.set("font-weight", weight)
    node.set("fill", "#111827")
    node.text = text


def add_panel(parent: ET.Element, panel: dict[str, object], figure_id: str) -> None:
    x = float(panel["x"])
    y = float(panel["y"])
    w = float(panel["w"])
    h = float(panel["h"])
    src = ROOT / str(panel["source"])
    child, vb = read_svg(src)
    vx, vy, vw, vh = vb
    scale = min(w / vw, h / vh)
    tx = x + (w - vw * scale) / 2 - vx * scale
    ty = y + 22 + (h - vh * scale) / 2 - vy * scale

    add_text(parent, str(panel["label"]), x, y + 3, 28)
    add_text(parent, str(panel["caption"]), x + 36, y + 3, 16)

    frame = ET.SubElement(parent, f"{{{SVG_NS}}}rect")
    frame.set("x", str(x))
    frame.set("y", str(y + 20))
    frame.set("width", str(w))
    frame.set("height", str(h))
    frame.set("rx", "4")
    frame.set("fill", "#ffffff")
    frame.set("stroke", "#d1d5db")
    frame.set("stroke-width", "1")

    group = ET.SubElement(parent, f"{{{SVG_NS}}}g")
    group.set("transform", f"translate({tx:.4f} {ty:.4f}) scale({scale:.6f})")
    group.set("data-source", str(panel["source"]))
    group.set("id", f"{figure_id}_panel_{panel['label']}")
    child = deepcopy(child)
    prefix_svg_ids(child, f"{figure_id}_{panel['label']}_")
    for child_node in list(child):
        tag = child_node.tag.split("}")[-1]
        if tag == "metadata":
            continue
        if tag == "defs":
            parent.append(child_node)
        else:
            group.append(child_node)


def prefix_svg_ids(node: ET.Element, prefix: str) -> None:
    id_map: dict[str, str] = {}
    for elem in node.iter():
        old_id = elem.get("id")
        if old_id:
            new_id = f"{prefix}{old_id}"
            id_map[old_id] = new_id
            elem.set("id", new_id)

    if not id_map:
        return

    url_pattern = re.compile(r"url\(#([^)]+)\)")
    href_keys = {"href", f"{{{XLINK_NS}}}href"}
    for elem in node.iter():
        for key, value in list(elem.attrib.items()):
            if not value:
                continue
            if key in href_keys and value.startswith("#"):
                target = value[1:]
                if target in id_map:
                    elem.set(key, f"#{id_map[target]}")
            if "url(#" in value:
                elem.set(key, url_pattern.sub(lambda m: f"url(#{id_map.get(m.group(1), m.group(1))})", value))


def build_figure(spec: dict[str, object]) -> None:
    figure_id = str(spec["file"]).replace(".svg", "")
    root = ET.Element(f"{{{SVG_NS}}}svg")
    root.set("width", "11.5in")
    root.set("height", "9.8in")
    root.set("viewBox", "0 0 1150 980")
    root.set("version", "1.1")

    bg = ET.SubElement(root, f"{{{SVG_NS}}}rect")
    bg.set("width", "1150")
    bg.set("height", "980")
    bg.set("fill", "#ffffff")

    add_text(root, str(spec["title"]), 38, 42, 26)
    add_text(root, str(spec["subtitle"]), 38, 70, 15, "600")

    for panel in spec["panels"]:
        add_panel(root, panel, figure_id)

    tree = ET.ElementTree(root)
    out_path = OUT / str(spec["file"])
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    manifest = []
    for spec in FIGURES:
        build_figure(spec)
        manifest.append(
            {
                "figure": spec["file"],
                "title": spec["title"],
                "sources": [panel["source"] for panel in spec["panels"]],
            }
        )
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
