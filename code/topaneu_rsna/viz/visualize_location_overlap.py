"""
Slide figures: Task 2 final location assignment (52-class), ground truth vs.
the RULE (Experiment 1), the CLASSIFIER (Experiment 2), and the targeted-
override HYBRID (Experiment 3, job 65) -- same 5 patients, same slice per
patient, so the four figures line up panel-by-panel. Same visual conventions
as viz/visualize_vessel_overlap.py (Agg backend, axial/coronal/sagittal
slicing, categorical color per class, stable across every figure).

Reuses job 65's exact join+override recipe (build_hybrid_assignment.py's
load_rows/join_tables, plus the same --override_resolved_by
arc_low_sample_fallback / --override_proba_threshold 0.5 the real pipeline
used) so what's drawn here is the actual job 65 hybrid, not a re-derived
approximation. Binary detections are Dataset304's real (non-oracle) pooled
5-fold validation predictions, matching every non-oracle experiment so far --
so, unlike the rule/classifier themselves, the RULE never says "background"
but a hallucinated (false-positive) instance still gets painted with
whatever location the rule assigned it; that's a real, instructive failure
mode to show on a slide, not a bug.

Per-case captions report instance-level correctness (prediction ==
true_class) restricted to real lesions (true_class != "background"), taken
straight from the joined table -- not a recomputed voxel Dice -- since that
is this whole project's own headline unit (pooled per-component accuracy).

Prerequisites (already produced by prior jobs, nothing new to run):
  job 62  logs/task2_rule_instances_final_idx.csv
  job 61  logs/task2_classifier_predictions_weighted.csv

Usage (see jobs/28_visualize_location_overlap/<job>.sbatch):
    python -m topaneu_rsna.viz.visualize_location_overlap --n_cases 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio
from topaneu_rsna.seg.assign_location_rule import load_binary_pred_paths
from topaneu_rsna.seg.build_hybrid_assignment import load_rows, join_tables

APPROACHES = [
    ("Rule (Experiment 1)", "rule_prediction", "rule"),
    ("Classifier (Experiment 2)", "classifier_prediction", "classifier"),
    ("Hybrid, targeted override (Experiment 3 / job 65)", "hybrid_prediction", "hybrid"),
]
OVERRIDE_RESOLVED_BY = ["arc_low_sample_fallback"]   # exactly job 65's recipe
OVERRIDE_PROBA_THRESHOLD = 0.5


def _windowed(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _categorical_colors(n: int) -> np.ndarray:
    """One stable RGB color per class (index 0 = label value 1), reused
    identically across the GT figure and all 3 approach figures."""
    base = np.vstack([plt.cm.tab20(np.linspace(0, 1, 20))[:, :3],
                      plt.cm.tab20b(np.linspace(0, 1, 20))[:, :3],
                      plt.cm.tab20c(np.linspace(0, 1, 20))[:, :3]])
    reps = int(np.ceil(n / len(base)))
    return np.tile(base, (reps, 1))[:n]


def _label_rgba(label_slice: np.ndarray, colors: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    h, w = label_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    fg = label_slice > 0
    idx = label_slice[fg].astype(int) - 1
    rgba[fg, :3] = colors[idx]
    rgba[fg, 3] = alpha
    return rgba


def _center_of(mask: np.ndarray, shape) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.asarray(shape) // 2
    return pts.mean(0).round().astype(int)


def add_hybrid_column(joined: list[dict]) -> None:
    for j in joined:
        pred = j["rule_prediction"]
        if (j["resolved_by"] in OVERRIDE_RESOLVED_BY
                and j["classifier_proba"] >= OVERRIDE_PROBA_THRESHOLD):
            pred = j["classifier_prediction"]
        j["hybrid_prediction"] = pred


def repaint(case: str, lab: np.ndarray, instances: list[tuple], loc_value: dict) -> np.ndarray:
    final = np.zeros(lab.shape, dtype=np.uint16)
    for instance_idx, pred in instances:
        if not pred or pred == "background":
            continue
        final[lab == int(instance_idx)] = loc_value[pred]
    return final


def _row(axes_row, img: np.ndarray, label: np.ndarray, colors: np.ndarray,
        center, title_prefix: str, stat: str | None):
    z, y, x = center
    win = _windowed(img)
    views = (("axial", win[z, :, :], label[z, :, :]),
             ("coronal", win[:, y, :], label[:, y, :]),
             ("sagittal", win[:, :, x], label[:, :, x]))
    for ax, (name, gray, lab_slice) in zip(axes_row, views):
        ax.imshow(gray, cmap="gray", origin="lower", interpolation="nearest")
        ax.imshow(_label_rgba(lab_slice, colors), origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=9)
    ylabel = title_prefix if stat is None else f"{title_prefix}\n{stat}"
    axes_row[0].set_ylabel(ylabel, fontsize=8)


def make_grid(cases: list[str], rows_data: dict, colors: np.ndarray, out_path: Path,
             suptitle: str) -> Path | None:
    rows = [rows_data[c] for c in cases if c in rows_data]
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for axes_row, (case, img, label, center, stat) in zip(axes, rows):
        _row(axes_row, img, label, colors, center, case, stat)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def make_legend(names: list[str], colors: np.ndarray, out_path: Path) -> Path:
    n = len(names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, ax = plt.subplots(figsize=(4.5 * ncols, 0.3 * nrows))
    ax.set_xlim(0, ncols); ax.set_ylim(0, nrows); ax.axis("off")
    for i, name in enumerate(names):
        col, row = i % ncols, i // ncols
        ax.add_patch(plt.Rectangle((col, nrows - row - 1), 0.2, 0.7, color=colors[i]))
        ax.text(col + 0.28, nrows - row - 0.65, name, fontsize=6.5, va="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule_instances_csv", type=Path,
                    default=C.LOG_ROOT / "task2_rule_instances_final_idx.csv")
    ap.add_argument("--classifier_predictions_csv", type=Path,
                    default=C.LOG_ROOT / "task2_classifier_predictions_weighted.csv")
    ap.add_argument("--binary_dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_cases", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path,
                    default=C.EXP_ROOT / "figures" / "location_overlap_viz")
    a = ap.parse_args()

    spec = C.load_labels()
    loc_value = {loc: i + 1 for i, loc in enumerate(spec.locations)}
    colors = _categorical_colors(spec.n_loc)

    rule_rows = load_rows(a.rule_instances_csv)
    clf_rows = load_rows(a.classifier_predictions_csv)
    joined = join_tables(rule_rows, clf_rows)
    add_hybrid_column(joined)
    print(f"{len(joined)} instances joined (rule assigned + has a classifier counterpart)")

    by_case = {}
    for j in joined:
        by_case.setdefault(j["case"], []).append(j)

    bin_paths = load_binary_pred_paths(a.binary_dataset, a.trainer, a.plans, a.folds)
    eligible = sorted(c for c in by_case
                      if c in bin_paths
                      and (C.LOCATION_MASKS / f"{c}{C.LABEL_SUFFIX}").exists()
                      and any(j["true_class"] != "background" for j in by_case[c]))
    print(f"{len(eligible)} cases have a real lesion, a binary prediction, and GT")

    rng = np.random.default_rng(a.seed)
    chosen = sorted(rng.choice(eligible, size=min(a.n_cases, len(eligible)),
                               replace=False).tolist())
    print(f"Cases chosen: {chosen}")

    gt_rows, rows_by_tag = {}, {tag: {} for _, _, tag in APPROACHES}
    for case in chosen:
        binmask, meta = uio.read(bin_paths[case])
        lab, _ = ndimage.label(binmask > 0)
        gt, _ = uio.read(C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}")
        img, _ = uio.read(C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}")
        center = _center_of(gt > 0, gt.shape)

        gt_rows[case] = (case, img, gt, center, None)

        real = [j for j in by_case[case] if j["true_class"] != "background"]
        for label_txt, field, tag in APPROACHES:
            instances = [(j["instance_idx"], j[field]) for j in by_case[case]]
            painted = repaint(case, lab, instances, loc_value)
            n_correct = sum(1 for j in real if j[field] == j["true_class"])
            stat = f"{n_correct}/{len(real)} real lesion(s) correctly located"
            rows_by_tag[tag][case] = (case, img, painted, center, stat)

    p_gt = make_grid(chosen, gt_rows, colors,
                     a.out_dir / "location_overlap_gt_reference.png",
                     "Task 2 ground truth -- 52-class aneurysm locations")
    print(f"  -> {p_gt}")

    for label_txt, field, tag in APPROACHES:
        p = make_grid(chosen, rows_by_tag[tag], colors,
                      a.out_dir / f"location_overlap_{tag}.png",
                      f"Task 2 location assignment -- {label_txt}")
        print(f"  -> {p}")

    p_leg = make_legend(spec.locations, colors, a.out_dir / "location_overlap_legend.png")
    print(f"  -> {p_leg}")


if __name__ == "__main__":
    main()
