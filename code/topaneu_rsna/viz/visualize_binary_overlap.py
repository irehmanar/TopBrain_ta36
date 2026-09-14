"""
Slide figures: Dataset304 (binary aneurysm segmentation) prediction vs.
ground truth overlap, for 5 cases where a real aneurysm exists and 5 cases
where none exists -- following the same style/conventions as
viz/visualize_pipeline.py (Agg backend, axial/coronal/sagittal 3-view
slicing, windowed grayscale background).

Predictions come from Dataset304's pooled 5-fold held-out `validation/`
outputs (the same source `assign_location_rule.py` reads), so every case
shown is scored on data that fold never trained on -- an honest look at
generalization, not the model grading its own homework.

Overlay colors:
  green   true positive  -- prediction and ground truth agree
  red     false negative -- ground truth has an aneurysm, prediction missed it
  yellow  false positive -- prediction says aneurysm, ground truth has none there

For cases with no real aneurysm (ground truth entirely empty), the slice is
centered on any false-positive voxels if there are some (to actually show
what the false alarm looks like), otherwise on the volume's own center.

Writes two PNGs: one 5-row grid for the aneurysm-present cases, one 5-row
grid for the aneurysm-absent cases, plus a small color-key legend.

Usage (see jobs/25_visualize_binary_overlap/70_visualize_binary_overlap.sbatch):
    python -m topaneu_rsna.viz.visualize_binary_overlap --n_positive 5 --n_negative 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio
from topaneu_rsna.seg.assign_location_rule import load_binary_pred_paths

COLOR_TP = (0.20, 0.75, 0.20)
COLOR_FN = (0.90, 0.15, 0.15)
COLOR_FP = (0.95, 0.85, 0.10)


def _windowed(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _center_of(mask: np.ndarray, shape) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.asarray(shape) // 2
    return pts.mean(0).round().astype(int)


def _overlay_rgba(pred_slice: np.ndarray, gt_slice: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    h, w = pred_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    tp = pred_slice & gt_slice
    fn = gt_slice & ~pred_slice
    fp = pred_slice & ~gt_slice
    for mask, color in ((tp, COLOR_TP), (fn, COLOR_FN), (fp, COLOR_FP)):
        rgba[mask, 0], rgba[mask, 1], rgba[mask, 2], rgba[mask, 3] = (*color, alpha)
    return rgba


def _row(axes_row, img: np.ndarray, pred: np.ndarray, gt: np.ndarray, title_prefix: str):
    center_mask = gt if gt.any() else pred
    z, y, x = _center_of(center_mask, img.shape)
    win = _windowed(img)
    views = (("axial", win[z, :, :], pred[z, :, :], gt[z, :, :]),
             ("coronal", win[:, y, :], pred[:, y, :], gt[:, y, :]),
             ("sagittal", win[:, :, x], pred[:, :, x], gt[:, :, x]))
    for ax, (name, gray, p, g) in zip(axes_row, views):
        ax.imshow(gray, cmap="gray", origin="lower", interpolation="nearest")
        ax.imshow(_overlay_rgba(p, g), origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=9)

    gt_vox, pred_vox = int(gt.sum()), int(pred.sum())
    inter = int((pred & gt).sum())
    if gt_vox > 0:
        dice = 2 * inter / (gt_vox + pred_vox) if (gt_vox + pred_vox) else 0.0
        stat = f"Dice={dice:.2f}, gt={gt_vox}vox, pred={pred_vox}vox"
    else:
        stat = f"no real aneurysm; {pred_vox} FP voxel(s)" if pred_vox else "no real aneurysm; clean"
    axes_row[0].set_ylabel(f"{title_prefix}\n{stat}", fontsize=8)


def make_grid(cases: list[str], bin_paths: dict, out_path: Path, suptitle: str) -> Path | None:
    rows = []
    for case in cases:
        bp = bin_paths.get(case)
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if bp is None or not gt_p.exists():
            print(f"[skip] {case}: missing prediction or ground truth")
            continue
        pred, meta = uio.read(bp)
        gt, _ = uio.read(gt_p)
        img, _ = uio.read(C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}")
        rows.append((case, img, pred > 0, gt > 0))

    if not rows:
        return None

    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for axes_row, (case, img, pred, gt) in zip(axes, rows):
        _row(axes_row, img, pred, gt, case)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def make_legend(out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4, 1.2))
    ax.set_xlim(0, 3); ax.set_ylim(0, 1); ax.axis("off")
    for i, (color, label) in enumerate((
            (COLOR_TP, "true positive"), (COLOR_FN, "false negative"),
            (COLOR_FP, "false positive"))):
        ax.add_patch(plt.Rectangle((i, 0.3), 0.25, 0.4, color=color))
        ax.text(i + 0.32, 0.5, label, fontsize=9, va="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary_dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_positive", type=int, default=5)
    ap.add_argument("--n_negative", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path,
                    default=C.EXP_ROOT / "figures" / "binary_overlap_viz")
    a = ap.parse_args()

    bin_paths = load_binary_pred_paths(a.binary_dataset, a.trainer, a.plans, a.folds)
    print(f"{len(bin_paths)} held-out binary predictions "
         f"(Dataset{a.binary_dataset}, folds {a.folds})")

    positive_cases, negative_cases = [], []
    for case in sorted(bin_paths):
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not gt_p.exists():
            continue
        gt, _ = uio.read(gt_p)
        (positive_cases if gt.any() else negative_cases).append(case)

    print(f"{len(positive_cases)} cases with a real aneurysm, "
         f"{len(negative_cases)} cases with none")

    rng = np.random.default_rng(a.seed)
    chosen_pos = sorted(rng.choice(positive_cases, size=min(a.n_positive, len(positive_cases)),
                                   replace=False).tolist())
    chosen_neg = sorted(rng.choice(negative_cases, size=min(a.n_negative, len(negative_cases)),
                                   replace=False).tolist())
    print(f"Positive cases chosen: {chosen_pos}")
    print(f"Negative cases chosen: {chosen_neg}")

    p1 = make_grid(chosen_pos, bin_paths, a.out_dir / "binary_overlap_positive.png",
                   "Dataset304: prediction vs. ground truth -- aneurysm PRESENT")
    p2 = make_grid(chosen_neg, bin_paths, a.out_dir / "binary_overlap_negative.png",
                   "Dataset304: prediction vs. ground truth -- aneurysm ABSENT")
    p3 = make_legend(a.out_dir / "binary_overlap_legend.png")

    for p in (p1, p2, p3):
        if p:
            print(f"  -> {p}")


if __name__ == "__main__":
    main()
