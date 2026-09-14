"""
Slide figures: Dataset302 (36-class vessel segmentation) prediction overlays,
same 5 patients, across all 3 trained trainer variants -- plus one ground-truth
reference figure so the professor can compare each model's tree against the
truth and against each other at a glance. Follows the same visual conventions
as viz/visualize_binary_overlap.py (Agg backend, axial/coronal/sagittal
slicing, windowed grayscale background) but with a categorical (36-color)
overlay instead of TP/FN/FP, since this is a multi-class label map, not a
binary mask.

Caveat: all three trainers here were only ever trained fold_all (no
cross-validation folds -- see config.SEG_FOLD), so `fold_all/validation` is
nnU-Net's own end-of-training inference pass over the training cases, NOT
held-out data the way Dataset304's pooled 5-fold `validation/` is in
visualize_binary_overlap.py. Fine for a qualitative "what does the predicted
vessel tree look like" slide figure; don't read the per-case Dice caption on
these figures as a generalization number the way Dataset304's binary overlap
figure's Dice can be read.

The same 5 cases and the same slice (ground truth's own centroid, so every
panel across every figure lines up) are reused throughout, so the four
figures this produces are directly comparable panel-by-panel:
  vessel_overlap_gt_reference.png             -- ground truth alone
  vessel_overlap_<model tag>.png              -- x3, one per trainer
  vessel_overlap_legend.png                   -- 36-class color key

Usage (see jobs/27_visualize_vessel_overlap/<job>.sbatch):
    python -m topaneu_rsna.viz.visualize_vessel_overlap --n_cases 5
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

# label, trainer dir name, short tag for filenames -- see config.py's
# TRAINER_M2/M3 for Models 2/3; job 09's trainer has no config.py constant of
# its own since it's a one-off loss-function ablation, not part of the main
# pipeline's cascade, so it's named directly here.
MODELS = [
    ("Model 2 (Tversky b=0.7, srec x1)", C.TRAINER_M2, "model2_tversky_srec1"),
    ("Model 3 (Tversky b=0.7, srec x3)", C.TRAINER_M3, "model3_tversky_srec3"),
    ("Job09 (plain Dice, srec x1)", "RSNA2025Trainer_moreDAv6_1_SkeletonRecall", "job09_dice_srec1"),
]


def _windowed(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _vessel_colors(n_vessel: int) -> np.ndarray:
    """One stable RGB color per vessel class (index 0 = label value 1), reused
    identically across the GT figure, all 3 prediction figures, and the legend."""
    base = np.vstack([plt.cm.tab20(np.linspace(0, 1, 20))[:, :3],
                      plt.cm.tab20b(np.linspace(0, 1, 20))[:, :3]])
    reps = int(np.ceil(n_vessel / len(base)))
    return np.tile(base, (reps, 1))[:n_vessel]


def _label_rgba(label_slice: np.ndarray, colors: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    h, w = label_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    fg = label_slice > 0
    idx = label_slice[fg].astype(int) - 1
    rgba[fg, :3] = colors[idx]
    rgba[fg, 3] = alpha
    return rgba


def load_vessel_pred_paths(trainer: str) -> dict:
    val_dir = C.seg_model_dir(C.DS_VESSEL, trainer) / f"fold_{C.SEG_FOLD}" / "validation"
    if not val_dir.exists():
        return {}
    return {case: val_dir / f"{case}.nii.gz" for case in uio.list_cases(val_dir, C.LABEL_SUFFIX)}


def _center_of(mask: np.ndarray, shape) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.asarray(shape) // 2
    return pts.mean(0).round().astype(int)


def _class_dice(pred: np.ndarray, gt: np.ndarray, n_vessel: int) -> float:
    present = [c for c in range(1, n_vessel + 1) if (gt == c).any()]
    if not present:
        return float("nan")
    dices = []
    for c in present:
        g, p = (gt == c), (pred == c)
        denom = int(g.sum() + p.sum())
        dices.append(2 * int((g & p).sum()) / denom if denom else 0.0)
    return float(np.mean(dices))


def _row(axes_row, img: np.ndarray, label: np.ndarray, colors: np.ndarray,
        center, title_prefix: str, stat: str | None):
    z, y, x = center
    win = _windowed(img)
    views = (("axial", win[z, :, :], label[z, :, :]),
             ("coronal", win[:, y, :], label[:, y, :]),
             ("sagittal", win[:, :, x], label[:, :, x]))
    for ax, (name, gray, lab) in zip(axes_row, views):
        ax.imshow(gray, cmap="gray", origin="lower", interpolation="nearest")
        ax.imshow(_label_rgba(lab, colors), origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=9)
    ylabel = title_prefix if stat is None else f"{title_prefix}\n{stat}"
    axes_row[0].set_ylabel(ylabel, fontsize=8)


def make_grid(cases: list[str], centers: dict, label_paths: dict, colors: np.ndarray,
             n_vessel: int, gt_paths: dict, out_path: Path, suptitle: str,
             show_dice: bool) -> Path | None:
    rows = []
    for case in cases:
        lp = label_paths.get(case)
        if lp is None or not Path(lp).exists():
            print(f"[skip] {case}: missing for this figure")
            continue
        label, _ = uio.read(lp)
        img, _ = uio.read(C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}")
        stat = None
        if show_dice:
            gt, _ = uio.read(gt_paths[case])
            dice = _class_dice(label, gt, n_vessel)
            stat = f"mean per-class Dice={dice:.2f}" if dice == dice else "no GT vessel voxels"
        rows.append((case, img, label, stat))

    if not rows:
        return None

    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for axes_row, (case, img, label, stat) in zip(axes, rows):
        _row(axes_row, img, label, colors, centers[case], case, stat)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def make_legend(vessels: list[str], colors: np.ndarray, out_path: Path) -> Path:
    n = len(vessels)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, ax = plt.subplots(figsize=(4 * ncols, 0.3 * nrows))
    ax.set_xlim(0, ncols); ax.set_ylim(0, nrows); ax.axis("off")
    for i, name in enumerate(vessels):
        col, row = i % ncols, i // ncols
        ax.add_patch(plt.Rectangle((col, nrows - row - 1), 0.2, 0.7, color=colors[i]))
        ax.text(col + 0.28, nrows - row - 0.65, name, fontsize=7, va="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cases", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path,
                    default=C.EXP_ROOT / "figures" / "vessel_overlap_viz")
    a = ap.parse_args()

    spec = C.load_labels()
    n_vessel = spec.n_vessel
    colors = _vessel_colors(n_vessel)

    pred_paths_by_tag = {tag: load_vessel_pred_paths(trainer) for _, trainer, tag in MODELS}
    for label, trainer, tag in MODELS:
        print(f"{label} ({trainer}): {len(pred_paths_by_tag[tag])} predictions found "
             f"(fold_{C.SEG_FOLD}/validation)")

    common = None
    for _, _, tag in MODELS:
        cases = set(pred_paths_by_tag[tag])
        common = cases if common is None else (common & cases)
    common = sorted(c for c in common if (C.VESSEL_MASKS / f"{c}{C.LABEL_SUFFIX}").exists())
    print(f"{len(common)} cases have predictions from all 3 models AND a ground-truth vessel mask")

    rng = np.random.default_rng(a.seed)
    chosen = sorted(rng.choice(common, size=min(a.n_cases, len(common)), replace=False).tolist())
    print(f"Cases chosen: {chosen}")

    gt_paths = {c: C.VESSEL_MASKS / f"{c}{C.LABEL_SUFFIX}" for c in chosen}
    centers = {}
    for c in chosen:
        gt, _ = uio.read(gt_paths[c])
        centers[c] = _center_of(gt > 0, gt.shape)

    p_gt = make_grid(chosen, centers, gt_paths, colors, n_vessel, gt_paths,
                     a.out_dir / "vessel_overlap_gt_reference.png",
                     "Dataset302 ground truth -- 36-class vessel regions", show_dice=False)
    print(f"  -> {p_gt}")

    for label, trainer, tag in MODELS:
        p = make_grid(chosen, centers, pred_paths_by_tag[tag], colors, n_vessel, gt_paths,
                      a.out_dir / f"vessel_overlap_{tag}.png",
                      f"Dataset302 prediction -- {label}", show_dice=True)
        print(f"  -> {p}")

    p_leg = make_legend(spec.vessels, colors, a.out_dir / "vessel_overlap_legend.png")
    print(f"  -> {p_leg}")


if __name__ == "__main__":
    main()
