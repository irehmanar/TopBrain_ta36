"""
Slide figures: Model 1 / Model 2 / Model 3 predictions on a handful of cases.

For each case, reads:
  Model 1  -- coarse 3-class vessel-group mask, from COARSE_PRED_DIR
              (job 05's `nnUNetv2_predict -d 301`), overlaid on the raw image
              resampled onto that same prediction grid.
  Model 2  -- fine 36-class vessel mask, from VESSEL_PRED_M2
              (job 06's Model 2 `nnUNetv2_predict -d 302`), overlaid on the
              coarse-ROI-cropped image (COARSE_ROI_DIR/<case>_0000.nii.gz).
  Model 3  -- same as Model 2 but VESSEL_PRED_M3 (the SkeletonRecall-w3 run).

and writes one PNG per case: 3 rows (Model 1/2/3) x 3 columns (axial /
coronal / sagittal), sliced through each model's own foreground centroid so
each row is centred on what that model actually predicted.

Orientation note: slices are shown as `arr[idx]` with `origin="lower"` and no
LPS/RAS-aware flipping -- verify left/right and superior/inferior orientation
before dropping a panel straight into slides.

Usage (see jobs/22_visualize_pipeline.sbatch):
    python -m topaneu_rsna.viz.visualize_pipeline --n 5
    python -m topaneu_rsna.viz.visualize_pipeline --cases case001,case002
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from topaneu_rsna import config as C
from topaneu_rsna.utils import geometry as geo
from topaneu_rsna.utils import io as uio

# tab20 + tab20b + tab20c: 60 visually distinct colors, more than enough for
# Model 1's 3 classes and Models 2/3's 36 -- indexed by (class_id - 1) so the
# same vessel always gets the same color across every case and every model.
_PALETTE = [c for cmap in ("tab20", "tab20b", "tab20c")
            for c in plt.get_cmap(cmap).colors]


def _color(class_id: int):
    return _PALETTE[(int(class_id) - 1) % len(_PALETTE)]


def _windowed(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _overlay_rgba(label_slice: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    h, w = label_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for cls_id in np.unique(label_slice):
        if cls_id == 0:
            continue
        r, g, b = _color(cls_id)
        m = label_slice == cls_id
        rgba[m, 0], rgba[m, 1], rgba[m, 2], rgba[m, 3] = r, g, b, alpha
    return rgba


def _centroid_or_center(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask > 0)
    if pts.size == 0:
        return np.asarray(mask.shape) // 2
    return pts.mean(0).round().astype(int)


def _panel_row(fig, axes_row, img, label, title_prefix, n_classes_hint):
    z, y, x = _centroid_or_center(label)
    win = _windowed(img)
    views = (("axial", win[z, :, :], label[z, :, :]),
             ("coronal", win[:, y, :], label[:, y, :]),
             ("sagittal", win[:, :, x], label[:, :, x]))
    for ax, (name, gray, lab) in zip(axes_row, views):
        ax.imshow(gray, cmap="gray", origin="lower", interpolation="nearest")
        ax.imshow(_overlay_rgba(lab), origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title_prefix} -- {name}", fontsize=10)
    n_fg = int((label > 0).sum())
    axes_row[0].set_ylabel(
        f"{n_fg:,} fg voxels, {len(np.unique(label)) - 1} classes present",
        fontsize=8)


def make_figure(case: str, coarse_pred_dir: Path, roi_dir: Path,
                 m2_dir: Path, m3_dir: Path, out_dir: Path) -> Path | None:
    p1 = coarse_pred_dir / f"{case}.nii.gz"
    p_img = roi_dir / f"{case}{C.IMAGE_SUFFIX}"
    p2 = m2_dir / f"{case}.nii.gz"
    p3 = m3_dir / f"{case}.nii.gz"
    if not (p1.exists() and p_img.exists() and p2.exists() and p3.exists()):
        missing = [str(p) for p in (p1, p_img, p2, p3) if not p.exists()]
        print(f"[skip] {case}: missing {missing}")
        return None

    pred1, meta1 = uio.read(p1)
    img1, _ = uio.resample_to_reference(C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}",
                                        p1, is_label=False)

    img2, meta2 = uio.read(p_img)
    v2, _ = uio.read(p2)
    v3, _ = uio.read(p3)
    v2 = geo.crop_pad(v2.astype(np.uint8), (0, 0, 0), img2.shape)
    v3 = geo.crop_pad(v3.astype(np.uint8), (0, 0, 0), img2.shape)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    _panel_row(fig, axes[0], img1, pred1.astype(np.int32), "Model 1 (coarse, 3 groups)", 3)
    _panel_row(fig, axes[1], img2, v2.astype(np.int32), "Model 2 (fine, 36 vessels)", 36)
    _panel_row(fig, axes[2], img2, v3.astype(np.int32), "Model 3 (fine, 36 vessels)", 36)
    fig.suptitle(case, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case}_pipeline.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def make_legend(spec, out_dir: Path) -> Path:
    n = spec.n_vessel
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(figsize=(6 * ncol / 3, 0.28 * nrow))
    ax.set_xlim(0, ncol); ax.set_ylim(0, nrow); ax.axis("off")
    for i, name in enumerate(spec.vessels):
        row, col = divmod(i, ncol)
        y = nrow - 1 - row
        ax.add_patch(plt.Rectangle((col, y), 0.25, 0.7, color=_color(i + 1)))
        ax.text(col + 0.32, y + 0.35, f"{i + 1}: {name}", fontsize=7, va="center")
    fig.tight_layout()
    out_path = out_dir / "vessel_legend.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse_pred_dir", type=Path, default=C.COARSE_PRED_DIR)
    ap.add_argument("--roi_dir", type=Path, default=C.COARSE_ROI_DIR)
    ap.add_argument("--m2_dir", type=Path, default=C.VESSEL_PRED_M2)
    ap.add_argument("--m3_dir", type=Path, default=C.VESSEL_PRED_M3)
    ap.add_argument("--out_dir", type=Path, default=C.EXP_ROOT / "figures" / "pipeline_viz")
    ap.add_argument("--n", type=int, default=5, help="number of cases if --cases not given")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cases", type=str, default=None,
                    help="comma-separated case ids; overrides --n/--seed")
    ap.add_argument("--legend", action="store_true", help="also dump a vessel-name/color legend")
    a = ap.parse_args()

    if a.cases:
        cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    else:
        c1 = set(uio.list_cases(a.coarse_pred_dir, ".nii.gz"))
        c2 = set(uio.list_cases(a.m2_dir, ".nii.gz"))
        c3 = set(uio.list_cases(a.m3_dir, ".nii.gz"))
        cr = set(uio.list_cases(a.roi_dir, C.IMAGE_SUFFIX))
        candidates = sorted(c1 & c2 & c3 & cr)
        if not candidates:
            raise SystemExit("no case has predictions from all three models -- "
                             "run jobs 05 and 06 first")
        rng = np.random.default_rng(a.seed)
        cases = sorted(rng.choice(candidates, size=min(a.n, len(candidates)),
                                  replace=False).tolist())

    print(f"Visualizing {len(cases)} case(s): {cases}")
    made = []
    for case in cases:
        p = make_figure(case, a.coarse_pred_dir, a.roi_dir, a.m2_dir, a.m3_dir, a.out_dir)
        if p:
            made.append(p)
            print(f"  -> {p}")

    if a.legend:
        spec = C.load_labels()
        p = make_legend(spec, a.out_dir)
        print(f"  -> {p}")

    print(f"{len(made)}/{len(cases)} figures written to {a.out_dir}")


if __name__ == "__main__":
    main()
