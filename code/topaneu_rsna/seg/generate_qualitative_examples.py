"""
Slide figures for the "Qualitative examples" slide: axial overlay, coronal
overlay, and a 3D vessel-tree render, for 10 TopBrain_Data patients, for
BOTH the submitted vessel model (Model 2, Dataset302, checkpoint_final.pth)
and the in-progress class-balanced retrain (job 165, checkpoint_best.pth) --
2 models x 10 patients x 3 images = 60 PNGs.

Uses TopBrain_Data (not Dataset302's own training cohort) so these images
are directly relevant to the actual TA36 submission story, not just a
self-evaluation figure. Both models run LIVE inference here (reusing the
same nnUNetPredictor + OOM patch-ladder pattern already proven in
evaluate_topbrain_baseline.py and the TA36 Docker container's inference.py)
because the class-balanced model has no fold_all/validation/ folder yet --
training hasn't finished, so there is nothing precomputed to read the way
viz/visualize_vessel_overlap.py does for already-completed models.

Reuses viz/visualize_vessel_overlap.py's own windowing/coloring/centering
helpers directly (_windowed, _vessel_colors, _label_rgba, _center_of) so the
color-per-vessel-class convention matches every other figure in this project
exactly, rather than inventing a second palette.

    python -m topaneu_rsna.seg.generate_qualitative_examples
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3d projection
import numpy as np
import SimpleITK as sitk
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from topaneu_rsna import config as C
from topaneu_rsna.viz.visualize_vessel_overlap import (
    _windowed, _vessel_colors, _label_rgba, _center_of)

IMAGE_SUBDIR = "imagesTr_topbrain"
IMAGE_SUFFIX = "_0000.nii.gz"

# (short tag for filenames, trainer name, checkpoint file) -- Model 2's own
# checkpoint_final.pth (job 08, fully trained) vs the class-balanced retrain's
# checkpoint_best.pth (job 165, still training -- see that job's own caveats).
MODELS = [
    ("model2_base", C.TRAINER_M2, "checkpoint_final.pth"),
    ("model2_classbalanced", "RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07_ClassBalanced",
     "checkpoint_best.pth"),
]

_PATCH_LADDER = ([128, 256, 256], [96, 192, 192], [64, 160, 160], [64, 128, 128], [48, 96, 96])


def _find_images_dir(root):
    if (root / IMAGE_SUBDIR).is_dir():
        return root / IMAGE_SUBDIR
    hits = sorted(root.glob(f"**/{IMAGE_SUBDIR}"))
    if not hits:
        raise FileNotFoundError(f"could not find '{IMAGE_SUBDIR}' under {root}")
    return hits[0]


def _load_predictor(trainer: str, checkpoint: str, device):
    model_dir = C.seg_model_dir(C.DS_VESSEL, trainer)
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=False,
        device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=("all",), checkpoint_name=checkpoint)
    return predictor


def _predict(predictor, img: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)[None]
    spacing_zyx = tuple(img.GetSpacing()[::-1])
    props = {"spacing": spacing_zyx}
    last_err = None
    for patch in _PATCH_LADDER:
        predictor.configuration_manager.configuration["patch_size"] = list(patch)
        try:
            return predictor.predict_single_npy_array(arr, props)
        except torch.OutOfMemoryError as e:
            last_err = e
            torch.cuda.empty_cache()
    raise last_err


def _save_slice(img, label, colors, z_yx, axis, out_path, title):
    win = _windowed(img)
    if axis == "axial":
        gray, lab = win[z_yx[0], :, :], label[z_yx[0], :, :]
    else:  # coronal
        gray, lab = win[:, z_yx[1], :], label[:, z_yx[1], :]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(gray, cmap="gray", origin="lower", interpolation="nearest")
    ax.imshow(_label_rgba(lab, colors), origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_3d(label, spacing_zyx, colors, out_path, title, max_points=20000, seed=0):
    zz, yy, xx = np.nonzero(label)
    if zz.size == 0:
        return
    if zz.size > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(zz.size, size=max_points, replace=False)
        zz, yy, xx = zz[keep], yy[keep], xx[keep]
    cls = label[zz, yy, xx].astype(int) - 1
    pt_colors = colors[cls]

    sz, sy, sx = spacing_zyx
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xx * sx, yy * sy, zz * sz, c=pt_colors, s=1.5, depthshade=True)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_patients", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=type(C.WORK), default=C.WORK / "qualitative_examples")
    a = ap.parse_args()

    spec = C.load_labels()
    colors = _vessel_colors(spec.n_vessel)

    images_dir = _find_images_dir(C.TOPBRAIN_DATA_ROOT)
    cases = sorted(p.name[: -len(IMAGE_SUFFIX)] for p in images_dir.glob(f"*{IMAGE_SUFFIX}"))
    rng = np.random.default_rng(a.seed)
    chosen = sorted(rng.choice(cases, size=min(a.n_patients, len(cases)), replace=False).tolist())
    print(f"[qualitative] {len(cases)} TopBrain cases available, chose {len(chosen)}: {chosen}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for tag, trainer, checkpoint in MODELS:
        print(f"[qualitative] loading {tag} ({trainer}, {checkpoint})")
        predictor = _load_predictor(trainer, checkpoint, device)

        for case in chosen:
            img = sitk.ReadImage(str(images_dir / f"{case}{IMAGE_SUFFIX}"))
            pred = _predict(predictor, img)
            spacing_zyx = img.GetSpacing()[::-1]
            center = _center_of(pred > 0, pred.shape)

            out_base = a.out_dir / tag
            _save_slice(sitk.GetArrayFromImage(img), pred, colors, center, "axial",
                       out_base / f"{case}_axial.png", f"{tag} -- {case} -- axial")
            _save_slice(sitk.GetArrayFromImage(img), pred, colors, center, "coronal",
                       out_base / f"{case}_coronal.png", f"{tag} -- {case} -- coronal")
            _save_3d(pred, spacing_zyx, colors, out_base / f"{case}_3d.png",
                    f"{tag} -- {case} -- 3D vessel tree")
            print(f"[qualitative]   {tag}/{case}: 3 images written")

        del predictor
        torch.cuda.empty_cache()

    n_images = len(MODELS) * len(chosen) * 3
    print(f"[qualitative] DONE -- {n_images} images written under {a.out_dir}")
    (a.out_dir / "chosen_cases.json").write_text(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
