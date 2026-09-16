"""
Experiment B, step 1/3: build the standalone crop classifier's training data
from REAL, non-oracle inputs only -- no ground truth as an input channel, no
ground-truth-centered crops.

CRITICAL DESIGN POINT (read before touching this file): training instances
are the REAL PREDICTED connected components from Dataset304's pooled 5-fold
binary prediction (the same real, non-oracle source assign_location_rule.py
and every other non-oracle experiment in this pipeline uses), NOT ground-
truth instances. Ground truth is used ONLY to look up each predicted
instance's true label via majority voxel overlap -- exactly
assign_location_rule.py's own `assign_case()` true_class snippet (vals,
counts = np.unique(gt[inst], return_counts=True); true_cls = the most
frequent nonzero value, else 0/background), copied here rather than
reimplemented differently.

Why not crop on ground-truth instances instead: doing so would train the
classifier on perfectly-centered, perfectly-shaped lesion crops, then run it
at inference on the real binary segmenter's noisy, offset, sometimes-
hallucinated predicted instances -- the exact oracle-vs-real distribution
mismatch this pipeline's earlier oracle-everything ablation already
diagnosed, just recurring one level down (crop instances instead of input
channels). Using real predicted instances for training means the model also
sees real hallucinations labeled "background" during training, which it
must learn to recognize as such -- this is standard practice in detection-
then-classify cascades (Fast/Faster R-CNN's own proposal-then-classify
recipe: train the classifier on the detector's own candidate regions,
matched to ground truth by overlap, background included, not on idealized
crops).

Crop size: NOT guessed -- this script scans every ground-truth lesion's own
bounding-box extent first, prints the percentile distribution (mm and
voxels), and derives a default crop size from the 95th percentile plus a
documented vessel-context margin. Override via --crop_size if the printed
stats suggest a different choice.

Channels (all three already share the same native per-case grid -- no
resampling needed, see assign_location_rule.py's own docstring on this):
  ch0 = raw image (IMAGES_DIR), z-scored (uio.zscore -- same normalization
        final_roi.py already uses to feed a raw crop into a pretrained
        nnU-Net-derived backbone, not invented fresh here)
  ch1 = Dataset304's real binary aneurysm prediction (0/1)
  ch2 = job 78's real whole-head Model 2 vessel prediction (0..36 label ids)
Label: the instance's true one-of-52 location, or "background" (id 0) for a
predicted instance that doesn't overlap any real lesion -- a real
hallucination, and a real, necessary training example of what one looks
like, not an artifact to filter out.

Fold split: reused from Dataset304's own real fold assignment -- whichever
fold_X/validation/ directory a case's prediction file lives under IS that
case's held-out fold (the same signal assign_location_rule.load_binary_pred_
paths already relies on, just tracked per-case here instead of pooled).
config.EXPB_VAL_FOLD is held out entirely; the rest is training data. This
is this exact codebase's own reused split, not an invented one.

Mandatory leakage checks (printed every run, not assumed correct):
  - total instances, per-fold instance counts
  - confirms no case ID is assigned to more than one fold

    python -m topaneu_rsna.seg.build_expB_crop_dataset
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import geometry as geo
from topaneu_rsna.utils import io as uio

MIN_VOXELS = 3   # same convention as assign_location_rule.py's own default


def build_case_to_fold(dataset_id: int, trainer: str, plans: str, folds=range(5)) -> dict:
    """Case -> which fold's validation/ directory its real prediction came
    from. A case appears under exactly one fold's validation/ (nnU-Net's own
    5-fold split is a partition), which is exactly why pooling all 5 gives a
    prediction for every case with none of them being the model's own
    training data."""
    model_dir = (C.nnUNet_results / f"Dataset{dataset_id:03d}_{C.DS_NAMES[dataset_id]}"
                / f"{trainer}__{plans}__3d_fullres")
    case_to_fold = {}
    for fold in folds:
        val_dir = model_dir / f"fold_{fold}" / "validation"
        if not val_dir.exists():
            continue
        for case in uio.list_cases(val_dir, C.LABEL_SUFFIX):
            if case in case_to_fold:
                print(f"[LEAKAGE WARNING] case {case} appears under more than one "
                     f"fold's validation/ (folds {case_to_fold[case]} and {fold}) -- "
                     f"this should never happen with a clean nnU-Net split, stopping.")
                raise SystemExit(1)
            case_to_fold[case] = fold
    return case_to_fold


def gt_extent_stats(cases: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth lesion bounding-box extents (voxels, mm) across the whole
    cohort -- used to justify the crop size from real data, not a guess."""
    extents_vox, extents_mm = [], []
    for case in tqdm(cases, desc="scanning GT extents"):
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not gt_p.exists():
            continue
        gt, meta = uio.read(gt_p)
        lab, n = ndimage.label(gt > 0)
        spacing = np.asarray(meta["spacing"])
        for i in range(1, n + 1):
            idx = np.argwhere(lab == i)
            if idx.shape[0] < MIN_VOXELS:
                continue
            ext_vox = idx.max(0) - idx.min(0) + 1
            extents_vox.append(ext_vox)
            extents_mm.append(ext_vox * spacing)
    return np.array(extents_vox), np.array(extents_mm)


def network_divisibility(dataset_id: int = None) -> np.ndarray:
    """Per-axis total downsampling factor (product of strides across every
    stage) of Dataset304's own ResEncUNetM encoder -- the crop size MUST be
    an exact multiple of this per axis, or a residual block's skip
    connection and main path round to different spatial sizes and crash
    with a shape mismatch (RuntimeError: size of tensor a (6) must match
    size of tensor b (5) ... -- exactly what an earlier, naive 'round to
    nearest 8' heuristic hit here: this architecture's actual per-axis
    factor is NOT simply 8, and guessing a rounding granularity instead of
    reading it from the architecture is exactly how that bug happened).
    Reads Dataset304's own plans.json since that is the encoder this
    experiment warm-starts from and must stay shape-compatible with."""
    import json
    dataset_id = dataset_id or C.DS_ANEURYSM
    model_dir = C.seg_model_dir(dataset_id, C.TRAINER_LOC)
    plans = json.loads((model_dir / "plans.json").read_text())
    strides = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["strides"]
    factor = np.ones(len(strides[0]), dtype=np.int64)
    for s in strides:
        factor *= np.asarray(s, dtype=np.int64)
    return factor


def pick_crop_size(extents_vox: np.ndarray, margin_factor: float = 2.0,
                   min_size: int = 32, max_size: int = 128) -> tuple[int, int, int]:
    divisor = network_divisibility()
    p95 = np.percentile(extents_vox, 95, axis=0)
    raw = np.clip(p95 * margin_factor, min_size, max_size)
    size = np.ceil(raw / divisor) * divisor          # round UP to a valid multiple, per axis --
                                                     # correctness (must be shape-compatible with
                                                     # the encoder) takes priority over hitting
                                                     # max_size exactly if those two ever conflict
    if np.any(size > max_size):
        print(f"[warn] per-axis divisibility ({divisor.tolist()}) pushed crop size to "
             f"{size.astype(int).tolist()}, past the requested max_size={max_size} -- "
             f"kept as-is since a shape-incompatible smaller crop would crash training.")
    return tuple(int(s) for s in size)


def true_class_of(inst_mask: np.ndarray, gt: np.ndarray) -> int:
    """Copied from assign_location_rule.py's assign_case(): most frequent
    nonzero GT value under the predicted instance's own voxels, else 0
    (background) -- ground truth is used ONLY for this label lookup."""
    vals, counts = np.unique(gt[inst_mask], return_counts=True)
    nz = vals != 0
    return int(vals[nz][np.argmax(counts[nz])]) if nz.any() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop_size", type=int, nargs=3, default=None,
                    help="z y x crop size in voxels; default is data-driven, "
                         "see printed extent statistics")
    ap.add_argument("--binary_dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--out_dir", type=Path, default=C.EXPB_CROP_CACHE)
    ap.add_argument("--manifest_csv", type=Path, default=C.EXPB_MANIFEST_CSV)
    a = ap.parse_args()

    spec = C.load_labels()
    id_to_loc = {i + 1: loc for i, loc in enumerate(spec.locations)}
    id_to_loc[0] = "background"

    case_to_fold = build_case_to_fold(a.binary_dataset, a.trainer, a.plans)
    cases = sorted(case_to_fold)
    print(f"{len(cases)} cases with a real (non-oracle) binary prediction, "
         f"case-to-fold mapping built from Dataset{a.binary_dataset}'s own real split")

    extents_vox, extents_mm = gt_extent_stats(cases)
    print(f"\nGround-truth lesion extent statistics (n={len(extents_vox)} lesions):")
    for label, arr, unit in (("voxels", extents_vox, ""), ("mm", extents_mm, "mm")):
        p50, p95, mx = np.percentile(arr, 50, axis=0), np.percentile(arr, 95, axis=0), arr.max(axis=0)
        print(f"  [{label}] median z,y,x={p50.round(1)}  p95={p95.round(1)}  max={mx.round(1)}{unit}")

    crop_size = tuple(a.crop_size) if a.crop_size else pick_crop_size(extents_vox)
    print(f"\nCrop size chosen: {crop_size} voxels "
         f"({'user-specified' if a.crop_size else 'data-driven: 2x the 95th-percentile lesion extent, rounded to a multiple of 8, clamped to [32,96]'})")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    fold_counts = {f: 0 for f in range(5)}

    for case in tqdm(cases, desc="building crops"):
        fold = case_to_fold[case]
        bin_p = (C.nnUNet_results / f"Dataset{a.binary_dataset:03d}_{C.DS_NAMES[a.binary_dataset]}"
                / f"{a.trainer}__{a.plans}__3d_fullres" / f"fold_{fold}" / "validation"
                / f"{case}{C.LABEL_SUFFIX}")
        img_p = C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}"
        vessel_p = C.VESSEL_PRED_M2_FULLHEAD / f"{case}.nii.gz"
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not (bin_p.exists() and img_p.exists() and vessel_p.exists() and gt_p.exists()):
            print(f"[skip] {case}: missing one of image/binary-pred/vessel-pred/gt")
            continue

        img, meta = uio.read(img_p)
        img = uio.zscore(img.astype(np.float32))
        binmask, _ = uio.read(bin_p)
        binmask = (binmask > 0).astype(np.float32)
        binmask = geo.crop_pad(binmask, (0, 0, 0), img.shape)
        vessel, _ = uio.read(vessel_p)
        vessel = geo.crop_pad(vessel.astype(np.float32), (0, 0, 0), img.shape)
        gt, _ = uio.read(gt_p)

        lab, n = ndimage.label(binmask > 0)
        for i in range(1, n + 1):
            inst = lab == i
            size = int(inst.sum())
            if size < MIN_VOXELS:
                continue
            center = np.argwhere(inst).mean(0).round().astype(int)
            lo = center - np.array(crop_size) // 2
            hi = lo + np.array(crop_size)

            crop = np.stack([
                geo.crop_pad(img, lo, hi),
                geo.crop_pad(binmask, lo, hi),
                geo.crop_pad(vessel, lo, hi),
            ], axis=0).astype(np.float32)

            true_id = true_class_of(inst, gt)
            npz_path = a.out_dir / f"{case}_{i}.npz"
            np.savez_compressed(npz_path, crop=crop, label=true_id)

            manifest.append(dict(case=case, instance_idx=i, fold=fold, size=size,
                                 true_class=id_to_loc[true_id], true_id=true_id,
                                 npz_path=str(npz_path)))
            fold_counts[fold] += 1

    a.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.manifest_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "instance_idx", "fold", "size",
                                          "true_class", "true_id", "npz_path"])
        w.writeheader()
        w.writerows(manifest)

    print(f"\n=== mandatory leakage / coverage checks ===")
    print(f"total instances: {len(manifest)}")
    for f in range(5):
        print(f"  fold {f}: {fold_counts[f]} instances, "
             f"{sum(1 for c, ff in case_to_fold.items() if ff == f)} cases")
    n_bg = sum(1 for m in manifest if m["true_id"] == 0)
    print(f"background (hallucination) instances: {n_bg} "
         f"({100 * n_bg / max(1, len(manifest)):.1f}%)")
    case_fold_counts = {}
    for m in manifest:
        case_fold_counts.setdefault(m["case"], set()).add(m["fold"])
    multi = [c for c, fs in case_fold_counts.items() if len(fs) > 1]
    print(f"cases appearing in more than one fold: {len(multi)} "
         f"({'PASS -- none found' if not multi else 'FAIL -- ' + str(multi)})")

    (C.EXPB_CROP_CACHE / "crop_size.json").write_text(json.dumps({"crop_size": crop_size}))
    print(f"\nmanifest written to {a.manifest_csv}")
    print(f"crops written to {a.out_dir}")


if __name__ == "__main__":
    main()
