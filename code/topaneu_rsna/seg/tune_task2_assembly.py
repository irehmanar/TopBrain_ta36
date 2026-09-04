"""
Task 2 (52-class aneurysm-location) mask assembly + local metric tuning.

Rather than the collapsed direct 52-class segmenter (Dataset303), this fuses
two pieces of the pipeline that already work:

  shape/position  <- binary aneurysm segmentation. Dataset304 (whole-head,
                      ~64% Dice) is the default, on the same native per-case
                      grid as LOCATION_MASKS -- no crop/back-projection
                      needed. --binary_dataset 305 (ROI-cropped + vessel
                      channel) also works: its ground truth is reconstructed
                      by re-cropping LOCATION_MASKS to the exact same box
                      build_aneurysm_roi_dataset.py used (recomputed from
                      Model 2's saved prediction, deterministic, since that
                      crop offset was never itself written to disk).
  class label     <- the ROI classifier's per-case location probabilities,
                      aggregated out-of-fold (cls/train.py's oof.npz, one
                      fold per case -- honest, held-out, no leakage).

Per case: take the binary prediction's connected components, keep the
`--max_components` largest, and paint them with the classifier's top
predicted location if its probability clears `--threshold`; otherwise the
case gets an all-background prediction. Scored against real LOCATION_MASKS
ground truth using the *same* Dice/VS/HD95/Precision/Recall/MCC formulas as
seg/evaluate_location.py (which mirrors the grand-challenge's own
evaluation), swept over threshold.

Known v1 simplification: every case gets a single location label (this
pipeline's classifier predicts per-case, not per-component), so a case with
more than one true aneurysm, or a binary prediction with more than one real
lesion, isn't handled correctly yet. Fine for now since most cases have
exactly one aneurysm; flagged here so it isn't forgotten.

    python -m topaneu_rsna.seg.tune_task2_assembly
    python -m topaneu_rsna.seg.tune_task2_assembly --thresholds 0.1 0.3 0.5 0.7
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import geometry as geo
from topaneu_rsna.utils import io as uio
from topaneu_rsna.seg.evaluate_location import hd95

# Datasets whose binary predictions live on the coarse-ROI-cropped grid
# (job 05's 140mm cube, resampled to FINE_SPACING) rather than the whole-head
# native grid LOCATION_MASKS uses. For these, the matching ground truth has
# to be cropped to the exact same box -- recomputed deterministically from
# Model 2's saved prediction, the same geometry build_aneurysm_roi_dataset.py
# used, since that crop offset was never itself written to disk.
CROPPED_DATASETS = {C.DS_ANEURYSM_ROI}


def load_oof_probs(results_dir: Path) -> dict:
    """case -> held-out (n_loc,) location-probability vector, aggregated
    across all folds' oof.npz. Each case appears in exactly one fold's
    held-out set, so this covers every case exactly once, leakage-free."""
    probs = {}
    for f in sorted(results_dir.glob("fold_*/oof.npz")):
        d = np.load(f, allow_pickle=True)
        for case, p in zip(d["cases"], d["p"]):
            probs[str(case)] = p
    return probs


def load_binary_pred_paths(dataset_id: int, trainer: str, plans: str, folds,
                           model_dir: Path | None = None) -> dict:
    """case -> path to its held-out binary prediction, pooled across the
    folds' `validation` dirs (same source evaluate_location.py reads).

    `model_dir`, if given, is used as-is instead of being reconstructed as
    `{trainer}__{plans}__3d_fullres` -- for results kept under a renamed
    (e.g. `__backup_<timestamp>`) directory rather than nnU-Net's default
    naming."""
    model_dir = model_dir or (C.nnUNet_results
                              / f"Dataset{dataset_id:03d}_{C.DS_NAMES[dataset_id]}"
                              / f"{trainer}__{plans}__3d_fullres")
    out = {}
    for fold in folds:
        val_dir = model_dir / f"fold_{fold}" / "validation"
        if not val_dir.exists():
            continue
        for case in uio.list_cases(val_dir, C.LABEL_SUFFIX):
            out[case] = val_dir / f"{case}.nii.gz"
    return out


def cropped_ground_truth(case: str, ref_shape) -> tuple[np.ndarray, tuple]:
    """The full 52-class location label, cropped to the exact same box
    build_aneurysm_roi_dataset.py used for this case's binary prediction --
    recomputed from Model 2's saved prediction (deterministic, same inputs)
    rather than from a stored crop offset, since none was ever written out."""
    img, _ = uio.read(C.COARSE_ROI_DIR / f"{case}{C.IMAGE_SUFFIX}")
    v2, _ = uio.read(C.VESSEL_PRED_M2 / f"{case}.nii.gz")
    v2 = geo.crop_pad(v2.astype(np.uint8), (0, 0, 0), img.shape)

    lo, hi = geo.tight_bounds(v2, C.ROI_REFINE_MARGIN_MM, C.FINE_SPACING)
    if lo is None:
        lo, hi = np.zeros(3, int), np.asarray(img.shape, int)
    lo, hi = geo.center_to_size(lo, hi, C.FINAL_ROI_SIZE)

    loc_full, gmeta = uio.read(C.COARSE_ROI_DIR / f"{case}_location{C.LABEL_SUFFIX}")
    gt = geo.crop_pad(loc_full.astype(np.uint8), lo, hi)
    if gt.shape != tuple(ref_shape):
        gt = geo.crop_pad(gt, (0, 0, 0), ref_shape)
    return gt, gmeta["spacing"]


def load_case(case: str, bin_path: Path, n_loc: int, probs: np.ndarray,
             cropped: bool) -> dict:
    binmask, _ = uio.read(bin_path)
    binmask = binmask > 0
    lab, n = ndimage.label(binmask)
    sizes = ndimage.sum(binmask, lab, index=np.arange(1, n + 1)) if n else np.array([])
    order = np.argsort(-sizes) if n else np.array([], dtype=int)

    if cropped:
        gt, spacing = cropped_ground_truth(case, binmask.shape)
    else:
        gt, gmeta = uio.read(C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}")
        spacing = gmeta["spacing"]
    gt_count = np.array([int((gt == c).sum()) for c in range(1, n_loc + 1)])

    top_loc = int(np.argmax(probs))
    top_p = float(probs[top_loc])

    return dict(lab=lab, sizes=sizes, order=order, gt=gt, gt_count=gt_count,
               spacing=spacing, shape=binmask.shape,
               top_loc=top_loc, top_p=top_p)


def score_at_threshold(per_case: dict, n_loc: int, threshold: float,
                       max_components: int, min_voxels: int) -> dict:
    inter = np.zeros(n_loc); pred_sum = np.zeros(n_loc); gt_sum = np.zeros(n_loc)
    tp = np.zeros(n_loc); fp = np.zeros(n_loc); fn = np.zeros(n_loc); tn = np.zeros(n_loc)
    hd_sum = np.zeros(n_loc); hd_n = np.zeros(n_loc)
    n_predicted = 0

    for d in per_case.values():
        gt_sum += d["gt_count"]
        predicted = d["top_p"] >= threshold
        pred_label = d["top_loc"] + 1 if predicted else None

        pm_full = None
        if predicted:
            keep = d["order"][:max_components]
            m = np.zeros(d["shape"], dtype=bool)
            for comp_idx in keep:
                if d["sizes"][comp_idx] < min_voxels:
                    continue
                m |= (d["lab"] == (comp_idx + 1))
            if m.any():
                pm_full = m
                n_predicted += 1
            else:
                pred_label = None  # confident but every candidate component was too small

        for c in range(1, n_loc + 1):
            gn = int(d["gt_count"][c - 1])
            if c == pred_label:
                gm = d["gt"] == c
                pn = int(pm_full.sum())
                it = int((pm_full & gm).sum())
                pred_sum[c - 1] += pn; inter[c - 1] += it
                if gn > 0 and it > 0:
                    tp[c - 1] += 1
                elif gn > 0:
                    fn[c - 1] += 1
                elif pn > 0:
                    fp[c - 1] += 1
                else:
                    tn[c - 1] += 1
                if pn > 0 and gn > 0:
                    h = hd95(pm_full, gm, d["spacing"])
                    if h is not None:
                        hd_sum[c - 1] += h; hd_n[c - 1] += 1
            else:
                if gn > 0:
                    fn[c - 1] += 1
                else:
                    tn[c - 1] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2 * inter / (pred_sum + gt_sum)
        vs = 1 - np.abs(pred_sum - gt_sum) / (pred_sum + gt_sum)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        hd = hd_sum / hd_n

    return dict(threshold=threshold,
               dice=float(np.nanmean(dice)), vs=float(np.nanmean(vs)),
               hd95=float(np.nanmean(hd)), precision=float(np.nanmean(precision)),
               recall=float(np.nanmean(recall)), mcc=float(np.nanmean(mcc)),
               n_predicted_cases=n_predicted, n_cases=len(per_case))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary_dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--model_dir", type=Path, default=None,
                    help="use this exact nnU-Net results dir instead of "
                         "reconstructing {trainer}__{plans}__3d_fullres -- "
                         "for results kept under a renamed/backup directory")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cls_results_dir", type=Path, default=C.CLS_RESULTS_DIR)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--max_components", type=int, default=1)
    ap.add_argument("--min_voxels", type=int, default=3)
    ap.add_argument("--out", type=Path, default=C.LOG_ROOT / "task2_assembly_sweep.csv")
    a = ap.parse_args()

    spec = C.load_labels()
    n_loc = spec.n_loc

    probs = load_oof_probs(a.cls_results_dir)
    bin_paths = load_binary_pred_paths(a.binary_dataset, a.trainer, a.plans, a.folds,
                                       a.model_dir)

    cases = sorted(set(probs) & set(bin_paths))
    print(f"{len(cases)} cases with both a binary prediction and a classifier OOF probability")
    if len(bin_paths) > len(cases):
        print(f"  {len(bin_paths) - len(cases)} binary predictions skipped (no OOF row)")
    if len(probs) > len(cases):
        print(f"  {len(probs) - len(cases)} OOF rows skipped (no binary prediction)")
    if not cases:
        raise SystemExit("no overlapping cases -- check --cls_results_dir / --binary_dataset")

    cropped = a.binary_dataset in CROPPED_DATASETS
    if cropped:
        print(f"Dataset{a.binary_dataset} is ROI-cropped -- reconstructing matching "
             "ground-truth crops from COARSE_ROI_DIR/VESSEL_PRED_M2 per case")

    per_case = {c: load_case(c, bin_paths[c], n_loc, probs[c], cropped)
               for c in tqdm(cases, desc="loading")}

    rows = []
    for thr in a.thresholds:
        row = score_at_threshold(per_case, n_loc, thr, a.max_components, a.min_voxels)
        rows.append(row)
        print(f"thr={thr:.2f}  dice={row['dice']:.4f}  vs={row['vs']:.4f}  "
              f"hd95={row['hd95']:.2f}mm  precision={row['precision']:.4f}  "
              f"recall={row['recall']:.4f}  mcc={row['mcc']:.4f}  "
              f"predicted-on={row['n_predicted_cases']}/{row['n_cases']}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
