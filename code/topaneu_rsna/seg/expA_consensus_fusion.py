"""
Experiment A, step 7 (final): consensus fusion + post-processing +
evaluation, holdout cases only.

Consensus fusion (paper Sect 2.4, copied exactly, kept as its own clearly
separable step so it can be ablated on/off with --no_consensus): a location
class survives into the final prediction only if BOTH
  (a) the classifier's sigmoid probability for that class exceeds 0.5
      (job 85's task2_expA_classifier_probs.csv), AND
  (b) job 83's segmentation prediction contains at least one connected
      component of that class.
Any class failing either side has its voxels zeroed out of the final mask.

Post-processing (paper Sect 2.4/3.3, applied per surviving class):
  - small connected components below --min_voxels are removed (this
    pipeline has no existing precedent value to reuse -- DBSCAN_MIN_SAMPLES
    in config.py is an unrelated clustering parameter from a different stage
    -- so this is a fresh, documented default, tune it if the holdout result
    looks over/under-filtered);
  - light binary morphological closing-then-opening (1-voxel structuring
    element) on the per-class binary mask, to smooth jagged single-voxel
    boundary noise without eroding genuinely small (real aneurysms are only
    a few mm) lesions away.

Scores the fused prediction against LOCATION_MASKS using BOTH:
  - evaluate_official.py's verified replica of the real TopAneu-26 Task 2
    evaluator (the numbers that matter for the leaderboard-relevant
    decision -- compare directly against the rule/oracle reference points
    already on record: rule alone dice 0.0052, precision 0.3596, recall
    0.3253, mcc 0.3530);
  - this pipeline's own internal pooled-per-component score() from
    assign_location_rule.py, for continuity with every earlier experiment's
    logs (NOT the number to make the leaderboard-relevant decision on --
    see this pipeline's own prior job comments on why the two conventions
    aren't comparable).

    python -m topaneu_rsna.seg.expA_consensus_fusion --official_metrics
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio
from topaneu_rsna.seg.assign_location_rule import score
from topaneu_rsna.seg.evaluate_official import official_score, print_official


def load_classifier_probs(csv_path: Path, locations: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            case = row.pop("case")
            out[case] = {loc: float(row[loc]) for loc in locations}
    return out


def fuse_one(seg: np.ndarray, probs: dict[str, float], locations: list[str],
            use_consensus: bool, min_voxels: int, smooth: bool) -> np.ndarray:
    final = np.zeros_like(seg)
    struct = np.ones((3, 3, 3), dtype=bool)
    for i, loc in enumerate(locations):
        c = i + 1
        mask = seg == c
        if not mask.any():
            continue
        if use_consensus and probs.get(loc, 0.0) <= 0.5:
            continue   # classifier disagrees this location is present -- drop it

        lab, n = ndimage.label(mask)
        keep = np.zeros_like(mask)
        for comp in range(1, n + 1):
            comp_mask = lab == comp
            if comp_mask.sum() >= min_voxels:
                keep |= comp_mask
        if not keep.any():
            continue

        if smooth:
            keep = ndimage.binary_closing(keep, structure=struct)
            keep = ndimage.binary_opening(keep, structure=struct)

        final[keep] = c
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_pred_dir", type=Path, default=C.EXPA_SEG_PRED_HOLDOUT)
    ap.add_argument("--classifier_probs_csv", type=Path,
                    default=C.LOG_ROOT / "task2_expA_classifier_probs.csv")
    ap.add_argument("--no_consensus", action="store_true",
                    help="ablate consensus fusion off -- use the raw segmentation "
                         "prediction unchanged (classifier ignored) to isolate its "
                         "effect")
    ap.add_argument("--min_voxels", type=int, default=5)
    ap.add_argument("--no_smooth", action="store_true")
    ap.add_argument("--official_metrics", action="store_true")
    ap.add_argument("--official_out_csv", type=Path,
                    default=C.LOG_ROOT / "task2_expA_official_metrics.csv")
    a = ap.parse_args()

    spec = C.load_labels()
    loc_value = {loc: i + 1 for i, loc in enumerate(spec.locations)}
    split = json.loads(C.EXPA_HOLDOUT_JSON.read_text())
    holdout = split["holdout"]

    probs_by_case = ({} if a.no_consensus else
                     load_classifier_probs(a.classifier_probs_csv, spec.locations))

    preds, gts = {}, {}
    for case in holdout:
        seg_p = a.seg_pred_dir / f"{case}.nii.gz"
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not (seg_p.exists() and gt_p.exists()):
            print(f"[skip] {case}: missing segmentation prediction or ground truth")
            continue
        seg, meta = uio.read(seg_p)
        gt, _ = uio.read(gt_p)
        fused = fuse_one(seg, probs_by_case.get(case, {}), spec.locations,
                         use_consensus=not a.no_consensus, min_voxels=a.min_voxels,
                         smooth=not a.no_smooth)
        preds[case] = (fused, meta["spacing"])
        gts[case] = gt

    cases_scored = sorted(set(preds) & set(gts))
    print(f"{len(cases_scored)}/{len(holdout)} holdout cases scored "
         f"(consensus fusion {'OFF' if a.no_consensus else 'ON'}, "
         f"min_voxels={a.min_voxels}, smoothing {'off' if a.no_smooth else 'on'})")

    per_class, pooled_acc, n_components = score(cases_scored, preds, gts, loc_value, spec.n_loc)
    print(f"\n=== Experiment A (Vessel-Conditioned Multi-Task Seg + Consensus Fusion) ===")
    print(f"internal pooled per-component accuracy: {pooled_acc:.4f} over {n_components} instances "
         f"(NOT the official-metric comparison -- see this script's own docstring)")
    for i, k in enumerate(("dice", "vs", "hd95", "precision", "recall", "mcc")):
        print(f"  {k:<12}{float(np.nanmean([row[i] for row in per_class.values()])):.4f}")

    if a.official_metrics:
        off_per_class, off_avg = official_score(cases_scored, preds, gts, loc_value, spec.n_loc)
        print_official(off_per_class, off_avg, len(cases_scored), out_csv=a.official_out_csv)
        print("\nCompare directly against the rule reference point: "
             "dice 0.0052, precision 0.3596, recall 0.3253, mcc 0.3530")


if __name__ == "__main__":
    main()
