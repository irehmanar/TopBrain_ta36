"""
Experiment Alpha, step 2: score any of the binary prediction sets against
Dataset304's ground truth -- baseline (existing fold_*/validation/, TTA on,
own fold only), notta, or ensemble5 (see expAlpha_predict_variants.py).

Reports, per variant:
  - pooled (voxel-summed) Dice, IoU, precision, recall -- comparable to this
    pipeline's existing ~0.60 pooled-Dice number
  - per-case Dice: mean, median, std, 25th/75th percentile, min, max, N cases
    (not just the global mean, per the task's own requirement)

Appends one row per variant to EXPALPHA_RESULTS_CSV so the three steps build
into a single comparison table rather than three disconnected runs.

    python -m topaneu_rsna.seg.expAlpha_evaluate_variants --variant baseline
    python -m topaneu_rsna.seg.expAlpha_evaluate_variants --variant notta
    python -m topaneu_rsna.seg.expAlpha_evaluate_variants --variant ensemble5
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.seg.assign_location_rule import load_binary_pred_paths
from topaneu_rsna.utils import io as uio


def pred_paths_for(variant: str, dataset_id: int, trainer: str, plans: str) -> dict:
    if variant == "baseline":
        return load_binary_pred_paths(dataset_id, trainer, plans, folds=range(5))
    pred_dir = C.EXPALPHA_PRED_DIR / variant
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"{pred_dir} missing. Run expAlpha_predict_variants.py --variant {variant} first.")
    return {case: pred_dir / f"{case}{C.LABEL_SUFFIX}"
            for case in uio.list_cases(pred_dir, C.LABEL_SUFFIX)}


def evaluate(pred_by_case: dict, gt_dir: Path):
    """Binary (single foreground class) pooled + per-case metrics."""
    inter = pred_sum = gt_sum = 0
    tp = fp = fn = tn = 0
    per_case_dice = []

    n_shape_mismatch = 0
    for case, pp in tqdm(sorted(pred_by_case.items()), desc="scoring"):
        gp = gt_dir / f"{case}{C.LABEL_SUFFIX}"
        if not gp.exists():
            continue
        pred, _ = uio.read(pp)
        gt, _ = uio.read(gp)
        if pred.shape != gt.shape:
            # Seen in practice: a handful of cases in Dataset304's raw
            # labelsTr/imagesTr had a stale/corrupted file from an unrelated
            # ROI-cropped experiment sitting where a native-shape whole-head
            # file should be. Skip and warn rather than crash the whole
            # evaluation over one bad case -- but a shape mismatch is never
            # silently ignorable, so it's still counted and reported.
            print(f"[WARNING] shape mismatch for {case}: pred={pred.shape} "
                 f"gt={gt.shape} -- skipping this case, investigate its raw "
                 f"files (see jobs/33_expAlpha_binary_postproc's known issue).")
            n_shape_mismatch += 1
            continue
        pm, gm = pred > 0, gt > 0
        pn, gn = int(pm.sum()), int(gm.sum())
        it = int((pm & gm).sum())

        inter += it; pred_sum += pn; gt_sum += gn
        if gn > 0 and it > 0:
            tp += 1
        elif gn > 0 and it == 0:
            fn += 1
        elif gn == 0 and pn > 0:
            fp += 1
        else:
            tn += 1

        # per-case Dice; a case with no GT and no prediction is a perfect
        # (trivial) match, not undefined -- scored as 1.0, consistent with
        # how it's already handled as a TN above rather than excluded.
        if pn + gn == 0:
            per_case_dice.append(1.0)
        else:
            per_case_dice.append(2 * it / (pn + gn))

    if n_shape_mismatch:
        print(f"[WARNING] {n_shape_mismatch} case(s) skipped for a pred/gt shape "
             f"mismatch -- see warnings above. Metrics below are computed over "
             f"the remaining cases only, NOT silently padded or assumed correct.")

    per_case_dice = np.asarray(per_case_dice)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled_dice = 2 * inter / (pred_sum + gt_sum) if (pred_sum + gt_sum) else float("nan")
        pooled_iou = inter / (pred_sum + gt_sum - inter) if (pred_sum + gt_sum - inter) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")

    return dict(
        n_shape_mismatch=n_shape_mismatch,
        n_cases=len(per_case_dice),
        pooled_dice=pooled_dice, pooled_iou=pooled_iou,
        precision=precision, recall=recall,
        tp=tp, fp=fp, fn=fn, tn=tn,
        case_dice_mean=float(per_case_dice.mean()),
        case_dice_median=float(np.median(per_case_dice)),
        case_dice_std=float(per_case_dice.std()),
        case_dice_p25=float(np.percentile(per_case_dice, 25)),
        case_dice_p75=float(np.percentile(per_case_dice, 75)),
        case_dice_min=float(per_case_dice.min()),
        case_dice_max=float(per_case_dice.max()),
    )


def append_result_row(variant: str, m: dict, csv_path: Path, note: str = ""):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    fields = ["variant", "n_cases", "pooled_dice", "pooled_iou", "precision", "recall",
              "tp", "fp", "fn", "tn", "case_dice_mean", "case_dice_median",
              "case_dice_std", "case_dice_p25", "case_dice_p75", "case_dice_min",
              "case_dice_max", "note"]
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            w.writeheader()
        row = dict(m); row["variant"] = variant; row["note"] = note
        w.writerow({k: row.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "notta", "ensemble5"], required=True)
    ap.add_argument("--dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--out", type=Path, default=C.EXPALPHA_RESULTS_CSV)
    a = ap.parse_args()

    gt_dir = C.nnUNet_raw / f"Dataset{a.dataset:03d}_{C.DS_NAMES[a.dataset]}" / "labelsTr"
    pred_by_case = pred_paths_for(a.variant, a.dataset, a.trainer, a.plans)
    print(f"[expAlpha] variant={a.variant}: {len(pred_by_case)} predicted cases")

    m = evaluate(pred_by_case, gt_dir)

    note = ""
    if a.variant == "ensemble5":
        note = ("LEAKED/OPTIMISTIC: every case here was training data for 4 of the "
                "5 ensembled folds -- not a valid unseen-data estimate, see "
                "config.py EXPALPHA_PRED_DIR docstring.")

    append_result_row(a.variant, m, a.out, note)

    print(f"\n{'metric':<18}{'value':>10}")
    for k in ("n_cases", "pooled_dice", "pooled_iou", "precision", "recall",
              "case_dice_mean", "case_dice_median", "case_dice_std",
              "case_dice_p25", "case_dice_p75"):
        v = m[k]
        print(f"{k:<18}{v:>10.4f}" if isinstance(v, float) else f"{k:<18}{v:>10d}")
    if note:
        print(f"\n[CAVEAT] {note}")
    print(f"\nrow appended to {a.out}")


if __name__ == "__main__":
    main()
