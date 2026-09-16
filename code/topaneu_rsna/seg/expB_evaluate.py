"""
Experiment B, step 3/3: run the trained crop classifier on its held-out fold
(config.EXPB_VAL_FOLD, real predicted instances only), score with both this
pipeline's internal score() and the verified official-metric replica, and
write a predictions CSV in the exact schema build_hybrid_assignment.py
already expects -- so the disagreement analysis against the rule engine
(agreement rate, real-lesion-vs-hallucination split, classifier-correct-
rule-wrong by resolved_by) is a straight reuse of that existing script, not
new code:

    python -m topaneu_rsna.seg.build_hybrid_assignment \\
        logs/task2_rule_instances_final_idx.csv \\
        logs/task2_expB_classifier_predictions.csv \\
        --binary_dataset 304 --folds 0 1 2 3 4

Fair comparison: the rule engine's own headline numbers (0.4064 internal /
dice 0.0052 precision 0.3596 recall 0.3253 mcc 0.3530 official) were computed
over ALL ~417 cases, but this classifier only has predictions for fold 4's
~1/5 of cases -- comparing against the full-cohort number directly would not
be apples-to-apples. This script ALSO recomputes the rule's own score
restricted to exactly the same fold-4 cases (from job 62's
task2_rule_instances_final_idx.csv), so the headline comparison here is a
genuinely matched cohort, with the full-cohort numbers printed alongside
only as additional context.

    python -m topaneu_rsna.seg.expB_evaluate --official_metrics
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio
from topaneu_rsna.seg.assign_location_rule import load_binary_pred_paths, score
from topaneu_rsna.seg.evaluate_official import official_score, print_official
from topaneu_rsna.seg.build_expB_crop_dataset import build_case_to_fold
from topaneu_rsna.seg.expB_train_classifier import CropClassifier, build_model


def run_inference(ckpt_path: Path, expanded_ckpt: Path, dataset304_model_dir: Path,
                  manifest_rows: list[dict], n_classes: int, device):
    model = build_model(expanded_ckpt, dataset304_model_dir, n_classes, device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    preds = []
    with torch.no_grad():
        for r in manifest_rows:
            d = np.load(r["npz_path"])
            crop = torch.from_numpy(d["crop"])[None].to(device)
            probs = torch.softmax(model(crop), dim=1)[0].cpu().numpy()
            pred_id = int(probs.argmax())
            preds.append((r, pred_id, float(probs[pred_id])))
    return preds


def write_predictions_csv(preds, id_to_loc: dict, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "instance_idx", "predicted_class", "predicted_proba", "true_class"])
        for r, pred_id, proba in preds:
            w.writerow([r["case"], r["instance_idx"], id_to_loc[pred_id], f"{proba:.6f}",
                       r["true_class"]])
    print(f"predictions written to {out_csv} (schema matches build_hybrid_assignment.py)")


def paint_and_score(pred_by_case: dict, cases_subset: set, spec, loc_value: dict,
                    label: str, official: bool, official_out_csv=None):
    bin_paths = load_binary_pred_paths(C.DS_ANEURYSM, C.TRAINER_LOC, C.PLANS_RESENC,
                                       folds=[0, 1, 2, 3, 4])
    preds, gts = {}, {}
    for case, bp in bin_paths.items():
        if case not in cases_subset:
            continue
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not gt_p.exists():
            continue
        binmask, meta = uio.read(bp)
        lab, _ = ndimage.label(binmask > 0)
        gt, _ = uio.read(gt_p)
        final = np.zeros(binmask.shape, dtype=np.uint16)
        for instance_idx, pred_name in pred_by_case.get(case, []):
            if not pred_name or pred_name == "background":
                continue
            final[lab == int(instance_idx)] = loc_value[pred_name]
        preds[case] = (final, meta["spacing"])
        gts[case] = gt

    cases_scored = sorted(set(preds) & set(gts))
    per_class, pooled_acc, n = score(cases_scored, preds, gts, loc_value, spec.n_loc)
    print(f"\n=== {label} (n={len(cases_scored)} cases) ===")
    print(f"internal pooled per-component accuracy: {pooled_acc:.4f} over {n} instances")
    for i, k in enumerate(("dice", "vs", "hd95", "precision", "recall", "mcc")):
        print(f"  {k:<12}{float(np.nanmean([row[i] for row in per_class.values()])):.4f}")

    if official:
        off_per_class, off_avg = official_score(cases_scored, preds, gts, loc_value, spec.n_loc)
        print_official(off_per_class, off_avg, len(cases_scored), out_csv=official_out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", type=Path, default=C.EXPB_MANIFEST_CSV)
    ap.add_argument("--ckpt", type=Path, default=C.EXPB_CKPT_DIR / "best.pt")
    ap.add_argument("--expanded_ckpt", type=Path,
                    default=C.WORK / "checkpoints" / "expB_dataset304_fold4_3ch_checkpoint_final.pth")
    ap.add_argument("--dataset304_model_dir", type=Path, default=None)
    ap.add_argument("--rule_instances_csv", type=Path,
                    default=C.LOG_ROOT / "task2_rule_instances_final_idx.csv")
    ap.add_argument("--out_csv", type=Path, default=C.EXPB_PREDICTIONS_CSV)
    ap.add_argument("--official_metrics", action="store_true")
    a = ap.parse_args()

    spec = C.load_labels()
    n_classes = spec.n_loc + 1
    id_to_loc = {i + 1: loc for i, loc in enumerate(spec.locations)}
    id_to_loc[0] = "background"
    loc_value = {loc: i + 1 for i, loc in enumerate(spec.locations)}
    dataset304_model_dir = a.dataset304_model_dir or C.seg_model_dir(C.DS_ANEURYSM, C.TRAINER_LOC)

    with open(a.manifest_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if int(r["fold"]) == C.EXPB_VAL_FOLD]
    val_cases = set(r["case"] for r in val_rows)
    print(f"{len(val_rows)} held-out instances across {len(val_cases)} cases (fold {C.EXPB_VAL_FOLD})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = run_inference(a.ckpt, a.expanded_ckpt, dataset304_model_dir, val_rows, n_classes, device)
    write_predictions_csv(preds, id_to_loc, a.out_csv)

    clf_by_case = {}
    for r, pred_id, _ in preds:
        clf_by_case.setdefault(r["case"], []).append((r["instance_idx"], id_to_loc[pred_id]))

    paint_and_score(clf_by_case, val_cases, spec, loc_value,
                    "Experiment B classifier (held-out fold only)",
                    a.official_metrics, C.LOG_ROOT / "task2_expB_official_metrics.csv")

    with open(a.rule_instances_csv, newline="") as f:
        rule_rows = list(csv.DictReader(f))
    rule_by_case = {}
    for r in rule_rows:
        if r["case"] in val_cases and r["assigned"]:
            rule_by_case.setdefault(r["case"], []).append((r["instance_idx"], r["assigned"]))
    paint_and_score(rule_by_case, val_cases, spec, loc_value,
                    "Rule engine, SAME held-out fold only (fair comparison cohort)",
                    a.official_metrics, C.LOG_ROOT / "task2_rule_foldmatch_official_metrics.csv")

    print("\nFor reference, the rule's FULL-COHORT numbers (all ~417 cases, NOT the "
         "same cohort as above): internal 0.4064 / official dice 0.0052 precision 0.3596 "
         "recall 0.3253 mcc 0.3530")
    print("\nFor the disagreement analysis against the rule engine, run:")
    print(f"  python -m topaneu_rsna.seg.build_hybrid_assignment "
         f"{a.rule_instances_csv} {a.out_csv} --binary_dataset 304 --folds 0 1 2 3 4")


if __name__ == "__main__":
    main()
