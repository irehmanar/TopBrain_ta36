"""
Experiment 2: does the declared rule's ~0.40 pooled accuracy actually need
declared anatomy, or would a plain learned classifier over the same
geometric inputs (build_feature_table.py's output) do just as well? This is
exactly Paper 1's Sect. 3.5 / Table 2 comparison: a learned ranker trained on
the rule's own features, evaluated leave-one-case-out so no two instances
from the same case ever cross the train/test split (matches Paper 1's
protocol, and is the only sane choice here given how many of the 52 classes
have 0-2 total examples in the whole cohort -- a fixed split would starve
whichever fold didn't get the rare ones).

The classifier is trained to predict `true_class` directly (including
"background", since a predicted instance that's really a segmentation
false alarm is part of the same population the rule itself has to handle --
excluding those rows would make the comparison easier than what the rule
actually faces). Predictions are painted back into per-case masks and scored
with assign_location_rule.py's own `score()` function against the same
ground truth, so the comparison is on identical metrics, not just a
different notion of "accuracy."

The point of this experiment isn't the headline pooled-accuracy number by
itself -- it's whether the learned classifier matches that number while
covering far fewer of the 52 classes at all (Paper 1's actual finding: a
classifier can never predict a class it never saw during training, while the
declared rule needs zero training data for any of them).

    python -m topaneu_rsna.seg.train_learned_assigner logs/task2_feature_table.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio
from topaneu_rsna.seg.assign_location_rule import load_binary_pred_paths, score

CATEGORICAL = ["vessel", "laterality", "resolved_by"]
NUMERIC = ["size", "host_dist_mm", "arc_fraction", "is_orientable", "junction_dist_mm"]
SENTINEL = -1.0


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(s: str) -> float:
    return SENTINEL if s == "" else float(s)


def build_matrices(rows: list[dict], spec, encoder: OneHotEncoder):
    cat = encoder.transform([[r[c] for c in CATEGORICAL] for r in rows])
    num = np.array([[to_float(r[c]) for c in NUMERIC] for r in rows], dtype=np.float64)
    X = np.hstack([cat, num])

    label_space = spec.locations + ["background"]
    label_to_idx = {name: i for i, name in enumerate(label_space)}
    y = np.array([label_to_idx[r["true_class"]] for r in rows], dtype=np.int64)
    return X, y, label_space


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feature_csv", type=Path)
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_voxels", type=int, default=3)
    ap.add_argument("--binary_dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rule_csv", type=Path, default=None,
                    help="optional rule per-class CSV (e.g. "
                         "task2_rule_assignment_final.csv) to print the "
                         "classes-ever-predicted comparison directly")
    a = ap.parse_args()

    spec = C.load_labels()
    rows = load_rows(a.feature_csv)
    print(f"{len(rows)} instances loaded from {a.feature_csv}")

    # Category vocabulary is declared, not fit from data -- vessel names and
    # the fixed small set of laterality/resolved_by values are known in
    # advance (labels.json, the rule's own resolved_by categories), so
    # encoding it costs no leakage and doesn't need per-fold refitting.
    vessel_vocab = spec.vessels
    laterality_vocab = ["R", "L", "none"]
    resolved_by_vocab = ["single", "junction", "arc", "arc_low_sample_fallback", "majority"]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2 used `sparse` instead
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoder.fit([[v, "none", "single"] for v in vessel_vocab]
               + [["R-VA", lat, "single"] for lat in laterality_vocab]
               + [["R-VA", "R", rb] for rb in resolved_by_vocab])

    X, y, label_space = build_matrices(rows, spec, encoder)
    cases = np.array([r["case"] for r in rows])
    unique_cases = sorted(set(cases))
    print(f"{len(unique_cases)} unique cases, leave-one-case-out")

    pred_idx = np.full(len(rows), -1, dtype=np.int64)
    for case in tqdm(unique_cases, desc="LOCO"):
        test_mask = cases == case
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            continue
        clf = RandomForestClassifier(n_estimators=a.n_estimators,
                                     class_weight="balanced_subsample",
                                     random_state=a.seed, n_jobs=-1)
        clf.fit(X[train_mask], y[train_mask])
        pred_idx[test_mask] = clf.predict(X[test_mask])

    predicted_label = [label_space[i] for i in pred_idx]
    n_ever_predicted = len({p for p in predicted_label if p != "background"})
    print(f"\nclassifier predicted {n_ever_predicted} of {spec.n_loc} location "
         f"classes at least once across all held-out instances")

    # Repaint per-case masks from the classifier's own predictions and score
    # with the exact same function the rule itself is scored with.
    by_case = defaultdict(list)
    for r, pred in zip(rows, predicted_label):
        by_case[r["case"]].append((int(r["instance_idx"]), pred))

    loc_value = {loc: i + 1 for i, loc in enumerate(spec.locations)}
    bin_paths = load_binary_pred_paths(a.binary_dataset, a.trainer, a.plans, a.folds)

    preds, gts = {}, {}
    for case, instance_preds in tqdm(by_case.items(), desc="repainting"):
        bp = bin_paths.get(case)
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if bp is None or not gt_p.exists():
            continue
        binmask, meta = uio.read(bp)
        binmask = binmask > 0
        lab, _ = ndimage.label(binmask)
        gt, _ = uio.read(gt_p)

        final = np.zeros(binmask.shape, dtype=np.uint16)
        for instance_idx, pred in instance_preds:
            if pred == "background":
                continue
            final[lab == instance_idx] = loc_value[pred]
        preds[case] = (final, meta["spacing"])
        gts[case] = gt

    cases_scored = sorted(set(preds) & set(gts))
    per_class, pooled_acc, n_components = score(cases_scored, preds, gts,
                                                loc_value, spec.n_loc)

    def nanmean(idx):
        return float(np.nanmean([row[idx] for row in per_class.values()]))

    print(f"\n=== learned classifier (leave-one-case-out) ===")
    print(f"pooled per-component accuracy: {pooled_acc:.4f} over {n_components} instances")
    for i, k in enumerate(("dice", "vs", "hd95", "precision", "recall", "mcc")):
        print(f"  {k:<12}{nanmean(i):.4f}")
    print(f"  classes ever predicted: {n_ever_predicted} / {spec.n_loc}")

    if a.rule_csv is not None and a.rule_csv.exists():
        with open(a.rule_csv, newline="") as f:
            rule_rows = list(csv.DictReader(f))
        rule_ever = sum(1 for r in rule_rows
                        if int(r["tp"]) + int(r["fp"]) > 0)
        rule_recall = np.nanmean([float(r["recall"]) if r["recall"] != "nan" else np.nan
                                  for r in rule_rows])
        print(f"\n=== comparison against {a.rule_csv.name} ===")
        print(f"  rule classes ever predicted: {rule_ever} / {spec.n_loc}")
        print(f"  rule macro recall: {rule_recall:.4f}")
        print(f"  classifier macro recall: {nanmean(4):.4f}")
        print(f"  classifier classes ever predicted: {n_ever_predicted} / {spec.n_loc}")


if __name__ == "__main__":
    main()
