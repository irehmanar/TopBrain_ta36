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

`--weighting capped` (the default) applies rarity-aware per-instance sample
weights within each LOCO training fold: for class c with nc positive
instances among N total training instances in that fold, wc =
min(20, max(1, (N-nc)/nc)); wc = 1 for a class absent from that fold (there's
nothing to weight). This is a different, more specific rebalancing than
`--weighting balanced` (sklearn's built-in `class_weight="balanced_subsample"`,
inverse-frequency with no cap) -- pass `balanced` for a quick check of that
simpler option, or `none` for an unweighted control.

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


def capped_sample_weights(y_train: np.ndarray) -> np.ndarray:
    """wc = min(20, max(1, (N - nc) / nc)) per class, applied per-instance --
    computed fresh inside each LOCO fold (N and nc are fold-local), never
    from the full dataset, so no leakage of held-out-case class frequencies
    into the training weights."""
    n = len(y_train)
    classes, counts = np.unique(y_train, return_counts=True)
    w_by_class = {c: min(20.0, max(1.0, (n - nc) / nc)) for c, nc in zip(classes, counts)}
    return np.array([w_by_class[c] for c in y_train], dtype=np.float64)


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
    ap.add_argument("--weighting", choices=["none", "balanced", "capped"],
                    default="capped",
                    help="none: unweighted control. balanced: sklearn's "
                         "built-in class_weight='balanced_subsample' "
                         "(uncapped inverse frequency) -- a quick check "
                         "before the hand-rolled formula. capped: the "
                         "min(20, max(1,(N-nc)/nc)) per-instance sample "
                         "weight, computed fresh per LOCO fold (default).")
    ap.add_argument("--predictions_csv", type=Path, default=None,
                    help="optional per-instance prediction dump (case, "
                         "instance_idx, vessel, true_class, "
                         "predicted_class, predicted_proba) -- needed to "
                         "join against the rule's own instances CSV for "
                         "the Experiment 3 hybrid comparison")
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

    class_weight_arg = "balanced_subsample" if a.weighting == "balanced" else None
    pred_idx = np.full(len(rows), -1, dtype=np.int64)
    pred_proba = np.zeros(len(rows), dtype=np.float64)
    for case in tqdm(unique_cases, desc="LOCO"):
        test_mask = cases == case
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            continue
        clf = RandomForestClassifier(n_estimators=a.n_estimators,
                                     class_weight=class_weight_arg,
                                     random_state=a.seed, n_jobs=-1)
        sample_weight = (capped_sample_weights(y[train_mask])
                        if a.weighting == "capped" else None)
        clf.fit(X[train_mask], y[train_mask], sample_weight=sample_weight)
        proba = clf.predict_proba(X[test_mask])
        local_pred = np.argmax(proba, axis=1)
        pred_idx[test_mask] = clf.classes_[local_pred]
        pred_proba[test_mask] = proba[np.arange(len(local_pred)), local_pred]

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

    # Score every held-out case with a binary prediction, not just ones with
    # >=1 feature-table row -- a case where Dataset304 found nothing still
    # has to count its real lesions (if any) as misses, exactly like the
    # rule's own evaluation does. Job 60 skipped these (by_case only had
    # entries for cases with instances), understating the denominator
    # (376 vs. the rule's 406) and mildly flattering the classifier's numbers.
    preds, gts = {}, {}
    for case, bp in tqdm(bin_paths.items(), desc="repainting"):
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not gt_p.exists():
            continue
        binmask, meta = uio.read(bp)
        binmask = binmask > 0
        lab, _ = ndimage.label(binmask)
        gt, _ = uio.read(gt_p)

        final = np.zeros(binmask.shape, dtype=np.uint16)
        for instance_idx, pred in by_case.get(case, []):
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

    if a.predictions_csv is not None:
        a.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.predictions_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "instance_idx", "vessel", "true_class",
                       "predicted_class", "predicted_proba"])
            for r, pred, proba in zip(rows, predicted_label, pred_proba):
                w.writerow([r["case"], r["instance_idx"], r["vessel"],
                           r["true_class"], pred, proba])
        print(f"\nper-instance predictions written to {a.predictions_csv}")


if __name__ == "__main__":
    main()
