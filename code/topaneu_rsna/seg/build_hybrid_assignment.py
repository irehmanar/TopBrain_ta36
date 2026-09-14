"""
Experiment 3: does combining the rule and the learned classifier beat either
alone? Joins the rule's own per-instance CSV (assign_location_rule.py's
--instances_csv, keyed by case+instance_idx) with a learned classifier's
predictions CSV (train_learned_assigner.py's --predictions_csv) on
(case, instance_idx) -- an inner join, so the ~27 instances the rule never
even finds a host vessel for (and which build_feature_table.py therefore
never included at all) simply don't appear here; there's no classifier
answer to compare them against in the first place.

Three things, matching Experiment 3's plan:
  1. Agreement analysis: how often rule and classifier agree, and how
     accurate that agreed answer is -- should be the highest-confidence
     bucket of the two, a sanity check on the whole approach.
  2. Disagreement tabulation: of the instances where they disagree, how
     often is the rule right and the classifier wrong, the reverse, or both
     wrong -- broken down by the rule's own resolved_by category, to see
     whether classifier-correct-rule-wrong cases concentrate in the rule's
     acknowledged weak fallback paths (arc_low_sample_fallback, majority)
     rather than being scattered noise.
  3. Two hybrids, both scored with assign_location_rule.py's own score() so
     the comparison uses identical metrics to Experiments 1 and 2:
       - "trivial": always the rule's answer. By construction this is
         identical to the rule alone -- not a real combination, just a
         sanity-check baseline confirming the join/scoring pipeline
         reproduces the known rule number before trying anything smarter.
       - "override" (only built if --override_resolved_by is given): use
         the classifier's answer instead of the rule's, but only when the
         rule's resolved_by is one of the given categories AND the
         classifier's predicted_proba clears --override_proba_threshold.
         Leave --override_resolved_by empty (the default) unless step 2's
         tabulation actually shows a real, concentrated pattern -- a
         blanket override with no such pattern would trade the rule's
         genuine coverage for noise, not improve it.

    python -m topaneu_rsna.seg.build_hybrid_assignment \\
        logs/task2_rule_instances_final.csv logs/task2_classifier_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio
from topaneu_rsna.seg.assign_location_rule import load_binary_pred_paths, score


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def join_tables(rule_rows: list[dict], clf_rows: list[dict]) -> list[dict]:
    clf_by_key = {(r["case"], r["instance_idx"]): r for r in clf_rows}
    joined = []
    for r in rule_rows:
        if not r["assigned"]:
            continue  # rule never found a host vessel -- no classifier counterpart
        c = clf_by_key.get((r["case"], r["instance_idx"]))
        if c is None:
            continue
        joined.append(dict(
            case=r["case"], instance_idx=r["instance_idx"],
            rule_prediction=r["assigned"], resolved_by=r["resolved_by"],
            classifier_prediction=c["predicted_class"],
            classifier_proba=float(c["predicted_proba"]),
            true_class=r["true_class"]))
    return joined


def score_predictions(pred_by_case: dict, spec, loc_value: dict, a, label: str):
    bin_paths = load_binary_pred_paths(a.binary_dataset, a.trainer, a.plans, a.folds)
    preds, gts = {}, {}
    for case, bp in bin_paths.items():
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not gt_p.exists():
            continue
        binmask, meta = uio.read(bp)
        binmask = binmask > 0
        lab, _ = ndimage.label(binmask)
        gt, _ = uio.read(gt_p)
        final = np.zeros(binmask.shape, dtype=np.uint16)
        for instance_idx, pred in pred_by_case.get(case, []):
            if not pred or pred == "background":
                continue
            final[lab == int(instance_idx)] = loc_value[pred]
        preds[case] = (final, meta["spacing"])
        gts[case] = gt

    cases_scored = sorted(set(preds) & set(gts))
    per_class, pooled_acc, n_components = score(cases_scored, preds, gts,
                                                loc_value, spec.n_loc)

    def nanmean(i):
        return float(np.nanmean([row[i] for row in per_class.values()]))

    print(f"\n=== {label} ===")
    print(f"pooled per-component accuracy: {pooled_acc:.4f} over {n_components} instances")
    for i, k in enumerate(("dice", "vs", "hd95", "precision", "recall", "mcc")):
        print(f"  {k:<12}{nanmean(i):.4f}")
    return pooled_acc, {k: nanmean(i) for i, k in
                       enumerate(("dice", "vs", "hd95", "precision", "recall", "mcc"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rule_instances_csv", type=Path)
    ap.add_argument("classifier_predictions_csv", type=Path)
    ap.add_argument("--override_resolved_by", nargs="*", default=[],
                    help="rule resolved_by categories eligible for classifier "
                         "override, e.g. arc_low_sample_fallback majority -- "
                         "leave empty (default) to skip building any override "
                         "and only report the trivial hybrid + disagreement "
                         "tabulation")
    ap.add_argument("--override_proba_threshold", type=float, default=0.5)
    ap.add_argument("--binary_dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out_csv", type=Path,
                    default=C.LOG_ROOT / "task2_hybrid_joined.csv")
    a = ap.parse_args()

    spec = C.load_labels()
    loc_value = {loc: i + 1 for i, loc in enumerate(spec.locations)}

    rule_rows = load_rows(a.rule_instances_csv)
    clf_rows = load_rows(a.classifier_predictions_csv)
    joined = join_tables(rule_rows, clf_rows)
    print(f"{len(joined)} instances joined (rule made an assignment AND has a "
         f"classifier counterpart)")

    agree = [j for j in joined if j["rule_prediction"] == j["classifier_prediction"]]
    disagree = [j for j in joined if j["rule_prediction"] != j["classifier_prediction"]]
    agree_acc = (sum(1 for j in agree if j["rule_prediction"] == j["true_class"]) / len(agree)
                if agree else float("nan"))
    print(f"\n=== agreement ===")
    print(f"  agree: {len(agree)}/{len(joined)} "
         f"({100 * len(agree) / len(joined):.1f}%), accuracy when agreed: {agree_acc:.4f}")

    print(f"\n=== disagreement ({len(disagree)} instances) ===")
    rule_right = sum(1 for j in disagree if j["rule_prediction"] == j["true_class"])
    clf_right = sum(1 for j in disagree if j["classifier_prediction"] == j["true_class"])
    both_wrong = len(disagree) - rule_right - clf_right
    print(f"  rule right, classifier wrong: {rule_right}")
    print(f"  classifier right, rule wrong: {clf_right}")
    print(f"  both wrong: {both_wrong}")

    # The rule always outputs a real location whenever it has a host vessel --
    # it never says "not a lesion" -- so every background-hallucination
    # instance (true_class == "background", ~29% of the whole population per
    # filter_background_fp.py) is an automatic loss for the rule here, while
    # the classifier can score a "win" simply by correctly recognising junk
    # as junk. That's a real, useful skill, but it is NOT location-assignment
    # accuracy, and it plays no part in the official pooled-per-component
    # metric (background instances never correspond to a ground-truth
    # component to be scored against either way). Splitting this out is the
    # only honest way to judge whether the classifier is actually better at
    # naming a real lesion's location, which is the only thing an override
    # into the rule should ever be based on.
    disagree_bg = [j for j in disagree if j["true_class"] == "background"]
    disagree_real = [j for j in disagree if j["true_class"] != "background"]

    def tally(subset):
        r = sum(1 for j in subset if j["rule_prediction"] == j["true_class"])
        c = sum(1 for j in subset if j["classifier_prediction"] == j["true_class"])
        return r, c, len(subset) - r - c

    print(f"\n  -- split by whether true_class is a real location or a "
         f"segmentation hallucination --")
    r, c, w = tally(disagree_real)
    print(f"  REAL lesions ({len(disagree_real)} disagreements): "
         f"rule right {r}, classifier right {c}, both wrong {w}")
    r, c, w = tally(disagree_bg)
    print(f"  BACKGROUND hallucinations ({len(disagree_bg)} disagreements): "
         f"rule right {r} (structurally always 0), classifier right {c} "
         f"(correctly called it junk), both wrong {w}")

    by_resolved = defaultdict(Counter)
    for j in disagree_real:
        by_resolved[j["resolved_by"]]["total"] += 1
        if j["classifier_prediction"] == j["true_class"] and j["rule_prediction"] != j["true_class"]:
            by_resolved[j["resolved_by"]]["classifier_right_rule_wrong"] += 1
    print(f"\n  classifier-correct-rule-wrong cases on REAL lesions only, "
         f"by rule's resolved_by (this, not the unsplit number above, is "
         f"what should inform --override_resolved_by):")
    for rb, counts in sorted(by_resolved.items()):
        print(f"    {rb:<28} {counts['classifier_right_rule_wrong']:>3} / "
             f"{counts['total']:>3} real-lesion disagreements")

    trivial_by_case = defaultdict(list)
    for j in joined:
        trivial_by_case[j["case"]].append((j["instance_idx"], j["rule_prediction"]))
    score_predictions(trivial_by_case, spec, loc_value, a,
                      "trivial hybrid (= rule alone, sanity check)")

    if a.override_resolved_by:
        override_by_case = defaultdict(list)
        for j in joined:
            pred = j["rule_prediction"]
            if (j["resolved_by"] in a.override_resolved_by
                    and j["classifier_proba"] >= a.override_proba_threshold):
                pred = j["classifier_prediction"]
            override_by_case[j["case"]].append((j["instance_idx"], pred))
        score_predictions(override_by_case, spec, loc_value, a,
                          f"targeted-override hybrid "
                          f"(resolved_by in {a.override_resolved_by}, "
                          f"proba>={a.override_proba_threshold})")
    else:
        print("\n--override_resolved_by not given -- skipping the targeted-override "
             "hybrid (see the disagreement tabulation above to decide whether one "
             "is justified)")

    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "instance_idx", "rule_prediction",
                                          "resolved_by", "classifier_prediction",
                                          "classifier_proba", "true_class"])
        w.writeheader()
        w.writerows(joined)
    print(f"\njoined table written to {a.out_csv}")


if __name__ == "__main__":
    main()
