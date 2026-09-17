"""
Experiment Gamma, step 2: disagreement analysis grouped by class,
complementing build_hybrid_assignment.py's own by-resolved_by breakdown
(which only asks "which rule mechanism produced this disagreement" -- this
asks "which classes does the classifier consistently call better than the
rule, regardless of how the rule reached its answer").

Two grouping modes -- these are NOT interchangeable, only one is usable as
a real override condition:

  --group_by classifier_prediction (default, ACTIONABLE)
      Groups by the classifier's OWN predicted class -- a value known at
      prediction time. Answers "whenever the classifier claims class X (and
      disagrees with the rule), how often is it actually right?" -- i.e.
      the classifier's own disagreement-conditional precision on X. A class
      with a high rate here is a legitimate candidate for a real override
      condition (e.g. build_hybrid_assignment.py's existing
      --override_resolved_by, extended to also accept
      --override_classifier_pred_class), because it only ever needs the
      classifier's own output, never the ground truth, to decide whether to
      trust it.

  --group_by true_class (DIAGNOSTIC ONLY, NOT ACTIONABLE)
      Groups by the ground-truth answer instead. This is useful for
      understanding WHERE the pipeline struggles (e.g. "Acom complex
      lesions tend to get misassigned by the rule") but true_class is never
      available at prediction time, so a class flagged here CANNOT be
      turned directly into a deployment-time override condition -- doing so
      would condition the rule's own decision on the answer it is trying to
      predict, which is leakage, not a real classifier signal. (An earlier
      version of this script's own guidance suggested an
      "--override_true_class" follow-up flag under this mode -- that was a
      mistake, corrected here. Use --group_by classifier_prediction for
      anything meant to inform a real override.)

Reads the same joined CSV build_hybrid_assignment.py already writes
(--out_csv, default task2_hybrid_joined.csv) -- no new joining or scoring,
pure tabulation. Background-hallucination rows (true_class == "background")
are excluded throughout: this is about naming a real lesion's location, the
same principle build_hybrid_assignment.py's own REAL-vs-BACKGROUND split
already established.

Only groups with at least --min_n disagreements get a verdict -- below
that, a 100% or 0% rate is not evidence of anything, just too few samples
(same spirit as build_vessel_location_prior.py's own min_arc_samples guard).

    python -m topaneu_rsna.seg.analyze_disagreement_by_class logs/task2_hybrid_joined.csv
    python -m topaneu_rsna.seg.analyze_disagreement_by_class logs/task2_hybrid_joined.csv --group_by true_class
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("joined_csv", type=Path)
    ap.add_argument("--group_by", choices=["classifier_prediction", "true_class"],
                    default="classifier_prediction",
                    help="classifier_prediction (default): actionable, groups by "
                         "what the classifier itself predicted. true_class: "
                         "diagnostic only, NOT usable as a real override condition "
                         "-- see this script's own module docstring.")
    ap.add_argument("--min_n", type=int, default=3,
                    help="minimum disagreements on a group before its win rate "
                         "is reported as a verdict rather than 'too few to trust'")
    ap.add_argument("--flag_threshold", type=float, default=0.6,
                    help="classifier-right-rule-wrong rate at/above which a "
                         "group is flagged as a candidate for an override")
    a = ap.parse_args()

    rows = load_rows(a.joined_csv)
    real = [r for r in rows if r["true_class"] != "background"]
    disagree = [r for r in real if r["rule_prediction"] != r["classifier_prediction"]]
    print(f"{len(rows)} joined instances, {len(real)} on real lesions, "
          f"{len(disagree)} real-lesion disagreements")
    print(f"grouping by: {a.group_by} "
          f"({'ACTIONABLE -- known at prediction time' if a.group_by == 'classifier_prediction' else 'DIAGNOSTIC ONLY -- true_class is never known at prediction time, do NOT build an override on this'})")

    by_group = defaultdict(lambda: {"rule_right": 0, "clf_right": 0, "both_wrong": 0})
    for r in disagree:
        key = r[a.group_by]
        if r["rule_prediction"] == r["true_class"]:
            by_group[key]["rule_right"] += 1
        elif r["classifier_prediction"] == r["true_class"]:
            by_group[key]["clf_right"] += 1
        else:
            by_group[key]["both_wrong"] += 1

    rows_out = []
    for key, c in by_group.items():
        n = c["rule_right"] + c["clf_right"] + c["both_wrong"]
        clf_rate = c["clf_right"] / n if n else float("nan")
        rows_out.append((key, n, c["rule_right"], c["clf_right"], c["both_wrong"], clf_rate))

    label = a.group_by
    print(f"\n{label:<32}{'n':>4}{'rule_right':>11}{'clf_right':>10}"
          f"{'both_wrong':>11}{'clf_rate':>9}  verdict")
    flagged = []
    for key, n, rr, cr, bw, rate in sorted(rows_out, key=lambda r: -r[1]):
        if n < a.min_n:
            verdict = "too few to trust"
        elif rate >= a.flag_threshold:
            verdict = "CLASSIFIER ADVANTAGE"
            flagged.append((key, n, rate))
        elif rate <= (1 - a.flag_threshold):
            verdict = "rule advantage"
        else:
            verdict = "no clear pattern"
        print(f"{key:<32}{n:>4}{rr:>11}{cr:>10}{bw:>11}{rate:>9.3f}  {verdict}")

    print(f"\n{len(flagged)} {a.group_by} value(s) with >= {a.min_n} disagreements "
          f"and a classifier win rate >= {a.flag_threshold}:")
    if flagged:
        for key, n, rate in flagged:
            print(f"  {key}: {rate:.3f} over {n} disagreements")
        if a.group_by == "classifier_prediction":
            print(f"\nThese ARE legitimate candidates for a real override in "
                  f"build_hybrid_assignment.py (e.g. an "
                  f"--override_classifier_pred_class flag: trust the classifier "
                  f"whenever its own predicted class is one of these) -- unlike "
                  f"the resolved_by-conditioned override, this only ever needs "
                  f"the classifier's own output, never the ground truth.")
        else:
            print(f"\nDIAGNOSTIC ONLY. Re-run with "
                  f"--group_by classifier_prediction to see whether any of these "
                  f"true classes correspond to a classifier_prediction value the "
                  f"classifier is also confident and correct about when it makes "
                  f"that specific claim -- that version is the one safe to act on.")
    else:
        print("  none -- no group shows a clean, high-confidence classifier "
              "advantage beyond what arc_low_sample_fallback (already "
              "incorporated) already captures. This is a valid, useful "
              "negative result: it means the current feature set/assignment "
              "logic split (Experiment Gamma step 3's question) rather than "
              "an unexploited disagreement pattern is the likelier next "
              "bottleneck.")


if __name__ == "__main__":
    main()
