"""
Experiment Gamma, step 2: disagreement analysis grouped by TRUE CLASS,
complementing build_hybrid_assignment.py's own by-resolved_by breakdown
(which only asks "which rule mechanism produced this disagreement" -- this
asks "which of the 52 locations does the classifier consistently call
better than the rule, regardless of how the rule reached its answer").

The existing targeted hybrid (job 65, +0.022 pooled accuracy) only trusts
the classifier inside one rule mechanism (arc_low_sample_fallback, 67%
classifier-right-rule-wrong). This checks whether any *location class*
shows a similarly clean, high-confidence pattern across its disagreements
overall -- a class-conditioned override (e.g. "trust the classifier
whenever true_class would be X") is a structurally different lever than a
resolved_by-conditioned one, and the two aren't mutually exclusive.

Reads the same joined CSV build_hybrid_assignment.py already writes
(--out_csv, default task2_hybrid_joined.csv) -- no new joining or scoring,
pure tabulation. Background-hallucination rows (true_class == "background")
are excluded throughout: a class-conditioned override is about naming a
real lesion's location, the same principle build_hybrid_assignment.py's own
REAL-vs-BACKGROUND split already established.

Only classes with at least --min_n disagreements are reported with a
verdict -- below that, a 100% or 0% win rate is not evidence of anything,
just too few samples (same spirit as build_vessel_location_prior.py's own
min_arc_samples guard).

    python -m topaneu_rsna.seg.analyze_disagreement_by_class logs/task2_hybrid_joined.csv
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
    ap.add_argument("--min_n", type=int, default=3,
                    help="minimum disagreements on a class before its win rate "
                         "is reported as a verdict rather than 'too few to trust'")
    ap.add_argument("--flag_threshold", type=float, default=0.6,
                    help="classifier-right-rule-wrong rate at/above which a "
                         "class is flagged as a candidate for a class-"
                         "conditioned override")
    a = ap.parse_args()

    rows = load_rows(a.joined_csv)
    real = [r for r in rows if r["true_class"] != "background"]
    disagree = [r for r in real if r["rule_prediction"] != r["classifier_prediction"]]
    print(f"{len(rows)} joined instances, {len(real)} on real lesions, "
          f"{len(disagree)} real-lesion disagreements")

    by_class = defaultdict(lambda: {"rule_right": 0, "clf_right": 0, "both_wrong": 0})
    for r in disagree:
        cls = r["true_class"]
        if r["rule_prediction"] == cls:
            by_class[cls]["rule_right"] += 1
        elif r["classifier_prediction"] == cls:
            by_class[cls]["clf_right"] += 1
        else:
            by_class[cls]["both_wrong"] += 1

    rows_out = []
    for cls, c in by_class.items():
        n = c["rule_right"] + c["clf_right"] + c["both_wrong"]
        clf_rate = c["clf_right"] / n if n else float("nan")
        rows_out.append((cls, n, c["rule_right"], c["clf_right"], c["both_wrong"], clf_rate))

    print(f"\n{'true_class':<32}{'n':>4}{'rule_right':>11}{'clf_right':>10}"
          f"{'both_wrong':>11}{'clf_rate':>9}  verdict")
    flagged = []
    for cls, n, rr, cr, bw, rate in sorted(rows_out, key=lambda r: -r[1]):
        if n < a.min_n:
            verdict = "too few to trust"
        elif rate >= a.flag_threshold:
            verdict = "CLASSIFIER ADVANTAGE"
            flagged.append((cls, n, rate))
        elif rate <= (1 - a.flag_threshold):
            verdict = "rule advantage"
        else:
            verdict = "no clear pattern"
        print(f"{cls:<32}{n:>4}{rr:>11}{cr:>10}{bw:>11}{rate:>9.3f}  {verdict}")

    print(f"\n{len(flagged)} class(es) with >= {a.min_n} disagreements and a "
          f"classifier win rate >= {a.flag_threshold}:")
    if flagged:
        for cls, n, rate in flagged:
            print(f"  {cls}: {rate:.3f} over {n} disagreements")
        print(f"\nThese are candidates for a class-conditioned override "
              f"alongside the existing resolved_by-conditioned one in "
              f"build_hybrid_assignment.py -- that script only conditions on "
              f"resolved_by today, so acting on this would need a small "
              f"follow-up (e.g. an --override_true_class-style flag), not yet "
              f"built since it should only be added if this list is real and "
              f"non-empty, not preemptively.")
    else:
        print("  none -- no class shows a clean, high-confidence classifier "
              "advantage beyond what arc_low_sample_fallback (already "
              "incorporated) already captures. This is a valid, useful "
              "negative result: it means the current feature set/assignment "
              "logic split (Experiment Gamma step 3's question) rather than "
              "an unexploited disagreement pattern is the likelier next "
              "bottleneck.")


if __name__ == "__main__":
    main()
