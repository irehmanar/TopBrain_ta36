"""
Measurement hygiene, not a rule change: separate "the rule assigned the
wrong location" from "the binary segmenter (Dataset304) hallucinated an
aneurysm that isn't there at all." Both currently count as a false positive
for whatever class an instance got assigned, but only the first is the rule
engine's fault -- a predicted instance whose true_class is background can
never be corrected by any amount of rule sophistication, since there is no
right answer for it to find.

Pure post-hoc analysis of an existing --instances_csv (e.g.
task2_rule_instances_tuned.csv) -- no rerun of assign_location_rule.py
needed, this only reads the CSV job 51/53/54 already wrote.

Reports, from the instance-level `correct` column:
  raw accuracy       correct.mean() over every assigned instance (what the
                      resolution-method breakdown's "N/M assigned" line
                      implicitly mixes background hallucinations into)
  clean accuracy      correct.mean() over instances whose true_class is a
                      real location (background rows excluded) -- the
                      cleanest read of "when the rule had a real target to
                      find, how often did it find the right one"
  contamination       what fraction of all assigned instances are pure
                      background hallucinations, overall and per assigned
                      class -- a class with most of its FPs coming from
                      background rows has a segmentation problem upstream,
                      not a rule problem; a class whose FPs are mostly other
                      *real* locations has a genuine rule confusion to fix.

    python -m topaneu_rsna.seg.filter_background_fp logs/task2_rule_instances_tuned.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_bool(s: str) -> bool | None:
    if s in ("True", "true", "1"):
        return True
    if s in ("False", "false", "0"):
        return False
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instances_csv", type=Path)
    a = ap.parse_args()

    rows = load_rows(a.instances_csv)
    assigned = [r for r in rows if r["assigned"]]
    n_assigned = len(assigned)
    n_bg = sum(1 for r in assigned if r["true_class"] == "background")

    corr = [to_bool(r["correct"]) for r in assigned]
    raw_acc = sum(1 for c in corr if c) / n_assigned if n_assigned else float("nan")

    non_bg = [r for r in assigned if r["true_class"] != "background"]
    clean_corr = [to_bool(r["correct"]) for r in non_bg]
    clean_acc = sum(1 for c in clean_corr if c) / len(non_bg) if non_bg else float("nan")

    print(f"{len(rows)} total instance rows, {n_assigned} assigned a location")
    print(f"  background-hallucination rows (true_class == background): "
         f"{n_bg} ({100 * n_bg / n_assigned:.1f}% of assigned instances)")
    print(f"\n  raw accuracy (background rows count as wrong, current reporting): "
         f"{raw_acc:.4f}")
    print(f"  clean accuracy (background rows excluded -- rule-only): {clean_acc:.4f}")
    print(f"  delta: {clean_acc - raw_acc:+.4f}")

    per_class_bg = Counter()
    per_class_other_fp = Counter()
    per_class_tp = Counter()
    for r in assigned:
        c = r["assigned"]
        if r["true_class"] == "background":
            per_class_bg[c] += 1
        elif r["true_class"] != c:
            per_class_other_fp[c] += 1
        else:
            per_class_tp[c] += 1

    print(f"\n{'class':<32}{'tp':>5}{'fp_other_class':>16}{'fp_background':>16}")
    classes = sorted(set(per_class_bg) | set(per_class_other_fp) | set(per_class_tp))
    for c in classes:
        print(f"{c:<32}{per_class_tp[c]:>5}{per_class_other_fp[c]:>16}"
             f"{per_class_bg[c]:>16}")


if __name__ == "__main__":
    main()
