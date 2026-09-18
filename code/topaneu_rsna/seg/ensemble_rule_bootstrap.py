"""
Experiment Gamma ensemble: bootstrap-vote the declared rule, still fully
rule-based (no learned model anywhere) -- maximizing pooled accuracy via
variance reduction rather than a new heuristic.

Motivation, grounded in this pipeline's own history: jobs 18-21 repeatedly
showed the arc-fraction/majority prior tables (vessel_location_prior.json)
are sensitive to exactly which training cases happen to land in them --
job 19's own diagnosis found a single noisy instance could flip an entire
bucket boundary ("L-4.3 A2"'s false positives went 0 -> 17 as its boundary
shifted), and the min_arc_samples guard only partially addresses this (it
excludes a bucket from ever being RETURNED, it doesn't reduce the
boundary's own sensitivity to which cases were sampled). Bootstrap
aggregation is the standard remedy for exactly this kind of estimator
variance: build the SAME prior computation (build_vessel_location_prior.py,
completely unchanged, no training/gradients involved -- still a declared,
non-learned cohort statistic) on N resamples of the training cases, run the
SAME rule-resolution logic under each, and majority-vote the final answer
per lesion instance.

Efficient by construction: geometry (host vessel, junction contact-patch
distances, raw arc-length fraction, side-reconciled junction candidate) is
identical across every bootstrap of the SAME instance -- only the prior's
bucket boundaries and majority table vary. seg/cache_shared_vessel_geometry.py
computes the expensive geometric half exactly once; this script only redoes
the cheap bucket-lookup half (arc_bucket_decision, pure dict/array ops) per
bootstrap prior, so N bootstraps cost barely more than one full rule pass,
not N times as much.

Prerequisites:
  1. python -m topaneu_rsna.seg.cache_shared_vessel_geometry
  2. for k in range(N): python -m topaneu_rsna.seg.build_vessel_location_prior \\
         --bootstrap_seed k --out <prior_dir>/prior_boot{k}.json

    python -m topaneu_rsna.seg.ensemble_rule_bootstrap --prior_dir <dir> --n_bootstrap 15
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio
from topaneu_rsna.seg.assign_location_rule import (
    arc_bucket_decision, build_vessel_to_locations, load_binary_pred_paths,
    oracle_binary_pred_paths, score)


def load_cache(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def decide_one(row: dict, prior: dict, vessel_to_locations: dict,
               junction_override_mm: float, arc_ambiguous_margin: float
               ) -> tuple[str | None, str]:
    """Replays resolve_shared_vessel's decision logic for one cached
    instance under one bootstrap prior, using only the prior-independent
    geometry already in `row` -- no skeleton/contact-patch recomputation.
    Returns (location_or_None, resolved_by), resolved_by matching
    assign_location_rule.py's own categories (single/junction/arc/
    arc_low_sample_fallback/majority) -- needed so a downstream
    build_hybrid_assignment.py run can condition an override on it exactly
    like it already does for the single-run rule."""
    if not row["vessel"]:
        return None, "unassigned"
    if int(row["n_locs"]) <= 1:
        return row["single_location"] or None, "single"

    vessel_name = row["vessel"]
    locs = vessel_to_locations.get(vessel_name, [])
    j_loc = row["junction_best_loc"] or None
    j_dist = float(row["junction_best_dist_mm"]) if row["junction_best_dist_mm"] else np.inf
    frac = row["arc_fraction"]

    if frac:
        arc_info = prior.get("arc", {}).get(vessel_name)
        a_loc, a_ambiguous, a_low_sample = arc_bucket_decision(
            float(frac), arc_info, arc_ambiguous_margin)
    else:
        a_loc, a_ambiguous, a_low_sample = None, False, False

    if j_loc is not None and (a_loc is None or (j_dist <= junction_override_mm and a_ambiguous)):
        return j_loc, "junction"
    if a_loc is not None:
        return a_loc, "arc"
    if a_low_sample:
        return prior["majority"].get(vessel_name, locs[0] if locs else None), "arc_low_sample_fallback"
    return prior["majority"].get(vessel_name, locs[0] if locs else None), "majority"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry_cache", type=Path,
                    default=C.LOG_ROOT / "task2_shared_vessel_geometry_cache.csv")
    ap.add_argument("--prior_dir", type=Path, required=True,
                    help="directory containing prior_boot0.json .. prior_boot{N-1}.json, "
                         "written by N separate build_vessel_location_prior.py "
                         "--bootstrap_seed runs")
    ap.add_argument("--n_bootstrap", type=int, required=True)
    ap.add_argument("--junction_override_mm", type=float, default=1.5)
    ap.add_argument("--arc_ambiguous_margin", type=float, default=0.03)
    ap.add_argument("--binary_dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--oracle_binary", action="store_true")
    ap.add_argument("--binary_pred_dir", type=Path, default=None,
                    help="MUST match whatever binary source cache_shared_vessel_geometry.py "
                         "was run with to build --geometry_cache, or instance_idx numbering "
                         "won't line up -- see oracle_binary_pred_paths()'s own consistency "
                         "warning for why this matters. Overrides --oracle_binary/"
                         "--binary_dataset/--folds when given.")
    ap.add_argument("--out_csv", type=Path,
                    default=C.LOG_ROOT / "task2_rule_assignment_ensemble.csv")
    ap.add_argument("--votes_csv", type=Path, default=None,
                    help="optional: dump every instance's per-bootstrap votes and "
                         "the winning majority answer, for auditing close calls")
    ap.add_argument("--instances_csv", type=Path, default=None,
                    help="optional: write a build_hybrid_assignment.py-compatible "
                         "per-instance CSV (case, instance_idx, vessel, size, "
                         "host_dist_mm, resolved_by, extra_dist_mm, assigned, "
                         "true_class, correct) so this bootstrap-ensemble rule's "
                         "own output can be combined with a learned classifier's "
                         "predictions the same way job 65 did for the single-run rule")
    a = ap.parse_args()

    spec = C.load_labels()
    vessel_to_locations = build_vessel_to_locations(spec)
    loc_value = {loc: i + 1 for i, loc in enumerate(spec.locations)}

    priors = []
    for k in range(a.n_bootstrap):
        p = a.prior_dir / f"prior_boot{k}.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- generate all {a.n_bootstrap} "
                                    f"bootstrap priors first (see this script's docstring)")
        priors.append(json.loads(p.read_text()))
    print(f"loaded {len(priors)} bootstrap priors from {a.prior_dir}")

    rows = load_cache(a.geometry_cache)
    print(f"{len(rows)} cached instances from {a.geometry_cache}")

    votes_out, instances_out = [], []
    final_by_key: dict[tuple, str | None] = {}
    for row in tqdm(rows, desc="voting"):
        decisions = [decide_one(row, p, vessel_to_locations, a.junction_override_mm,
                                a.arc_ambiguous_margin) for p in priors]
        locations = [loc for loc, _ in decisions]
        tally = Counter(v for v in locations if v is not None)
        winner = tally.most_common(1)[0][0] if tally else None
        final_by_key[(row["case"], int(row["instance_idx"]))] = winner

        # resolved_by for the winning LOCATION: the most common resolution
        # path among just the bootstrap replicates that actually voted for
        # it -- the most representative single tag for a majority answer
        # that different replicates may have reached via different paths.
        winner_resolved_by = (Counter(rb for loc, rb in decisions if loc == winner)
                              .most_common(1)[0][0] if winner is not None else "unassigned")

        if a.votes_csv is not None:
            votes_out.append(dict(case=row["case"], instance_idx=row["instance_idx"],
                                  true_class=row["true_class"], winner=winner or "",
                                  n_votes_for_winner=tally.get(winner, 0) if winner else 0,
                                  votes=";".join(v or "None" for v in locations)))
        if a.instances_csv is not None:
            instances_out.append(dict(
                case=row["case"], instance_idx=row["instance_idx"], vessel=row["vessel"],
                size=row["size"], host_dist_mm=row["host_dist_mm"],
                resolved_by=winner_resolved_by, extra_dist_mm="",
                assigned=(winner or ""), true_class=row["true_class"],
                correct=(winner == row["true_class"] if winner is not None else False)))

    if a.votes_csv is not None:
        a.votes_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.votes_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "instance_idx", "true_class",
                                              "winner", "n_votes_for_winner", "votes"])
            w.writeheader()
            w.writerows(votes_out)
        print(f"per-instance votes written to {a.votes_csv}")

    if a.instances_csv is not None:
        a.instances_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.instances_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "instance_idx", "vessel", "size",
                                              "host_dist_mm", "resolved_by", "extra_dist_mm",
                                              "assigned", "true_class", "correct"])
            w.writeheader()
            w.writerows(instances_out)
        print(f"hybrid-compatible per-instance CSV written to {a.instances_csv}")

    # Rebuild final per-case masks (needed for score()'s Dice/VS/HD95, not
    # just the pooled-component accuracy) -- MUST use the same binary source
    # cache_shared_vessel_geometry.py was built from, or instance_idx numbering
    # won't line up (same requirement oracle_binary_pred_paths() itself warns
    # about elsewhere in this codebase).
    if a.binary_pred_dir is not None:
        bin_paths = {case: a.binary_pred_dir / f"{case}{C.LABEL_SUFFIX}"
                    for case in uio.list_cases(a.binary_pred_dir, C.LABEL_SUFFIX)}
    elif a.oracle_binary:
        bin_paths = oracle_binary_pred_paths()
    else:
        bin_paths = load_binary_pred_paths(a.binary_dataset, a.trainer, a.plans, a.folds)

    preds, gts = {}, {}
    for case, bp in tqdm(bin_paths.items(), desc="repainting"):
        binmask, meta = uio.read(bp)
        binmask = binmask > 0
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not gt_p.exists():
            continue
        gt, _ = uio.read(gt_p)
        lab, n = ndimage.label(binmask)
        final = np.zeros(binmask.shape, dtype=np.uint16)
        for i in range(1, n + 1):
            assigned = final_by_key.get((case, i))
            if assigned is not None:
                final[lab == i] = loc_value[assigned]
        preds[case] = (final, meta["spacing"])
        gts[case] = gt

    cases = sorted(set(preds) & set(gts))
    per_class, pooled_acc, n_components = score(cases, preds, gts, loc_value, spec.n_loc)

    classes_ever_predicted = sum(1 for row in per_class.values() if row[6] + row[7] > 0)

    def nanmean(i):
        return float(np.nanmean([row[i] for row in per_class.values()]))

    print(f"\n=== bootstrap-ensemble rule ({a.n_bootstrap} replicates) ===")
    print(f"pooled per-component accuracy: {pooled_acc:.4f} over {n_components} instances")
    print(f"  macro recall (mean recall over classes): {nanmean(4):.4f}")
    print(f"  classes ever predicted: {classes_ever_predicted} / {spec.n_loc}")
    for i, k in enumerate(("dice", "vs", "hd95", "precision", "recall", "mcc")):
        print(f"  {k:<12}{nanmean(i):.4f}")

    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "dice", "vs", "hd95_mm", "precision", "recall", "mcc",
                    "tp", "fp", "fn", "tn"])
        for name, r in per_class.items():
            w.writerow([name, *r])
    print(f"\nper-class detail written to {a.out_csv}")
    print(f"\nCompare pooled accuracy directly against the single-run rule "
          f"(0.4064) and hybrid (0.4286) baselines.")


if __name__ == "__main__":
    main()
