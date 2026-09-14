"""
Experiment 2: is the declared rule earning its ~0.40 pooled accuracy, or
would a plain learned classifier over the same geometric inputs do just as
well? Paper 1 runs exactly this comparison (Sect. 3.5, Table 2) and finds a
learned ranker over the rule's own features reaches near-parity on pooled
accuracy but loses badly on macro recall -- it collapses toward whichever
classes are common, emitting far fewer of the 52 classes at all than the
declared rule does.

This extracts the same per-instance features the rule already computes --
host vessel identity, host_dist_mm, arc-length fraction (continuous, not
just its bucketed decision), which resolution path was even available
(junction/arc/single), the junction-contact distance, and laterality -- into
one flat table, one row per lesion instance, alongside its true_class label.
Deliberately reuses assign_location_rule.py's own `_junction_candidate` /
`_arc_candidate` for the `resolved_by` category (rather than re-deriving
that comparison logic here) so this can never silently drift from whatever
the actual rule currently does; only the raw, continuous arc-fraction value
is computed separately, since `_arc_candidate` only returns its already-
bucketed decision.

One row per lesion instance that found a host vessel at all (the same
population the rule itself attempts to assign -- an instance with no vessel
within `--tau_mm` is excluded here exactly as the rule leaves it unassigned).
Missing/inapplicable numeric features (arc_fraction and junction_dist_mm for
a single-location vessel, or arc_fraction when unorientable) are left blank
rather than the row being dropped, with `is_orientable` flagging the latter
explicitly -- the rule handles these cases too (via majority fallback), so a
fair comparison can't just discard them.

    python -m topaneu_rsna.seg.build_feature_table
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import geometry as geo
from topaneu_rsna.utils import io as uio
from topaneu_rsna.utils import vessel_skeleton as vsk
from topaneu_rsna.seg.assign_location_rule import (
    PRIOR_PATH, build_vessel_to_locations, load_binary_pred_paths,
    oracle_binary_pred_paths, _junction_candidate, _arc_candidate)

FIELDNAMES = ["case", "instance_idx", "vessel", "laterality", "size", "host_dist_mm",
             "resolved_by", "arc_fraction", "is_orientable",
             "junction_dist_mm", "true_class"]


def laterality_of(vessel_name: str) -> str:
    if vessel_name.startswith("R-"):
        return "R"
    if vessel_name.startswith("L-"):
        return "L"
    return "none"


def raw_arc_fraction(inst, vessel_map, spacing, vessel_name, vessel_id, name_to_id):
    """The continuous arc-length fraction itself, independent of which
    bucket it lands in -- _arc_candidate only exposes the already-discretised
    decision, but a learned classifier should see the raw position."""
    raw = vsk.extract_skeleton(vessel_map == vessel_id, spacing)
    if raw is None:
        return None
    skel = vsk.orient_skeleton(raw, vessel_map, spacing,
                               vsk.PROXIMAL_ANCHOR.get(vessel_name, []), name_to_id)
    centroid_mm = np.argwhere(inst).mean(0) * np.asarray(spacing, np.float64)
    frac, _ = vsk.arc_fraction(skel, centroid_mm)
    return frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary_dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--oracle_binary", action="store_true",
                    help="use ground-truth LOCATION_MASKS (binarized) as a "
                         "perfect binary detector -- must match whatever "
                         "train_learned_assigner.py/build_hybrid_assignment.py "
                         "use for the same run, see oracle_binary_pred_paths()")
    ap.add_argument("--tau_mm", type=float, default=4.0)
    ap.add_argument("--junction_tau_mm", type=float, default=2.0)
    ap.add_argument("--junction_override_mm", type=float, default=1.5)
    ap.add_argument("--arc_ambiguous_margin", type=float, default=0.03)
    ap.add_argument("--min_voxels", type=int, default=3)
    ap.add_argument("--prior", type=Path, default=PRIOR_PATH)
    ap.add_argument("--out", type=Path, default=C.LOG_ROOT / "task2_feature_table.csv")
    a = ap.parse_args()

    spec = C.load_labels()
    vessel_to_locations = build_vessel_to_locations(spec)
    prior = json.loads(a.prior.read_text())
    name_to_id = {v: i + 1 for i, v in enumerate(spec.vessels)}

    if a.oracle_binary:
        bin_paths = oracle_binary_pred_paths()
        print(f"{len(bin_paths)} ORACLE binary 'predictions' (ground-truth binarized)")
    else:
        bin_paths = load_binary_pred_paths(a.binary_dataset, a.trainer, a.plans, a.folds)
        print(f"{len(bin_paths)} held-out binary predictions")

    rows = []
    for case, bp in tqdm(bin_paths.items(), desc="extracting features"):
        binmask, meta = uio.read(bp)
        binmask = binmask > 0
        spacing = meta["spacing"]

        vp = C.VESSEL_MASKS / f"{case}{C.LABEL_SUFFIX}"
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not vp.exists() or not gt_p.exists():
            continue
        vessel_map, _ = uio.read(vp)
        gt, _ = uio.read(gt_p)

        lab, n = ndimage.label(binmask)
        for i in range(1, n + 1):
            inst = lab == i
            size = int(inst.sum())
            if size < a.min_voxels:
                continue
            vessel_id, host_dist, _ = geo.nearest_vessel_label(
                inst, vessel_map, spacing, tau_mm=a.tau_mm)
            if vessel_id is None:
                continue
            vessel_name = spec.vessels[vessel_id - 1]
            locs = vessel_to_locations.get(vessel_name, [])

            vals, counts = np.unique(gt[inst], return_counts=True)
            nz = vals != 0
            true_id = int(vals[nz][np.argmax(counts[nz])]) if nz.any() else 0
            true_name = spec.locations[true_id - 1] if true_id else "background"

            row = dict(case=case, instance_idx=i, vessel=vessel_name, size=size,
                      host_dist_mm=host_dist, laterality=laterality_of(vessel_name),
                      true_class=true_name)

            if len(locs) <= 1:
                row.update(resolved_by="single", arc_fraction="",
                          is_orientable=0, junction_dist_mm="")
            else:
                j_loc, j_dist = _junction_candidate(inst, vessel_map, spacing, vessel_id,
                                                    locs, name_to_id, a.junction_tau_mm)
                a_loc, a_dist, a_ambiguous, a_low_sample = _arc_candidate(
                    inst, vessel_map, spacing, vessel_name, vessel_id, prior,
                    name_to_id, a.arc_ambiguous_margin)
                frac = raw_arc_fraction(inst, vessel_map, spacing, vessel_name,
                                        vessel_id, name_to_id)

                if j_loc is not None and (a_loc is None or (j_dist <= a.junction_override_mm
                                                            and a_ambiguous)):
                    resolved_by = "junction"
                elif a_loc is not None:
                    resolved_by = "arc"
                elif a_low_sample:
                    resolved_by = "arc_low_sample_fallback"
                else:
                    resolved_by = "majority"

                row.update(
                    resolved_by=resolved_by,
                    arc_fraction=("" if frac is None else frac),
                    is_orientable=int(frac is not None),
                    junction_dist_mm=("" if j_dist == np.inf else j_dist),
                )
            rows.append(row)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} instances written to {a.out}")


if __name__ == "__main__":
    main()
