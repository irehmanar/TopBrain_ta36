"""
Inspection only, no rerun of the full pipeline: find out whether a location
stuck at 0 recall (e.g. "1.10 BA tip") is a *table* gap (fixable in minutes)
or a *vessel-segmentation coverage* problem (bigger, separate issue).

Two checks, both read-only against what jobs 50-54 already produced:

  1. Does the location have an arc-fraction table entry at all, and is it
     flagged low_sample? (straight from vessel_location_prior.json)
  2. For every held-out lesion instance whose nearest/host vessel is the one
     this location shares, what arc-fraction does it actually compute to?
     If real instances' fractions cluster near where the table says this
     location's bucket should be, but assign_location_rule.py never emits
     it, that points at a boundary/bucketing bug. If no instance's fraction
     ever lands anywhere near the table's median for this location, the
     lesions that should be landing there are probably being caught by a
     *different* host vessel entirely at the nearest_vessel_label step (a
     vessel-segmentation/topology issue upstream of this whole table), or
     ground truth for this location just doesn't co-occur with a correctly
     segmented copy of the shared vessel in this held-out set.

    python -m topaneu_rsna.seg.debug_dead_class --vessel BA --location "1.10 BA tip"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import geometry as geo
from topaneu_rsna.utils import io as uio
from topaneu_rsna.utils import vessel_skeleton as vsk
from topaneu_rsna.seg.assign_location_rule import (PRIOR_PATH, build_vessel_to_locations,
                                                    load_binary_pred_paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vessel", required=True, help='e.g. "BA"')
    ap.add_argument("--location", required=True, help='e.g. "1.10 BA tip"')
    ap.add_argument("--prior", type=Path, default=PRIOR_PATH)
    ap.add_argument("--binary_dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tau_mm", type=float, default=4.0)
    ap.add_argument("--min_voxels", type=int, default=3)
    a = ap.parse_args()

    prior = json.loads(a.prior.read_text())
    arc_info = prior.get("arc", {}).get(a.vessel)
    print(f"=== check 1: table entry for {a.location!r} on vessel {a.vessel!r} ===")
    if not arc_info or not arc_info["locations"]:
        print(f"  no arc table at all for {a.vessel} -- every instance here falls "
             f"back to majority ({prior['majority'].get(a.vessel)!r})")
    else:
        entry = next((e for e in arc_info["locations"] if e["location"] == a.location), None)
        if entry is None:
            print(f"  TABLE GAP: {a.location!r} has no entry in vessel_location_prior.json "
                 f"at all -- it never had a single training instance with an orientable "
                 f"skeleton on {a.vessel}. This is fixable only by getting real training "
                 f"examples oriented (check PROXIMAL_ANCHOR coverage), not a boundary bug.")
        else:
            print(f"  entry exists: median_frac={entry['median_frac']:.3f}, "
                 f"n={entry['n']}, low_sample={entry['low_sample']}")
            print(f"  full sorted order: "
                 f"{[(e['location'], round(e['median_frac'], 3), e['n']) for e in arc_info['locations']]}")
            print(f"  bucket boundaries: {arc_info['boundaries']}")
            if entry["low_sample"]:
                print(f"  -> flagged low_sample: assign_location_rule.py will never "
                     f"actually return this location even if a lesion's fraction lands "
                     f"in its bucket -- falls back to majority instead. This alone "
                     f"could fully explain a stuck-at-zero class.")

    print(f"\n=== check 2: real instances hosted on {a.vessel} -- where do their "
         f"arc-fractions actually land? ===")
    spec = C.load_labels()
    name_to_id = {v: i + 1 for i, v in enumerate(spec.vessels)}
    vessel_to_locations = build_vessel_to_locations(spec)
    target_id = name_to_id.get(a.vessel)
    if target_id is None:
        raise SystemExit(f"{a.vessel!r} not in labels.json vessels")

    bin_paths = load_binary_pred_paths(a.binary_dataset, a.trainer, a.plans, a.folds)
    n_hosted = 0
    fracs = []
    for case, bp in tqdm(bin_paths.items(), desc="scanning"):
        binmask, meta = uio.read(bp)
        binmask = binmask > 0
        vp = C.VESSEL_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not vp.exists():
            continue
        vessel_map, _ = uio.read(vp)
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not gt_p.exists():
            continue
        gt, _ = uio.read(gt_p)

        from scipy import ndimage
        lab, n = ndimage.label(binmask)
        for i in range(1, n + 1):
            inst = lab == i
            if int(inst.sum()) < a.min_voxels:
                continue
            vessel_id, dist, _ = geo.nearest_vessel_label(
                inst, vessel_map, meta["spacing"], tau_mm=a.tau_mm)
            if vessel_id != target_id:
                continue
            n_hosted += 1

            raw = vsk.extract_skeleton(vessel_map == target_id, meta["spacing"])
            frac = None
            if raw is not None:
                skel = vsk.orient_skeleton(raw, vessel_map, meta["spacing"],
                                           vsk.PROXIMAL_ANCHOR.get(a.vessel, []),
                                           name_to_id)
                centroid_mm = (np.argwhere(inst).mean(0)
                              * np.asarray(meta["spacing"], np.float64))
                frac, _ = vsk.arc_fraction(skel, centroid_mm)

            vals, counts = np.unique(gt[inst], return_counts=True)
            nz = vals != 0
            true_id = int(vals[nz][np.argmax(counts[nz])]) if nz.any() else 0
            true_name = spec.locations[true_id - 1] if true_id else "background"
            fracs.append((case, frac, true_name))

    print(f"  {n_hosted} held-out instances have {a.vessel} as their host vessel")
    resolvable = [f for f in fracs if f[1] is not None]
    print(f"  {len(resolvable)} of those got a valid (oriented) arc-fraction")
    matching_true = [f for f in fracs if f[2] == a.location]
    print(f"\n  instances whose TRUE label is {a.location!r}: {len(matching_true)}")
    for case, frac, true_name in matching_true:
        print(f"    {case}: computed arc-fraction = "
             f"{'None (unorientable)' if frac is None else f'{frac:.3f}'}")
    if not matching_true:
        print(f"  -> no held-out instance was even found hosted on {a.vessel} with "
             f"true label {a.location!r}. Either its lesions are being caught by a "
             f"*different* host vessel at the nearest_vessel_label step (check "
             f"--tau_mm and whether {a.vessel}'s segmentation actually reaches that "
             f"anatomical region in these cases), or Dataset304's binary segmenter is "
             f"missing them entirely (a segmentation-recall problem, not a rule one).")

    print(f"\n  all {a.vessel}-hosted instances, for context (case, fraction, true label):")
    for case, frac, true_name in sorted(fracs, key=lambda f: (f[1] is None, f[1] or 0)):
        print(f"    {case:<28} frac={'None ' if frac is None else f'{frac:.3f}'} "
             f"true={true_name}")


if __name__ == "__main__":
    main()
