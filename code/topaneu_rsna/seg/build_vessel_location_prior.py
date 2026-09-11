"""
Cohort-derived fallback table for the simple first-pass rule engine
(seg/assign_location_rule.py).

Several vessel labels in the 36-class taxonomy host more than one of the 52
location classes along their length (e.g. "BA" alone hosts 7: 1.4 BA trunk,
1.5 VA-BA junction, R/L-1.7 BA-AICA junction, R/L-1.9 BA-SCA junction, 1.10 BA
tip). Once the rule engine has picked a lesion's single nearest/host vessel
(utils/geometry.nearest_vessel_label), that vessel-to-location step is
one-to-many and needs a tie-break. Paper 1's declared rule resolves this with
sub-parcellation along the vessel's own geodesic (landmark, ostium-anchor, or
cohort-median arc fraction, in that priority order). This is the simple-first
stand-in for all three: no geometry, just each host vessel's single most
frequent location, counted by connected component across the whole training
cohort -- exactly the "cohort-median" fallback Paper 1 itself falls back to
when a landmark isn't segmented, just applied unconditionally rather than
only as a last resort. It is a fixed anatomical prior derived once from all
training cases, not a per-fold-sensitive learned parameter (the same
treatment the declared location_to_vessel map itself already gets in
labels.json), so no train/val split is needed here.

Output: vessel_location_prior.json next to labels.json --
  {"multi_location_vessels": {vessel_name: [{"location": ..., "count": ...}, ...]},
   "majority": {vessel_name: location_name}}
only for vessels whose location_to_vessel fan-in is > 1; single-location
vessels need no prior; the rule engine assigns those directly.

    python -m topaneu_rsna.seg.build_vessel_location_prior
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scipy import ndimage
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio

OUT_PATH = C.CODE_ROOT / "topaneu_rsna" / "vessel_location_prior.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    a = ap.parse_args()

    spec = C.load_labels()
    vessel_to_locations: dict[str, list[str]] = defaultdict(list)
    for loc, v in spec.location_to_vessel.items():
        vessel_to_locations[v].append(loc)
    multi = {v: locs for v, locs in vessel_to_locations.items() if len(locs) > 1}
    print(f"{len(multi)} of {spec.n_vessel} vessels host more than one location: "
          f"{sorted(multi)}")

    loc_value = {loc: i + 1 for i, loc in enumerate(spec.locations)}
    counts = {v: Counter() for v in multi}

    cases = uio.list_cases(C.LOCATION_MASKS, C.LABEL_SUFFIX)
    for case in tqdm(cases, desc="counting components per location"):
        lab, _ = uio.read(C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}")
        for loc in spec.locations:
            v = spec.location_to_vessel[loc]
            if v not in multi:
                continue
            m = lab == loc_value[loc]
            if not m.any():
                continue
            _, n = ndimage.label(m)
            counts[v][loc] += n

    out = {"multi_location_vessels": {}, "majority": {}}
    for v, locs in multi.items():
        ranked = counts[v].most_common()
        # locations that never occur still need a legal fallback -- append
        # them (count 0) so `majority` always resolves to something in locs
        seen = {loc for loc, _ in ranked}
        ranked += [(loc, 0) for loc in locs if loc not in seen]
        out["multi_location_vessels"][v] = [
            {"location": loc, "count": n} for loc, n in ranked]
        out["majority"][v] = ranked[0][0]
        print(f"  {v:20s} -> {ranked}")

    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
