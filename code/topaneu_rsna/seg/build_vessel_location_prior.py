"""
Cohort-derived fallback tables for the rule engine (seg/assign_location_rule.py)
on the 13 vessel labels that host more than one of the 52 location classes
(e.g. "BA" alone hosts 7: 1.4 BA trunk, 1.5 VA-BA junction, R/L-1.7
BA-AICA junction, R/L-1.9 BA-SCA junction, 1.10 BA tip).

Two tables are built, matching Paper 1's own priority order for resolving a
shared vessel (sub-parcellation by geodesic position, cohort-median
fallback):

  "arc"       Per shared vessel, every hosted location's median arc-length
              fraction along that vessel's centerline (0.0 = proximal anchor
              end, 1.0 = the other end -- see utils.vessel_skeleton), sorted,
              with decision boundaries at the midpoints between consecutive
              medians. Computed per training case by skeletonizing that
              case's own ground-truth vessel label and projecting each
              ground-truth lesion's centroid onto it -- a genuine
              cohort-derived position statistic, not a per-fold-sensitive
              learned parameter (same treatment labels.json's declared
              location_to_vessel map already gets), so no train/val split is
              needed here.
  "majority"  The original flat "most frequent location on this vessel"
              table, kept as a fallback for cases where a training case's own
              vessel segmentation couldn't be oriented (no anchor vessel
              present -- see PROXIMAL_ANCHOR) or a location never had enough
              instances to get a stable median.

Junction-type locations (name contains "junction"/"bifurcation"/"terminus")
still get an arc-fraction entry here as a fallback, but
assign_location_rule.py checks their branch-contact-patch distance first
(Paper 1's junction-object treatment) and only falls through to the
arc-fraction bucket below if that check doesn't fire.

    python -m topaneu_rsna.seg.build_vessel_location_prior
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio
from topaneu_rsna.utils import vessel_skeleton as vsk

OUT_PATH = C.CODE_ROOT / "topaneu_rsna" / "vessel_location_prior.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    a = ap.parse_args()

    spec = C.load_labels()
    name_to_id = {v: i + 1 for i, v in enumerate(spec.vessels)}
    loc_value = {loc: i + 1 for i, loc in enumerate(spec.locations)}

    vessel_to_locations: dict[str, list[str]] = defaultdict(list)
    for loc, v in spec.location_to_vessel.items():
        vessel_to_locations[v].append(loc)
    multi = {v: locs for v, locs in vessel_to_locations.items() if len(locs) > 1}
    print(f"{len(multi)} of {spec.n_vessel} vessels host more than one location: "
          f"{sorted(multi)}")
    missing_anchor = sorted(set(multi) - set(vsk.PROXIMAL_ANCHOR))
    if missing_anchor:
        print(f"WARNING: no PROXIMAL_ANCHOR declared for {missing_anchor} -- "
             "these will only ever get the flat majority fallback")

    majority_counts = {v: Counter() for v in multi}
    arc_fractions: dict[str, dict[str, list[float]]] = {v: defaultdict(list) for v in multi}
    n_oriented = n_unoriented = 0

    cases = uio.list_cases(C.LOCATION_MASKS, C.LABEL_SUFFIX)
    for case in tqdm(cases, desc="skeletonizing + projecting"):
        loc_lab, meta = uio.read(C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}")
        spacing = meta["spacing"]
        vessel_p = C.VESSEL_MASKS / f"{case}{C.LABEL_SUFFIX}"
        vessel_map = None  # loaded lazily, only if this case hosts a multi-location vessel

        for v, locs in multi.items():
            case_has_v_location = any((loc_lab == loc_value[loc]).any() for loc in locs)
            if not case_has_v_location:
                continue
            if vessel_map is None:
                if not vessel_p.exists():
                    continue
                vessel_map, _ = uio.read(vessel_p)

            skel = None
            if v in vsk.PROXIMAL_ANCHOR:
                raw = vsk.extract_skeleton(vessel_map == name_to_id[v], spacing)
                if raw is not None:
                    skel = vsk.orient_skeleton(raw, vessel_map, spacing,
                                               vsk.PROXIMAL_ANCHOR[v], name_to_id)

            for loc in locs:
                m = loc_lab == loc_value[loc]
                if not m.any():
                    continue
                lab, n = ndimage.label(m)
                majority_counts[v][loc] += n
                for i in range(1, n + 1):
                    pts = np.argwhere(lab == i)
                    centroid_mm = pts.mean(0) * np.asarray(spacing, np.float64)
                    if skel is not None:
                        frac, _ = vsk.arc_fraction(skel, centroid_mm)
                        if frac is not None:
                            arc_fractions[v][loc].append(frac)
                            n_oriented += 1
                            continue
                    n_unoriented += 1

    print(f"\n{n_oriented} lesion instances got an arc-fraction, "
         f"{n_unoriented} fell back to majority-only (no orientable skeleton)")

    out = {"multi_location_vessels": {}, "majority": {}, "arc": {}}
    for v, locs in multi.items():
        ranked = majority_counts[v].most_common()
        seen = {loc for loc, _ in ranked}
        ranked += [(loc, 0) for loc in locs if loc not in seen]
        out["multi_location_vessels"][v] = [
            {"location": loc, "count": n} for loc, n in ranked]
        out["majority"][v] = ranked[0][0]
        print(f"  {v:20s} majority -> {ranked}")

        entries = sorted(
            ({"location": loc, "median_frac": float(np.median(fracs)), "n": len(fracs)}
             for loc, fracs in arc_fractions[v].items() if fracs),
            key=lambda e: e["median_frac"])
        boundaries = [round((e1["median_frac"] + e2["median_frac"]) / 2.0, 4)
                     for e1, e2 in zip(entries, entries[1:])]
        out["arc"][v] = {"locations": entries, "boundaries": boundaries}
        if entries:
            print(f"  {v:20s} arc-order -> "
                 f"{[(e['location'], round(e['median_frac'], 3), e['n']) for e in entries]}")
        else:
            print(f"  {v:20s} arc-order -> no orientable instances, majority-only")

    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
