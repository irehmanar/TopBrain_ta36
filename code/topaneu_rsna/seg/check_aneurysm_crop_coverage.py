"""
Audit-only: does the coarse-aneurysm crop (seg/aneurysm_coarse_roi.py) actually
contain the true aneurysm, per case?

Deliberately separate from the crop step itself and CPU-only -- this only
*reports* coverage, it never feeds back into or changes the crop. The crop is
built purely from the coarse model's own prediction (self-consistent with
inference); this script is the ground-truth check run alongside it.

Uses the crop's own cropped `{case}_location.nii.gz` (written by
aneurysm_coarse_roi.py from the exact same lo/hi as the image crop) rather
than recomputing coordinates -- if that cropped label is all-background but
the case's real, uncropped location_masks has a labeled aneurysm somewhere,
the crop missed it.

    python -m topaneu_rsna.seg.check_aneurysm_crop_coverage
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio


def check_case(case: str, roi_dir: Path) -> str:
    lp_full = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
    if not lp_full.exists():
        return "no_ground_truth"

    full, _ = uio.read(lp_full)
    if not (full > 0).any():
        return "no_aneurysm_in_case"

    lp_crop = roi_dir / f"{case}_location{C.LABEL_SUFFIX}"
    if not lp_crop.exists():
        return "crop_missing"

    crop, _ = uio.read(lp_crop)
    return "covered" if (crop > 0).any() else "missed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_dir", type=Path, default=C.ANEURYSM_COARSE_ROI_DIR)
    ap.add_argument("--out", type=Path,
                    default=C.LOG_ROOT / "aneurysm_coarse_crop_coverage.csv")
    a = ap.parse_args()

    cases = sorted(p.name[: -len(C.IMAGE_SUFFIX)]
                   for p in a.roi_dir.glob(f"*{C.IMAGE_SUFFIX}"))
    if not cases:
        raise SystemExit(f"no crops in {a.roi_dir} -- run aneurysm_coarse_roi.py first")

    rows = []
    for case in tqdm(cases, desc="checking coverage"):
        status = check_case(case, a.roi_dir)
        fallback = None
        cj = a.roi_dir / f"{case}_coarse.json"
        if cj.exists():
            fallback = json.loads(cj.read_text()).get("dbscan_fallback")
        rows.append({"case": case, "status": status, "dbscan_fallback": fallback})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "status", "dbscan_fallback"])
        w.writeheader(); w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    n_relevant = sum(v for k, v in counts.items() if k in ("covered", "missed"))

    print(f"\n{len(cases)} cases checked, written to {a.out}\n")
    for status, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status:<20} {n}")
    if n_relevant:
        missed = counts.get("missed", 0)
        print(f"\ncoverage rate (of cases with a real aneurysm): "
             f"{100 * (n_relevant - missed) / n_relevant:.1f}%")

    missed_fallback = sum(1 for r in rows if r["status"] == "missed" and r["dbscan_fallback"])
    if counts.get("missed", 0):
        print(f"of {counts['missed']} missed cases, {missed_fallback} had an empty "
             "coarse prediction (centre-of-volume fallback) -- worth checking those first")


if __name__ == "__main__":
    main()
