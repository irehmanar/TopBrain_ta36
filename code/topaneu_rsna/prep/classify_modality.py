"""
Experiment Delta, step 0: classify every case as CTA or MRA.

No modality field exists anywhere in this dataset's own metadata (confirmed
by grepping location_jsons/, dataset.json, and every prep/ script -- the
only tag anywhere is build_nnunet_datasets.py's own generic "CTA_MRA"
channel_names entry, which pools both modalities together, not one that
distinguishes them per case). So this uses a physics-grounded intensity
heuristic instead of a guess:

  CTA stores calibrated Hounsfield Units. Air is *always* ~-1000 HU, so a
  real CT/CTA volume has a large fraction of its voxels (the background air
  around the head, plus air-filled sinuses) sitting in a narrow band right
  at -1000.

  MRA has arbitrary, scanner- and sequence-dependent intensity units with no
  fixed physical anchor -- it is never calibrated to a universal "air = X"
  constant, and its raw values are effectively always >= 0.

A case is classified as CTA if a meaningful fraction of its voxels sit in
the [-1050, -950] HU band; as MRA if its minimum intensity is close to 0
(no substantially negative voxels at all); anything that fits neither
pattern cleanly is marked UNCERTAIN and EXCLUDED from the modality-split
datasets by build_modality_split_dataset.py, rather than guessed -- a wrong
guess would quietly poison whichever modality-specific model inherits it.

Writes:
  MODALITY_JSON        {case: "CTA"|"MRA"|"UNCERTAIN"}
  MODALITY_STATS_CSV   per-case raw stats, for manual audit of the boundary
                        cases before trusting the split

    python -m topaneu_rsna.prep.classify_modality
    python -m topaneu_rsna.prep.classify_modality --air_band_frac 0.02
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio

AIR_LO, AIR_HI = -1050.0, -950.0


def classify_one(img: np.ndarray, air_band_frac: float) -> tuple[str, dict]:
    vmin = float(img.min())
    vmax = float(img.max())
    air_frac = float(((img >= AIR_LO) & (img <= AIR_HI)).mean())
    neg_frac = float((img < -100).mean())

    stats = dict(vmin=vmin, vmax=vmax, air_band_frac=air_frac, neg_frac=neg_frac)

    if air_frac >= air_band_frac:
        return "CTA", stats
    if vmin > -10.0 and neg_frac < 1e-4:
        return "MRA", stats
    return "UNCERTAIN", stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--air_band_frac", type=float, default=0.01,
                    help="min fraction of voxels in [-1050,-950] HU to call a case CTA")
    ap.add_argument("--out", type=Path, default=C.MODALITY_JSON)
    ap.add_argument("--stats_out", type=Path, default=C.MODALITY_STATS_CSV)
    a = ap.parse_args()

    cases = uio.list_cases(C.IMAGES_DIR, C.IMAGE_SUFFIX)
    print(f"[modality] classifying {len(cases)} cases...")

    labels, rows = {}, []
    for case in tqdm(cases):
        img, _ = uio.read(C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}")
        label, stats = classify_one(img, a.air_band_frac)
        labels[case] = label
        rows.append({"case": case, "modality": label, **stats})

    counts = {k: sum(1 for v in labels.values() if v == k) for k in ("CTA", "MRA", "UNCERTAIN")}
    print(f"\n[modality] CTA={counts['CTA']}  MRA={counts['MRA']}  "
          f"UNCERTAIN={counts['UNCERTAIN']} (excluded from the split)")
    for m in ("CTA", "MRA"):
        if 0 < counts[m] < 30:
            print(f"[modality] WARNING: only {counts[m]} {m} cases -- a real 5-fold CV "
                  f"on this few cases will be noisy. Consider whether this modality "
                  f"has enough data to train a dedicated model at all before proceeding.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(labels, indent=2, sort_keys=True))
    print(f"\nwrote {a.out}")

    a.stats_out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.stats_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "modality", "vmin", "vmax",
                                          "air_band_frac", "neg_frac"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.stats_out} -- please skim the UNCERTAIN rows before "
          f"running build_modality_split_dataset.py")


if __name__ == "__main__":
    main()
