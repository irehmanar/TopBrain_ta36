"""
Experiment Delta, step 1: build Dataset314_TopAneuAneurysmCTA /
Dataset315_TopAneuAneurysmMRA -- the exact same binary target as Dataset304
(LOCATION_MASKS collapsed to background/aneurysm), restricted to one
modality's cases only, per prep/classify_modality.py's split.

Mirrors build_nnunet_datasets.py's own "aneurysm" stage byte-for-byte (same
label rule, same symlink-not-copy convention for images) -- duplicated here
rather than extending that shared file, so this last-try experiment cannot
touch anything the main pipeline depends on.

UNCERTAIN cases are always excluded (not assigned to either modality) --
see classify_modality.py's own docstring for why a wrong guess there would
be worse than silently dropping a handful of ambiguous cases.

    python -m topaneu_rsna.seg.build_modality_split_dataset --modality cta
    python -m topaneu_rsna.seg.build_modality_split_dataset --modality mra
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio

DS_ID = {"cta": C.DS_ANEURYSM_CTA, "mra": C.DS_ANEURYSM_MRA}


def _root(ds_id):
    return C.nnUNet_raw / f"Dataset{ds_id:03d}_{C.DS_NAMES[ds_id]}"


def _link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def build(modality: str, limit: int | None):
    if not C.MODALITY_JSON.exists():
        raise FileNotFoundError(
            f"{C.MODALITY_JSON} missing. Run prep.classify_modality first.")
    case_modality = json.loads(C.MODALITY_JSON.read_text())

    want = modality.upper()
    cases = sorted(c for c, m in case_modality.items() if m == want)
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError(f"no cases classified as {want} -- check {C.MODALITY_STATS_CSV}")

    ds_id = DS_ID[modality]
    root = _root(ds_id)
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    n = 0
    for case in tqdm(cases, desc=f"build {want}"):
        sp = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        ip = C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}"
        if not sp.exists() or not ip.exists():
            continue
        _link(ip, root / "imagesTr" / f"{case}_0000.nii.gz")
        lab, meta = uio.read(sp)
        uio.write((lab > 0).astype(np.uint8), meta, root / "labelsTr" / f"{case}.nii.gz")
        n += 1

    (root / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": want},   # single-modality now, unlike Dataset304's pooled "CTA_MRA"
        "labels": {"background": 0, "aneurysm": 1},
        "numTraining": int(n),
        "file_ending": ".nii.gz",
    }, indent=2))
    print(f"{root}  ({n} {want} cases)")
    if n < 30:
        print(f"[WARNING] only {n} cases -- a real 5-fold CV on this cohort will be "
              f"noisy; treat any Dice delta with real skepticism, not just a point "
              f"estimate.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["cta", "mra"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build(a.modality, a.limit)
