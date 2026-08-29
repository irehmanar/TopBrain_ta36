"""
Force the author's spacing / patch size into an nnU-Net plans file.

Dataset301 is handled by the ForcedLowres planner + -overwrite_target_spacing,
so it only needs the patch size set to 128^3 (exactly what the author's README
does in step 4.1).  Dataset302 needs the anisotropic (0.80, 0.45, 0.44) spacing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from topaneu_rsna import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, required=True)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--spacing", type=float, nargs=3, default=None, help="z y x")
    ap.add_argument("--patch", type=int, nargs=3, default=None, help="z y x")
    a = ap.parse_args()

    ds_dir = next(C.nnUNet_preprocessed.glob(f"Dataset{a.dataset:03d}_*"))
    p = ds_dir / f"{a.plans}.json"
    plans = json.loads(p.read_text())
    cfg = plans["configurations"][a.configuration]
    if a.spacing:
        cfg["spacing"] = list(a.spacing)
    if a.patch:
        cfg["patch_size"] = list(a.patch)
    p.write_text(json.dumps(plans, indent=2))
    print(f"patched {p}: spacing={cfg['spacing']} patch={cfg['patch_size']}")


if __name__ == "__main__":
    main()
