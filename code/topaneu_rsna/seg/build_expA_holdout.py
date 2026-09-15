"""
Experiment A, step 0: a fixed ~15% case holdout, generated ONCE and then
reused unchanged by every later Experiment A job -- dataset build (excludes
these cases from imagesTr/labelsTr so Dataset313's fold_all training never
sees them), classifier training (same exclusion), and consensus-fusion
evaluation (the only cases actually scored).

Why a manual holdout instead of the paper's real 5-fold CV: this pipeline's
segmentation side has no center/modality-stratified split anywhere (see
config.py's DS_VESSELCOND_SEG comment), and training 5 folds of a network
this size was judged too expensive for a 1-week budget -- fold_all + one
honest, never-trained-on holdout was the explicit tradeoff made instead (same
one already made for Models 1/2/3, which also only ever train fold_all).

Deterministic and seeded so re-running this script is a no-op (same holdout
every time) unless --seed or --frac is deliberately changed -- everything
downstream depends on this list staying fixed once other jobs start
referencing it.

    python -m topaneu_rsna.seg.build_expA_holdout
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=C.EXPA_HOLDOUT_FRAC)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=type(C.EXPA_HOLDOUT_JSON), default=C.EXPA_HOLDOUT_JSON)
    a = ap.parse_args()

    if a.out.exists():
        existing = json.loads(a.out.read_text())
        print(f"[expA holdout] {a.out} already exists ({len(existing['holdout'])} cases) "
             f"-- leaving it untouched. Delete it first if you really want to "
             f"regenerate (this would invalidate any Experiment A job that already "
             f"ran against the old split).")
        return

    cases = uio.list_cases(C.IMAGES_DIR, C.IMAGE_SUFFIX)
    cases = [c for c in cases
            if (C.LOCATION_MASKS / f"{c}{C.LABEL_SUFFIX}").exists()]
    rng = np.random.default_rng(a.seed)
    n_holdout = max(1, round(len(cases) * a.frac))
    holdout = sorted(rng.choice(cases, size=n_holdout, replace=False).tolist())
    train = sorted(set(cases) - set(holdout))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "seed": a.seed, "frac": a.frac,
        "n_total": len(cases), "n_holdout": len(holdout), "n_train": len(train),
        "holdout": holdout, "train": train,
    }, indent=2))
    print(f"[expA holdout] {len(cases)} total cases -> {len(train)} train / "
         f"{len(holdout)} holdout ({100 * len(holdout) / len(cases):.1f}%)")
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
