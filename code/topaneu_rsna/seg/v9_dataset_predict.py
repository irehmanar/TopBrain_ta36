"""Run the submitted v9 container's own inference.py (docker/task2_rule_based_v9/
inference.py, unchanged) on the training cases on Narval and write one uint8
52-class mask per case, ready for the OFFICIAL TopAneu-26 Task 2 evaluator
(github.com/Bangulli/TopAneu-26/eval/task2, the same code that scored the test
set). This gives a "training set" number for the same pipeline that got the
final-test score, computed with the same metrics.

What this measures, exactly: every case here was training data for 4 of the 5
Dataset304 folds and for Dataset302 (fold_all), and the rule priors were built
from all cases, so this is an optimistic, seen-data number, not a held-out one.

Differences from the grand-challenge T4 run, both deliberate:
  * inference._T0 is reset per case and LOGITS_DEADLINE_S is set huge, so the
    time-aware fold cut-off never fires (on the A100 it would not fire anyway;
    on the T4 it only fires on very large or slow cases);
  * the A100 has 40 GB, but the patch/mirroring plan is still decided from the
    tile count exactly as in the container, so the same settings are used.

    python -m topaneu_rsna.seg.v9_dataset_predict --inference_dir <dir with inference.py> \
        --out_dir <dir> --shard 0 --nshards 5
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import SimpleITK as sitk

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_dir", type=Path, required=True,
                    help="folder containing the v9 inference.py")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    a = ap.parse_args()

    sys.path.insert(0, str(a.inference_dir))
    import inference as inf  # the unmodified v9 file
    inf.LOGITS_DEADLINE_S = 1e9

    pred_dir = a.out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    images_dir = (C.nnUNet_raw / f"Dataset{C.DS_ANEURYSM:03d}_{C.DS_NAMES[C.DS_ANEURYSM]}"
                  / "imagesTr")
    cases = sorted(uio.list_cases(images_dir, C.IMAGE_SUFFIX))
    cases = [c for c in cases if (C.LOCATION_MASKS / f"{c}{C.LABEL_SUFFIX}").exists()]
    if a.limit:
        cases = cases[:a.limit]
    mine = cases[a.shard::a.nshards]
    print(f"{len(cases)} cases with GT; shard {a.shard}/{a.nshards}: {len(mine)} cases",
          flush=True)

    failed = []
    for i, case in enumerate(mine):
        out_path = pred_dir / f"{case}{C.LABEL_SUFFIX}"
        if out_path.exists():
            continue
        t0 = time.time()
        try:
            img = sitk.ReadImage(str(images_dir / f"{case}{C.IMAGE_SUFFIX}"))
            inf._T0 = time.time()
            out = inf.infer_ct(img)
            sitk.WriteImage(out, str(out_path), useCompression=True)
            print(f"[{i + 1}/{len(mine)}] {case} done in {time.time() - t0:.0f}s", flush=True)
        except Exception:
            failed.append(case)
            print(f"[{i + 1}/{len(mine)}] {case} FAILED\n{traceback.format_exc()}", flush=True)

    print(f"finished; {len(failed)} failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
