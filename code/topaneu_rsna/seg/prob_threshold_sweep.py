"""Probability-threshold and hysteresis sweep for the binary aneurysm model, on
the same honest protocol as patch_size_ablation.py (each fold predicts ONLY its
own held-out cases with ONLY that fold's model), using the v3 container's
binary settings: [112,224,224] patch + left-right mirroring only.

The container currently keeps every voxel whose argmax class is aneurysm, i.e.
foreground probability > 0.5. Dataset304 was trained with a recall-weighted
(Tversky beta 0.7) loss, so 0.5 is not necessarily the best cutoff. For each
case this computes the foreground probability ONCE, then scores:
  plain t       : p > t
  hyst lo/hi    : grow components at p > lo, keep only those whose peak
                  probability is >= hi (seeded region growing)
Metrics per config, pooled over cases: Dice, case-level precision / recall /
MCC (a case is a TP if the mask overlaps GT, same rule as evaluate_location.py),
and component precision (the fraction of predicted components that overlap GT).
HD95 is skipped (it needs a distance transform per case per config).

    python -m topaneu_rsna.seg.prob_threshold_sweep
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio

PLAIN = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
HYST = [(lo, hi) for lo in (0.3, 0.4, 0.5) for hi in (0.6, 0.7, 0.8, 0.9) if hi > lo]
CONFIGS = [("plain", t, t) for t in PLAIN] + [("hyst", lo, hi) for lo, hi in HYST]
LO_MIN = min([c[1] for c in CONFIGS])


def config_name(c):
    return f"plain_{c[1]}" if c[0] == "plain" else f"hyst_lo{c[1]}_hi{c[2]}"


def masks_for_case(p, sl):
    """Predicted mask per config, computed only inside the bbox `sl` of
    p > LO_MIN (everything outside cannot be foreground under any config)."""
    pc = p[sl]
    out = {}
    labs = {}
    for c in CONFIGS:
        name = config_name(c)
        if c[0] == "plain":
            out[name] = pc > c[1]
        else:
            lo = c[1]
            if lo not in labs:
                labs[lo] = ndimage.label(pc > lo)
            lab, n = labs[lo]
            if n == 0:
                out[name] = np.zeros(pc.shape, bool)
                continue
            peak = np.asarray(ndimage.maximum(pc, lab, index=np.arange(1, n + 1)))
            keep = np.zeros(n + 1, bool)
            keep[1:] = peak >= c[2]
            out[name] = keep[lab]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=str(C.LOG_ROOT / "prob_threshold_sweep.csv"))
    a = ap.parse_args()

    name = f"Dataset{a.dataset:03d}_{C.DS_NAMES[a.dataset]}"
    model_dir = C.nnUNet_results / name / f"{a.trainer}__{a.plans}__3d_fullres"
    images_dir = C.nnUNet_raw / name / "imagesTr"
    gt_dir = C.nnUNet_raw / name / "labelsTr"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st = {config_name(c): dict(tp=0, fp=0, fn=0, tn=0, inter=0, ps=0, gs=0,
                               comps=0, comp_hit=0) for c in CONFIGS}

    def report(tag):
        print(f"\n--- {tag} ---")
        print(f"{'config':<20}{'dice':>7}{'prec':>7}{'rec':>7}{'mcc':>7}{'compP':>7}"
              f"{'tp':>5}{'fp':>5}{'fn':>5}")
        rows = []
        for c in CONFIGS:
            n = config_name(c); s = st[n]
            dice = 2 * s["inter"] / (s["ps"] + s["gs"]) if s["ps"] + s["gs"] else float("nan")
            prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else float("nan")
            rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else float("nan")
            den = np.sqrt((s["tp"] + s["fp"]) * (s["tp"] + s["fn"]) * (s["tn"] + s["fp"]) * (s["tn"] + s["fn"]))
            mcc = (s["tp"] * s["tn"] - s["fp"] * s["fn"]) / den if den else float("nan")
            cp = s["comp_hit"] / s["comps"] if s["comps"] else float("nan")
            print(f"{n:<20}{dice:7.4f}{prec:7.4f}{rec:7.4f}{mcc:7.4f}{cp:7.4f}"
                  f"{s['tp']:5d}{s['fp']:5d}{s['fn']:5d}", flush=True)
            rows.append([n, dice, prec, rec, mcc, cp, s["tp"], s["fp"], s["fn"], s["tn"]])
        return rows

    n_cases = 0
    for fold in a.folds:
        cases = uio.list_cases(model_dir / f"fold_{fold}" / "validation", C.LABEL_SUFFIX)
        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=False, device=device,
            verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(
            str(model_dir), use_folds=(fold,), checkpoint_name="checkpoint_final.pth")
        predictor.configuration_manager.configuration["patch_size"] = [112, 224, 224]
        predictor.allowed_mirroring_axes = (2,)

        for case in tqdm(cases, desc=f"fold {fold}"):
            gp = gt_dir / f"{case}{C.LABEL_SUFFIX}"
            if not gp.exists():
                continue
            img, meta = uio.read(images_dir / f"{case}{C.IMAGE_SUFFIX}")
            gt = uio.read(gp)[0] > 0
            _, probs = predictor.predict_single_npy_array(
                img[None].astype(np.float32), {"spacing": meta["spacing"]},
                None, None, True)
            p = np.asarray(probs)[1].astype(np.float32)
            n_cases += 1
            gs = int(gt.sum())

            box = ndimage.find_objects((p > LO_MIN).astype(np.uint8))
            if box:
                sl = box[0]
                masks = masks_for_case(p, sl)
                gtc = gt[sl]
            else:
                masks, gtc = None, None

            for c in CONFIGS:
                n = config_name(c); s = st[n]
                if masks is None:
                    m = None; ps = it = 0
                else:
                    m = masks[n]; ps = int(m.sum()); it = int((m & gtc).sum())
                s["ps"] += ps; s["gs"] += gs; s["inter"] += it
                if gs > 0 and it > 0:
                    s["tp"] += 1
                elif gs > 0:
                    s["fn"] += 1
                elif ps > 0:
                    s["fp"] += 1
                else:
                    s["tn"] += 1
                if m is not None and ps > 0:
                    lab, k = ndimage.label(m)
                    s["comps"] += k
                    hit = np.unique(lab[m & gtc])
                    s["comp_hit"] += int((hit > 0).sum())

        del predictor
        torch.cuda.empty_cache()
        report(f"after fold {fold} ({n_cases} cases)")

    rows = report(f"FINAL ({n_cases} cases)")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "dice", "precision", "recall", "mcc", "comp_precision",
                    "tp", "fp", "fn", "tn"])
        w.writerows(rows)
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
