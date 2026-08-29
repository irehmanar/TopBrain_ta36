"""
Inference: EMA weights, left-right flip TTA, best-N fold ensemble, and the
author's fail-safe (fall back to mean out-of-fold probabilities when a case has
no usable ROI rather than guessing).

    python -m topaneu_rsna.cls.predict --cache_dir ... --out preds.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from topaneu_rsna import config as C
from topaneu_rsna.cls.dataset import TopAneuRoiDataset
from topaneu_rsna.cls.model import build_model


def pick_folds(results_dir: Path, n_keep: int):
    scored = []
    for d in sorted(results_dir.glob("fold_*")):
        ck = d / "best.pt"
        if ck.exists():
            scored.append((torch.load(ck, map_location="cpu")["best"], ck))
    scored.sort(key=lambda t: -t[0])
    keep = scored[:n_keep]
    print("using folds: " + ", ".join(f"{c.parent.name}({s:.4f})" for s, c in keep))
    return [c for _, c in keep]


def oof_fallback(results_dir: Path, n_loc: int):
    ps = [np.load(f)["p"] for f in results_dir.glob("fold_*/oof.npz")]
    if not ps:
        return np.full(n_loc, 0.05, np.float32), 0.5
    allp = np.concatenate(ps, 0)
    return allp.mean(0).astype(np.float32), float(allp.max(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=Path, default=C.CLS_CACHE_DIR)
    ap.add_argument("--results_dir", type=Path, default=C.CLS_RESULTS_DIR)
    ap.add_argument("--seg_model_dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_folds", type=int, default=None)
    ap.add_argument("--no_tta", action="store_true")
    a = ap.parse_args()

    cfg = C.CLS
    spec = C.load_labels()
    seg_dir = a.seg_model_dir or C.seg_model_dir(C.DS_VESSEL, C.TRAINER_M2)
    ckpts = pick_folds(a.results_dir, a.n_folds or cfg.n_folds_ensemble)
    if not ckpts:
        raise SystemExit(f"no checkpoints under {a.results_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for c in ckpts:
        ck = torch.load(c, map_location="cpu")
        m = build_model(spec, cfg, seg_dir, C.SEG_FOLD, pretrained=False).to(device).eval()
        m.load_state_dict(ck.get("ema", ck["model"]))
        models.append(m)

    cases = sorted(p.stem for p in a.cache_dir.glob("*.npz"))
    dl = DataLoader(TopAneuRoiDataset(cases, a.cache_dir, spec, train=False),
                    batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    fb_loc, fb_ap = oof_fallback(a.results_dir, spec.n_loc)
    loc_p = np.zeros((len(cases), spec.n_loc), np.float32)
    ap_p = np.zeros(len(cases), np.float32)
    order, n_fallback = [], 0

    with torch.no_grad():
        for i, b in enumerate(dl):
            img = b["image"].to(device); v2 = b["vessel_m2"].to(device)
            v3 = b["vessel_m3"].to(device)
            order.append(b["case"][0])

            if int((v2 > 0).sum()) == 0:          # fail-safe: no vessels found
                loc_p[i] = fb_loc; ap_p[i] = fb_ap; n_fallback += 1
                continue

            views = [(img, v2, v3)]
            if not a.no_tta:
                views.append((torch.flip(img, dims=[4]),
                              torch.flip(v2, dims=[3]), torch.flip(v3, dims=[3])))
            al = np.zeros(spec.n_loc, np.float32); aa = 0.0
            for vi, vv2, vv3 in views:
                for m in models:
                    with torch.autocast("cuda", enabled=device.type == "cuda"):
                        o = m(vi, vv2, vv3)
                    al += torch.sigmoid(o["loc_logits"].float()).cpu().numpy()[0]
                    aa += float(torch.sigmoid(o["ap_logit"].float()).cpu().numpy().reshape(-1)[0])
            denom = len(views) * len(models)
            loc_p[i] = al / denom; ap_p[i] = aa / denom

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "aneurysm_present"] + spec.locations)
        for i, c in enumerate(order):
            w.writerow([c, f"{ap_p[i]:.6f}"] + [f"{v:.6f}" for v in loc_p[i]])
    print(f"wrote {a.out}" + (f"  ({n_fallback} fail-safe rows)" if n_fallback else ""))


if __name__ == "__main__":
    main()
