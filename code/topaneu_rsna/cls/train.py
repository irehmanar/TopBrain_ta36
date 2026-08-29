"""
ROI classifier training.

    python -m topaneu_rsna.cls.train --fold 0

Matches the author's schedule: 25 epochs, AdamW lr 1e-4 / wd 1e-2, warmup 4
epochs into cosine annealing down to 1e-5, effective batch 8 by accumulation,
EMA 0.995 evaluated each epoch.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from topaneu_rsna import config as C
from topaneu_rsna.cls.augment import BatchAugment, rasterize_spheres
from topaneu_rsna.cls.dataset import TopAneuRoiDataset, load_or_make_folds
from topaneu_rsna.cls.ema import ModelEMA
from topaneu_rsna.cls.losses import AneurysmLoss
from topaneu_rsna.cls.model import build_model


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def lr_at(step, total, warmup, base, start, eta_min):
    if step < warmup:
        return start + (base - start) * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return eta_min + (base - eta_min) * 0.5 * (1 + math.cos(math.pi * p))


def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])
            if 0 < y[:, j].sum() < len(y)]
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


@torch.no_grad()
def evaluate(model, loader, device, spec, amp):
    model.eval()
    ys, ps, ya, pa = [], [], [], []
    for b in loader:
        img = b["image"].to(device, non_blocking=True)
        v2 = b["vessel_m2"].to(device, non_blocking=True)
        v3 = b["vessel_m3"].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            o = model(img, v2, v3)
        ys.append(b["loc"].numpy()); ps.append(torch.sigmoid(o["loc_logits"].float()).cpu().numpy())
        ya.append(b["ap"].numpy()); pa.append(torch.sigmoid(o["ap_logit"].float()).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    auc_loc, n = macro_auc(y, p)
    auc_ap, _ = macro_auc(np.concatenate(ya).reshape(-1, 1),
                          np.concatenate(pa).reshape(-1, 1))
    return auc_loc, auc_ap, n, y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--cache_dir", type=Path, default=C.CLS_CACHE_DIR)
    ap.add_argument("--out_dir", type=Path, default=C.CLS_RESULTS_DIR)
    ap.add_argument("--seg_model_dir", type=Path,
                    default=None, help="defaults to the Model 2 results dir")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--resume", type=Path, default=None)
    a = ap.parse_args()

    cfg = C.CLS
    if a.epochs:
        cfg.epochs = a.epochs
    set_seed(cfg.seed + a.fold)

    spec = C.load_labels()
    seg_dir = a.seg_model_dir or C.seg_model_dir(C.DS_VESSEL, C.TRAINER_M2)
    splits, _ = load_or_make_folds(a.cache_dir, cfg.n_folds, cfg.seed, spec.n_loc)
    tr, va = splits[a.fold]["train"], splits[a.fold]["val"]
    print(f"fold {a.fold}: {len(tr)} train / {len(va)} val | "
          f"{spec.n_loc} locations, {spec.n_vessel} vessels, {spec.n_type} types")

    dl_tr = DataLoader(TopAneuRoiDataset(tr, a.cache_dir, spec), batch_size=cfg.batch_size,
                       shuffle=True, num_workers=cfg.num_workers, pin_memory=True,
                       drop_last=True, persistent_workers=cfg.num_workers > 0)
    dl_va = DataLoader(TopAneuRoiDataset(va, a.cache_dir, spec), batch_size=1,
                       shuffle=False, num_workers=max(1, cfg.num_workers // 2),
                       pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"
    model = build_model(spec, cfg, seg_dir, C.SEG_FOLD,
                        pretrained=not a.no_pretrained).to(device)
    ema = ModelEMA(model, cfg.ema_decay, cfg.ema_update_after_step) if cfg.ema else None

    criterion = AneurysmLoss(cfg).to(device)
    aug = BatchAugment(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    out_dir = Path(a.out_dir) / f"fold_{a.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    spe = max(1, len(dl_tr) // cfg.accumulate_grad_batches)
    total_steps, warmup = spe * cfg.epochs, spe * cfg.warmup_epochs

    start, best, gstep, hist = 0, -1.0, 0, []
    if a.resume and a.resume.exists():
        ck = torch.load(a.resume, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if ema and "ema" in ck:
            ema.ema.load_state_dict(ck["ema"])
        start, best, gstep = ck["epoch"] + 1, ck["best"], ck["gstep"]
        print(f"resumed @ epoch {start}")

    for epoch in range(start, cfg.epochs):
        model.train(); t0 = time.time(); run = {}
        opt.zero_grad(set_to_none=True)
        for it, b in enumerate(dl_tr):
            img = b["image"].to(device, non_blocking=True)
            v2 = b["vessel_m2"].to(device, non_blocking=True)
            v3 = b["vessel_m3"].to(device, non_blocking=True)
            pts = b["points"].to(device, non_blocking=True)
            img, v2, v3, pts = aug(img, v2, v3, pts)

            tgt = {"loc": b["loc"].to(device), "ap": b["ap"].to(device),
                   "sphere": rasterize_spheres(pts, img.shape[2:], cfg.sphere_radius, device)}
            if "typ" in b:
                tgt["typ"] = b["typ"].to(device)

            with torch.autocast("cuda", enabled=amp):
                o = model(img, v2, v3)
                loss, parts = criterion(o, tgt)
                loss = loss / cfg.accumulate_grad_batches

            scaler.scale(loss).backward()
            if (it + 1) % cfg.accumulate_grad_batches == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(gstep, total_steps, warmup, cfg.lr,
                                    cfg.warmup_start_lr, cfg.eta_min)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if ema:
                    ema.update(model)
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + float(v)

        n = max(1, len(dl_tr))
        msg = " ".join(f"{k}={v / n:.4f}" for k, v in run.items())
        eval_model = ema.ema if ema else model
        auc_loc, auc_ap, ncls, y, p = evaluate(eval_model, dl_va, device, spec, amp)
        score = float(np.nanmean([auc_loc, auc_ap]))
        print(f"[f{a.fold}] ep {epoch + 1}/{cfg.epochs} {msg} | "
              f"AUC_loc={auc_loc:.4f}({ncls}) AUC_ap={auc_ap:.4f} score={score:.4f} "
              f"| {time.time() - t0:.0f}s", flush=True)
        hist.append({"epoch": epoch + 1, "auc_loc": auc_loc, "auc_ap": auc_ap,
                     "score": score})

        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
              "best": best, "gstep": gstep, "n_loc": spec.n_loc,
              "seg_model_dir": str(seg_dir)}
        if ema:
            ck["ema"] = ema.state_dict()
        torch.save(ck, out_dir / "last.pt")
        if not np.isnan(score) and score > best:
            best = score; ck["best"] = best
            torch.save(ck, out_dir / "best.pt")
            np.savez(out_dir / "oof.npz", y=y, p=p, cases=np.array(va))
        (out_dir / "history.json").write_text(json.dumps(hist, indent=2))

    print(f"best fold {a.fold}: {best:.4f}")


if __name__ == "__main__":
    main()
