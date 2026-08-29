"""Dataset + multilabel-stratified folds over the stage-2 npz cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MAX_POINTS = 16


class TopAneuRoiDataset(Dataset):
    def __init__(self, cases, cache_dir, spec, train=True):
        self.cases = list(cases)
        self.dir = Path(cache_dir)
        self.spec = spec
        self.train = train

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        case = self.cases[i]
        z = np.load(self.dir / f"{case}.npz")
        out = {
            "case": case,
            "image": torch.from_numpy(z["image"].astype(np.float32)),
            "vessel_m2": torch.from_numpy(z["vessel_m2"].astype(np.int64)),
            "vessel_m3": torch.from_numpy(z["vessel_m3"].astype(np.int64)),
        }
        if self.train:
            loc = z["loc"].astype(np.float32) if "loc" in z else np.zeros(self.spec.n_loc, np.float32)
            out["loc"] = torch.from_numpy(loc)
            out["ap"] = torch.tensor(float(loc.max() if loc.size else 0.0))
            if self.spec.n_type:
                typ = z["typ"].astype(np.float32) if "typ" in z else np.zeros(self.spec.n_type, np.float32)
                out["typ"] = torch.from_numpy(typ)
            pts = z["points"].astype(np.float32) if "points" in z else np.zeros((0, 3), np.float32)
            padded = np.full((MAX_POINTS, 3), np.nan, np.float32)
            if len(pts):
                padded[: min(len(pts), MAX_POINTS)] = pts[:MAX_POINTS]
            out["points"] = torch.from_numpy(padded)
        return out


def load_or_make_folds(cache_dir, n_folds, seed, n_loc):
    cache_dir = Path(cache_dir)
    p = cache_dir / "splits.json"
    cases = sorted(x.stem for x in cache_dir.glob("*.npz"))
    if p.exists():
        return json.loads(p.read_text()), cases

    Y = np.zeros((len(cases), n_loc + 1), np.int64)
    for i, c in enumerate(cases):
        z = np.load(cache_dir / f"{c}.npz")
        if "loc" in z:
            Y[i, :n_loc] = z["loc"]
            Y[i, n_loc] = int(z["loc"].max())

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        kf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits_iter = kf.split(np.zeros(len(cases)), Y)
    except ImportError:
        # iterative-stratification not installed -> stratify on presence only
        from sklearn.model_selection import StratifiedKFold
        print("[folds] iterative-stratification missing, "
              "falling back to presence-only stratification")
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits_iter = kf.split(np.zeros(len(cases)), Y[:, n_loc])

    splits = [{"train": [cases[i] for i in tr], "val": [cases[i] for i in va]}
              for tr, va in splits_iter]
    p.write_text(json.dumps(splits, indent=2))
    return splits, cases
