"""
Experiment A, step 6: the paper's Task-1 classification head (Sect 2.2),
trained standalone on job 84's cached frozen-encoder features -- the
decoupled counterpart to the paper's bottleneck-attached branch trained
jointly with segmentation (see expA_vessel_cond_seg.py's module docstring
for why this pipeline splits the two).

Architecture, copied exactly from the paper minus the vessel-gating
attention branch (that branch modulates the SAME bottleneck features this
network is already conditioned on via Dataset313's second input channel, so
gating it again post-hoc on an already-vessel-aware feature vector would be
redundant, not a simplification with a real cost):
    global-average-pooled encoder feature (already pooled by
    expA_extract_features.py -- this script trains on the pooled vector, not
    a 3D feature map, so no separate pooling layer here)
    -> FC(512), Dropout(0.3), ReLU
    -> FC(256), Dropout(0.3), ReLU
    -> FC(52), sigmoid (applied by the loss / at inference, not in the module)

Loss: BCE-Focal, per-class independent binary presence (gamma=2.0, alpha=0.25
-- the paper names "BCE-Focal" for this term but only gives gamma/alpha for
the SEGMENTATION focal term in Sect 2.3; reusing the same values here rather
than inventing new unstated ones is a documented assumption, not a paper
value). weight_decay + dropout are the only regularization needed at this
scale (a few hundred pooled feature vectors, a two-layer MLP).

Targets: per-case 52-dim multi-label presence, derived directly from
LOCATION_MASKS (which of the 52 location classes has >=1 voxel in this
case's ground truth) -- exactly what the paper's own classification head is
trained against.

    python -m topaneu_rsna.seg.expA_train_classifier --train
    python -m topaneu_rsna.seg.expA_train_classifier --predict holdout
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio

CKPT_PATH = C.WORK / "expA_classifier" / "head.pt"


class ClsHead(nn.Module):
    def __init__(self, in_dim: int, n_loc: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.Dropout(dropout), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.Dropout(dropout), nn.ReLU(inplace=True),
            nn.Linear(256, n_loc),
        )

    def forward(self, x):
        return self.net(x)


class BCEFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, target):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def case_presence_labels(case: str, spec) -> np.ndarray:
    gt, _ = uio.read(C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}")
    present = np.unique(gt)
    y = np.zeros(spec.n_loc, dtype=np.float32)
    for v in present:
        if 1 <= v <= spec.n_loc:
            y[int(v) - 1] = 1.0
    return y


def load_features(cases: list[str], feature_dir: Path) -> tuple[np.ndarray, list[str]]:
    feats, kept = [], []
    for c in cases:
        p = feature_dir / f"{c}.npy"
        if p.exists():
            feats.append(np.load(p))
            kept.append(c)
    return np.stack(feats).astype(np.float32), kept


def train(a):
    spec = C.load_labels()
    split = json.loads(C.EXPA_HOLDOUT_JSON.read_text())
    X, cases = load_features(split["train"], a.feature_dir)
    Y = np.stack([case_presence_labels(c, spec) for c in cases])
    print(f"{len(cases)} training cases, feature dim {X.shape[1]}, "
         f"{Y.sum(0).astype(int).tolist()} positives per class")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClsHead(X.shape[1], spec.n_loc, dropout=a.dropout).to(device)
    loss_fn = BCEFocalLoss(gamma=a.focal_gamma, alpha=a.focal_alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    n = Xt.shape[0]

    model.train()
    for epoch in range(a.epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, a.batch_size):
            idx = perm[i:i + a.batch_size]
            opt.zero_grad(set_to_none=True)
            logits = model(Xt[idx])
            loss = loss_fn(logits, Yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{a.epochs}: mean BCE-focal loss = {total / n:.4f}")

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": X.shape[1],
               "n_loc": spec.n_loc, "dropout": a.dropout}, CKPT_PATH)
    print(f"classifier head saved to {CKPT_PATH}")


def predict(a, split_name: str):
    spec = C.load_labels()
    split = json.loads(C.EXPA_HOLDOUT_JSON.read_text())
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = ClsHead(ckpt["in_dim"], ckpt["n_loc"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X, cases = load_features(split[split_name], a.feature_dir)
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X))).numpy()

    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case"] + spec.locations)
        for case, row in zip(cases, probs):
            w.writerow([case] + [f"{p:.4f}" for p in row])
    print(f"{len(cases)} cases' per-class probabilities written to {a.out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", type=Path, default=C.EXPA_FEATURE_CACHE)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=3e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", choices=["train", "holdout"], default=None)
    ap.add_argument("--out_csv", type=Path,
                    default=C.LOG_ROOT / "task2_expA_classifier_probs.csv")
    a = ap.parse_args()

    if a.train:
        train(a)
    if a.predict:
        predict(a, a.predict)


if __name__ == "__main__":
    main()
