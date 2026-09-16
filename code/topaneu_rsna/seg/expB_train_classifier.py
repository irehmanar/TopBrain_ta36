"""
Experiment B, step 2/3: train the standalone 3D crop classifier. Plain
PyTorch Dataset/DataLoader/training loop -- deliberately NOT wired into
nnU-Net's trainer/dataloader machinery, since this is fixed-size crop
classification, a different shape of problem than the full-volume
segmentation nnU-Net's dataloader is built for.

Data: seg/build_expB_crop_dataset.py's cached .npz crops (3-channel: raw
image z-scored, real binary aneurysm prediction, real vessel prediction) and
manifest CSV. Train = every instance whose case is NOT in config.EXPB_VAL_FOLD;
val = every instance whose case IS in that fold.

Warm start: Dataset304's fold_4 checkpoint specifically (fold 4 == this
experiment's own held-out val fold) -- not any other fold. This matters:
Dataset304's fold_0 checkpoint, for example, WAS trained on fold 4's raw
images (as a segmentation task), which would make "held out" fold 4 not
actually novel to the network's pretrained weights. fold_4's own checkpoint
never saw fold 4 during its own segmentation training either, so this is the
one choice that keeps this experiment's held-out fold genuinely unseen by
every component of the pipeline, not just this script's own fine-tuning.
Uses expand_pretrained_channels.py's already-produced 3-channel checkpoint
(1->3 channels: Dataset304 is single-channel, this classifier needs 3) --
see 91_verify_expB_warmstart.sbatch for that step.

Architecture: nnU-Net's own ResEncUNetM encoder (built directly via
get_network_from_plans from Dataset304's plans.json, not through
nnUNetTrainer) -> global average pool -> FC(512)+Dropout(0.3)+ReLU ->
FC(256)+Dropout(0.3)+ReLU -> FC(53) [52 locations + background]. The whole
encoder is fine-tuned, not frozen -- this is a much smaller forward/backward
(encoder-only, no U-Net decode, no deep supervision) than Experiment A's
segmentation training, so GPU memory pressure should be far lower.

Class imbalance: class-balanced loss via the "effective number of samples"
method (Cui et al., CVPR 2019) -- NOT flat inverse-frequency, which
overreacts for the 1-2-example classes this problem is full of. Per-class
weight = (1-beta)/(1-beta^n_c), beta=0.999 (the paper's own recommended
default for long-tailed problems at this kind of imbalance ratio).

Logs macro-averaged validation recall every epoch, and the full 53-class
recall breakdown every 10 epochs (not just accuracy, and not only at the
end) -- so a class getting genuinely learned vs. papered over by common-
class performance is visible during the run, not discovered after.

    python -m topaneu_rsna.seg.expB_train_classifier
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from topaneu_rsna import config as C


class CropDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        d = np.load(r["npz_path"])
        crop = torch.from_numpy(d["crop"])          # (3, D, H, W) float32
        label = int(d["label"])
        return crop, label


class CropClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_channels: int, n_classes: int,
                dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder_channels, 512), nn.Dropout(dropout), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.Dropout(dropout), nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        feat = self.encoder(x)[-1]             # (B, C, d, h, w) bottleneck
        pooled = feat.mean(dim=(2, 3, 4))       # (B, C)
        return self.head(pooled)


def build_model(expanded_ckpt: Path, dataset304_model_dir: Path, n_classes: int,
                device: torch.device) -> CropClassifier:
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    plans = json.loads((dataset304_model_dir / "plans.json").read_text())
    arch = plans["configurations"]["3d_fullres"]["architecture"]
    network = get_network_from_plans(
        arch_class_name=arch["network_class_name"], arch_kwargs=arch["arch_kwargs"],
        arch_kwargs_req_import=arch["_kw_requires_import"],
        input_channels=3, output_channels=1, allow_init=True, deep_supervision=False)

    state = torch.load(expanded_ckpt, map_location="cpu", weights_only=False)
    weights = state.get("network_weights", state)
    weights = {k: v for k, v in weights.items() if not k.startswith("decoder.seg_layers.")}
    missing, unexpected = network.load_state_dict(weights, strict=False)
    print(f"[warm-start] loaded {expanded_ckpt}\n"
         f"[warm-start] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 20:
        raise RuntimeError("Too many missing keys -- checkpoint architecture doesn't "
                           "match Dataset304's plans.json. Check --expanded_ckpt.")

    encoder_channels = int(network.encoder.output_channels[-1])
    model = CropClassifier(network.encoder, encoder_channels, n_classes).to(device)
    return model


def effective_number_weights(class_counts: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Cui et al., 'Class-Balanced Loss Based on Effective Number of Samples',
    CVPR 2019. Classes with zero training examples get max weight (as if
    count=1) -- they can't actually be learned regardless of weight, this
    just avoids a divide-by-zero."""
    counts = np.maximum(class_counts, 1)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(counts)
    return weights.astype(np.float32)


def load_manifest(manifest_csv: Path) -> list[dict]:
    with open(manifest_csv, newline="") as f:
        return list(csv.DictReader(f))


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    tp = np.zeros(n_classes); support = np.zeros(n_classes)
    for crops, labels in loader:
        crops, labels = crops.to(device), labels.to(device)
        logits = model(crops)
        pred = logits.argmax(1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        for c in range(n_classes):
            mask = labels_np == c
            support[c] += mask.sum()
            tp[c] += (pred[mask] == c).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = tp / support
    macro_recall = float(np.nanmean(recall))
    return recall, macro_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", type=Path, default=C.EXPB_MANIFEST_CSV)
    ap.add_argument("--expanded_ckpt", type=Path,
                    default=C.WORK / "checkpoints" / "expB_dataset304_fold4_3ch_checkpoint_final.pth")
    ap.add_argument("--dataset304_model_dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--beta", type=float, default=0.999)
    ap.add_argument("--out_dir", type=Path, default=C.EXPB_CKPT_DIR)
    a = ap.parse_args()

    spec = C.load_labels()
    n_classes = spec.n_loc + 1   # + background
    dataset304_model_dir = a.dataset304_model_dir or (
        C.seg_model_dir(C.DS_ANEURYSM, C.TRAINER_LOC))

    rows = load_manifest(a.manifest_csv)
    train_rows = [r for r in rows if int(r["fold"]) != C.EXPB_VAL_FOLD]
    val_rows = [r for r in rows if int(r["fold"]) == C.EXPB_VAL_FOLD]
    print(f"{len(train_rows)} training instances (folds != {C.EXPB_VAL_FOLD}), "
         f"{len(val_rows)} validation instances (fold {C.EXPB_VAL_FOLD})")
    assert not (set(r["case"] for r in train_rows) & set(r["case"] for r in val_rows)), \
        "LEAKAGE: a case appears in both train and val -- stopping"
    print("leakage check PASSED: no case appears in both train and val")

    class_counts = np.zeros(n_classes)
    for r in train_rows:
        class_counts[int(r["true_id"])] += 1
    print(f"training instances per class (id 0 = background): {class_counts.astype(int).tolist()}")
    weights = effective_number_weights(class_counts, beta=a.beta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(a.expanded_ckpt, dataset304_model_dir, n_classes, device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    dl_tr = DataLoader(CropDataset(train_rows), batch_size=a.batch_size, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True)
    dl_val = DataLoader(CropDataset(val_rows), batch_size=a.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    best_macro_recall = -1.0
    for epoch in range(a.epochs):
        model.train()
        total_loss = 0.0
        for crops, labels in dl_tr:
            crops, labels = crops.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(crops)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * crops.shape[0]
        sched.step()

        recall, macro_recall = evaluate(model, dl_val, device, n_classes)
        print(f"epoch {epoch + 1}/{a.epochs}: train_loss={total_loss / len(train_rows):.4f} "
             f"val_macro_recall={macro_recall:.4f} lr={sched.get_last_lr()[0]:.2e}")
        if (epoch + 1) % 10 == 0:
            print(f"  per-class val recall: {np.round(recall, 3).tolist()}")

        ck = {"model": model.state_dict(), "epoch": epoch, "macro_recall": macro_recall,
             "n_classes": n_classes}
        torch.save(ck, a.out_dir / "last.pt")
        if not np.isnan(macro_recall) and macro_recall > best_macro_recall:
            best_macro_recall = macro_recall
            torch.save(ck, a.out_dir / "best.pt")

    print(f"\nbest val macro recall: {best_macro_recall:.4f}")


if __name__ == "__main__":
    main()
