"""
Experiment A, step 5: cache one global-average-pooled encoder feature vector
per case, from Dataset313's own trained (frozen) segmentation backbone --
feeds the small classification head trained by seg/expA_train_classifier.py.

Reuses cls/backbone.py's NnUNetTruncatedBackbone entirely for network
construction + checkpoint loading (same plans.json-driven
get_network_from_plans() call already proven for the RSNA classifier's own
Model-2-backed backbone; the seg-head key mismatch it already drops via
`if not k.startswith("decoder.seg_layers.")` is exactly what's needed here
too, no new loading logic required). The one deliberate deviation from that
class's own forward(): this script calls `backbone.nnunet.encoder(x)`
directly instead of `backbone.forward(x)`, skipping NnUNetTruncatedBackbone's
own decoder branch (built for a different, sphere-localization auxiliary
task this experiment doesn't use) entirely -- a full decode pass over a
whole, native-resolution head volume is a real, avoidable memory cost for a
feature this cheap to get.

Correctness note: the encoder was trained on nnU-Net-preprocessed data
(resampled to the plans' target spacing, per-channel normalized per the
dataset fingerprint) -- feeding it raw, un-preprocessed voxel values directly
would silently put every feature on the wrong distribution with no error or
warning to catch it. This script therefore runs each case through nnU-Net's
own DefaultPreprocessor.run_case() (the exact function nnUNetv2_preprocess
and nnUNetv2_predict both call internally) before the encoder forward pass,
rather than hand-rolling a normalization scheme that might not match.

Known risk: an encoder forward pass over a full native-resolution volume
(rather than a 128x256x256 training patch) has never been run in this
pipeline before -- if this OOMs on Narval, the fix is a center-crop or
downsample of the preprocessed array before the forward pass, not a code
change to the preprocessing step itself; note the failure and the volume
shape in the job's .err log before deciding.

    python -m topaneu_rsna.seg.expA_extract_features --split all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from topaneu_rsna import config as C
from topaneu_rsna.cls.backbone import NnUNetTruncatedBackbone


def build_backbone_and_preprocessor(device):
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    model_dir = C.seg_model_dir(C.DS_VESSELCOND_SEG, C.TRAINER_EXPA_SEG)
    backbone = NnUNetTruncatedBackbone(
        nnunet_model_dir=model_dir, fold=C.SEG_FOLD, pretrained=True,
        checkpoint_name="checkpoint_final.pth", configuration="3d_fullres",
        in_channels=2, num_truncate_stages=1)
    backbone.to(device).eval()

    plans = json.loads((model_dir / "plans.json").read_text())
    dataset_json = json.loads((model_dir / "dataset.json").read_text())
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration("3d_fullres")
    preprocessor = DefaultPreprocessor()
    return backbone, preprocessor, plans_manager, configuration_manager, dataset_json


def extract_one(backbone, device, preprocessor, plans_manager, configuration_manager,
                dataset_json, img_p: Path, cond_p: Path) -> np.ndarray:
    data, _, _ = preprocessor.run_case(
        [str(img_p), str(cond_p)], None, plans_manager, configuration_manager, dataset_json)
    x = torch.from_numpy(data[None].astype(np.float32)).to(device)   # (1, 2, D, H, W)
    with torch.no_grad():
        feat = backbone.nnunet.encoder(x)[-1]           # (1, C, d, h, w) bottleneck
        pooled = feat.mean(dim=(2, 3, 4))[0]             # (C,)
    return pooled.cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "holdout", "all"], default="all")
    ap.add_argument("--out_dir", type=Path, default=C.EXPA_FEATURE_CACHE)
    a = ap.parse_args()

    split = json.loads(C.EXPA_HOLDOUT_JSON.read_text())
    cases = (split["train"] + split["holdout"] if a.split == "all"
            else split[a.split])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, preprocessor, plans_manager, configuration_manager, dataset_json = \
        build_backbone_and_preprocessor(device)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip = 0, 0
    for case in cases:
        out_p = a.out_dir / f"{case}.npy"
        if out_p.exists():
            n_ok += 1
            continue
        img_p = C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}"
        cond_p = C.VESSEL_PRED_M2_FULLHEAD / f"{case}.nii.gz"
        if not (img_p.exists() and cond_p.exists()):
            print(f"[skip] {case}: missing image or vessel prediction")
            n_skip += 1
            continue
        try:
            feat = extract_one(backbone, device, preprocessor, plans_manager,
                               configuration_manager, dataset_json, img_p, cond_p)
        except RuntimeError as e:
            print(f"[FAIL] {case}: {e} -- likely OOM on full-volume encoder forward, "
                 f"see this script's module docstring for the mitigation")
            n_skip += 1
            continue
        np.save(out_p, feat)
        n_ok += 1

    print(f"{n_ok} feature vectors cached under {a.out_dir} ({n_skip} skipped)")


if __name__ == "__main__":
    main()
