"""
Fix a real shape mismatch left by reusing Dataset304's plan wholesale for
Dataset317. nnUNetv2_move_plans_between_datasets (job 174) copies the ENTIRE
source plan dict as-is -- it does not know or care that Dataset304 has 1
input channel and Dataset317 has 2, so every configuration's
normalization_schemes/use_mask_for_norm lists come over at length 1, and the
preprocessor indexes these lists per channel. Left unpatched, preprocessing
either crashes with an index error on channel 1, or (depending on the nnU-Net
version's zip/broadcast behavior) silently mis-normalizes it -- neither is
acceptable for an experiment whose whole point is a clean, correct 2-channel
comparison.

This extends both lists to length 2 (channel 1 = the vessel mask gets the
same ZScoreNormalization as channel 0 -- the only scheme this project uses
anywhere, including Dataset305's own real, previously-trained 2-channel
vessel-conditioned dataset) and adds a real channel-1 entry to
foreground_intensity_properties_per_channel, taken from Dataset317's own
freshly extracted fingerprint (job 174 already runs
nnUNetv2_extract_fingerprint -d 317 before this), not copied from Dataset304.

Run this AFTER nnUNetv2_move_plans_between_datasets and BEFORE
nnUNetv2_preprocess (see jobs/43_vessel_cond_binary_controlled/174).

    python -m topaneu_rsna.seg.patch_vesselcond_plan
"""
from __future__ import annotations

import argparse
import json

from topaneu_rsna import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", type=int, default=C.DS_ANEURYSM_VESSELCOND)
    ap.add_argument("--plans_name", default="nnUNetResEncUNetMPlans")
    a = ap.parse_args()

    dataset_name = f"Dataset{a.dataset_id:03d}_{C.DS_NAMES[a.dataset_id]}"
    preprocessed_dir = C.nnUNet_preprocessed / dataset_name
    plan_path = preprocessed_dir / f"{a.plans_name}.json"
    fingerprint_path = preprocessed_dir / "dataset_fingerprint.json"

    plan = json.loads(plan_path.read_text())
    fingerprint = json.loads(fingerprint_path.read_text())

    n_channels = 2  # image + vessel mask -- this dataset's own dataset.json declares both
    fixed_configs = []
    for cfg_name, cfg in plan["configurations"].items():
        if "normalization_schemes" not in cfg:
            continue  # e.g. 3d_cascade_fullres just inherits, nothing to patch
        before = len(cfg["normalization_schemes"])
        if before >= n_channels:
            continue  # already correct length, don't touch
        cfg["normalization_schemes"] = cfg["normalization_schemes"] * n_channels
        cfg["use_mask_for_norm"] = cfg["use_mask_for_norm"] * n_channels
        fixed_configs.append((cfg_name, before, n_channels))

    fp_props = fingerprint.get("foreground_intensity_properties_per_channel", {})
    if "1" not in plan.get("foreground_intensity_properties_per_channel", {}) and "1" in fp_props:
        plan["foreground_intensity_properties_per_channel"]["1"] = fp_props["1"]
        print(f"[patch] added real channel-1 foreground_intensity_properties_per_channel "
             f"from Dataset317's own fingerprint")

    plan_path.write_text(json.dumps(plan, indent=4))

    if fixed_configs:
        for cfg_name, before, after in fixed_configs:
            print(f"[patch] {cfg_name}: normalization_schemes/use_mask_for_norm "
                 f"{before} -> {after} channels")
    else:
        print("[patch] nothing to fix -- all configurations already had 2-channel "
             "normalization lists")
    print(f"[patch] wrote {plan_path}")


if __name__ == "__main__":
    main()
