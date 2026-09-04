"""
Dry-run check: will `nnUNetv2_train ... -pretrained_weights <ckpt>` actually
load, without spending a GPU allocation to find out.

Builds the exact target network architecture from the dataset's own
plans.json (pure CPU, no data, no GPU -- just constructs the nn.Module graph)
and runs the *same* key-and-shape validation load_pretrained_weights.py uses,
against every key at once -- so unlike nnU-Net's own assert (which stops at
the first mismatch), this reports every problem in one pass.

    python -m topaneu_rsna.seg.verify_pretrained_weights \\
        --dataset 307 --trainer RSNA2025Trainer_moreDAv7 \\
        --pretrained $SCRATCH_ROOT/work/checkpoints/model1_2ch_checkpoint_final.pth
"""
from __future__ import annotations

import argparse
import json

import torch

from topaneu_rsna import config as C

SKIP_STRINGS = [".seg_layers."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, required=True)
    ap.add_argument("--trainer", required=True, help="only used for the printed summary")
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--pretrained", required=True)
    a = ap.parse_args()

    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    ds_root = C.nnUNet_preprocessed / f"Dataset{a.dataset:03d}_{C.DS_NAMES[a.dataset]}"
    plans = json.loads((ds_root / f"{a.plans}.json").read_text())
    dataset_json = json.loads((ds_root / "dataset.json").read_text())

    plans_manager = PlansManager(plans)
    cfg = plans_manager.get_configuration(a.configuration)
    label_manager = plans_manager.get_label_manager(dataset_json)

    num_input_channels = determine_num_input_channels(plans_manager, cfg, dataset_json)
    network = get_network_from_plans(
        cfg.network_arch_class_name, cfg.network_arch_init_kwargs,
        cfg.network_arch_init_kwargs_req_import,
        num_input_channels, label_manager.num_segmentation_heads,
        allow_init=True, deep_supervision=True)

    model_dict = network.state_dict()
    saved = torch.load(a.pretrained, map_location="cpu", weights_only=False)
    pretrained_dict = saved["network_weights"]

    print(f"Dataset{a.dataset} / {a.trainer} -- built network expects "
         f"{num_input_channels} input channel(s), {label_manager.num_segmentation_heads} "
         f"segmentation head(s)\n")

    missing, mismatched, ok = [], [], 0
    for key in model_dict:
        if any(s in key for s in SKIP_STRINGS):
            continue
        if key not in pretrained_dict:
            missing.append(key)
        elif model_dict[key].shape != pretrained_dict[key].shape:
            mismatched.append((key, tuple(pretrained_dict[key].shape), tuple(model_dict[key].shape)))
        else:
            ok += 1

    print(f"{ok} key(s) match, {len(missing)} missing, {len(mismatched)} shape-mismatched\n")

    if missing:
        print("MISSING from checkpoint:")
        for k in missing:
            print(f"  {k}")
    if mismatched:
        print("SHAPE MISMATCH (checkpoint -> expected):")
        for k, pshape, mshape in mismatched:
            print(f"  {k}: {pshape} -> {mshape}")

    if not missing and not mismatched:
        print("PASS -- nnUNetv2_train's load_pretrained_weights() will load this checkpoint cleanly.")
    else:
        print("\nFAIL -- this would still crash the real training job. Do not submit yet.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
