"""
Dry-run check: will `nnUNetv2_train <dataset> <config> <fold> -tr <trainer>`
even get past trainer class lookup + __init__, without spending a GPU
allocation to find out.

nnU-Net finds trainer classes by recursively importing every .py file under
nnunetv2/training/nnUNetTrainer/ until one has an attribute matching the
requested class name (recursive_find_python_class in find_class_by_name.py).
That import scan runs regardless of which trainer you actually asked for, so
an unrelated broken import anywhere in that tree (e.g. swinunetr_trainer.py's
now-guarded `from src...` -- see its own try/except) can non-deterministically
break lookup of any trainer, depending on filesystem directory-listing order
on whichever node the job lands on. This calls the exact same
get_trainer_from_args() nnUNetv2_train uses, forcing device="cpu" so it never
touches the GPU, to confirm the lookup + trainer __init__ (loss construction,
num_epochs/save_every overrides, etc.) succeed before submitting the real job.

    python -m topaneu_rsna.seg.verify_trainer_lookup \\
        --dataset 312 --trainer RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07_ep250

Pass --also_test_dataloaders to additionally call trainer.get_dataloaders() and
pull one real batch through it -- still CPU-only (no GPU needed for data
loading), but exercises any custom get_dataloaders()/sampling logic a trainer
overrides (e.g. class_balanced_focal.py's per-case class-balanced sampling,
which reads every training case's .pkl off disk) before it ever gets to spend
a GPU allocation finding out it has a bug.
"""
from __future__ import annotations

import argparse

from topaneu_rsna import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, required=True)
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--also_test_dataloaders", action="store_true")
    a = ap.parse_args()

    import torch
    from nnunetv2.run.run_training import get_trainer_from_args

    fold = "all" if a.fold == "all" else int(a.fold)

    print(f"Looking up trainer '{a.trainer}' for Dataset{a.dataset:03d} "
         f"({a.configuration}, plans={a.plans}, fold={fold})...")

    try:
        trainer = get_trainer_from_args(
            str(a.dataset), a.configuration, fold, a.trainer, a.plans,
            device=torch.device("cpu"),
        )
        # Exercises nearly everything run_training() does before the actual
        # training loop -- network construction, loss construction, deep
        # supervision setup -- all CPU-only (no data loading, no GPU compute).
        trainer.initialize()
    except Exception:
        print("\nFAIL -- trainer lookup/construction raised an exception "
             "(see traceback above). Do not submit the GPU job yet.")
        raise

    print(f"Trainer class:    {type(trainer).__name__}")
    print(f"num_epochs:       {trainer.num_epochs}")
    print(f"save_every:       {trainer.save_every}")
    print(f"batch_size:       {trainer.configuration_manager.batch_size}")
    print(f"patch_size:       {trainer.configuration_manager.patch_size}")
    print(f"num_input_ch:     {trainer.num_input_channels}")
    print(f"num_seg_heads:    {trainer.label_manager.num_segmentation_heads}")
    print(f"loss:             {type(trainer.loss).__name__}")

    if a.also_test_dataloaders:
        print("\nBuilding dataloaders (this reads every training case's .pkl "
             "off disk if the trainer overrides sampling -- may take a bit)...")
        try:
            dl_tr, dl_val = trainer.get_dataloaders()
            batch = next(dl_tr)
            print(f"Train batch data shape:   {tuple(batch['data'].shape)}")
            target = batch['target']
            target_shape = tuple(target[0].shape) if isinstance(target, list) else tuple(target.shape)
            print(f"Train batch target shape: {target_shape}"
                 + (" (deep supervision -- showing highest-res head)" if isinstance(target, list) else ""))
        except Exception:
            print("\nFAIL -- get_dataloaders()/first batch raised an exception "
                 "(see traceback above). Do not submit the GPU job yet.")
            raise

    print("\nPASS -- trainer lookup and full CPU-side initialization succeeded.")


if __name__ == "__main__":
    main()
