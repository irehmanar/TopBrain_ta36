# TopAneu Task 2 → TopBrain TA36 Vessel Segmentation Pipeline

Code for a coarse-to-fine aneurysm detection and location-assignment
pipeline built for [TopAneu 2026](https://topaneu2026.grand-challenge.org/)
Task 2, whose vessel-segmentation component ("Model 2") was separately
submitted to the [TopBrain 2026 TA36 track](https://topbrain2026.grand-challenge.org/)
— evaluated: mean Dice 0.763 on the TA36 final-test leaderboard.

This repo contains the full research codebase (dataset builders, nnU-Net
trainer variants, the rule-based location-assignment engine, evaluation
scripts) with real commit history. It does **not** contain the packaged
TA36 submission container, cluster job scripts, trained weights, or the
challenge data itself — see "What's not here" below for why, and where to
find each one instead.

## What's not here, and why

| Excluded | Where to find it instead |
|---|---|
| `jobs/` — SLURM submission scripts for this project's own compute cluster (Compute Canada / Narval) | Removed from history entirely (cluster-specific, not portable). Every reproduction step below is the underlying command those scripts ran, so nothing is lost, just de-scripted. |
| `docker/` — the packaged TA36 submission container (`inference.py`, `main.py`, `Dockerfile`) | A separate, submission-only repo: **[TopBrain-TA36-Vessel-Segmentation](https://github.com/irehmanar/TopBrain_ta36)** *(adjust link if this is hosted elsewhere)* |
| Trained model weights (`*.pth`) | Not redistributed here due to size (~780 MB for Model 2 alone). Available via the TA36 Algorithm page or on request. |
| TopAneu / TopBrain imaging data | Proprietary/challenge data — request access via the respective challenge's own data-access process. Not ours to redistribute. |
| `logs/`, training run outputs | Cluster-specific run artifacts, not needed to reproduce the pipeline itself. |

## Repository layout

```
code/
  topaneu_rsna/            This project's own package
    config.py               All dataset IDs, paths, trainer names, hyperparameters
    labels.json              36-vessel and 52-location label definitions
    prep/                    Dataset construction (nnU-Net raw layout builders)
    seg/                     Segmentation dataset builders, the rule-based
                             location-assignment engine, evaluation scripts
    cls/                     RSNA-style presence/location classifier (Task 1 side)
    viz/                     Figure-generation scripts
    utils/                   Shared geometry/IO helpers
  rsna2025_1st_place/       Vendored nnU-Net fork this project builds on
    nnUNet/                  nnU-Net v2 source, with custom trainer classes under
                             nnunetv2/training/nnUNetTrainer/project_specific/rsna2025/
    pip_packages/requirements.txt   Full dependency list
setup_env.sh                Cluster environment setup (paths, venv activation)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r code/rsna2025_1st_place/pip_packages/requirements.txt
pip install -e code/rsna2025_1st_place/nnUNet
```

Set the following environment variables (see `setup_env.sh` for the exact
convention this project uses; adjust paths for your own environment):

```bash
export TOPANEU_DATA=/path/to/topaneu/data          # raw challenge data (see below)
export TOPBRAIN_DATA=/path/to/topbrain/data        # only needed for TA36 fine-tuning experiments
export nnUNet_raw=/path/to/scratch/nnUNet_raw
export nnUNet_preprocessed=/path/to/scratch/nnUNet_preprocessed
export nnUNet_results=/path/to/scratch/nnUNet_results
export PYTHONPATH="$(pwd)/code/rsna2025_1st_place/nnUNet:$(pwd)/code:$PYTHONPATH"
```

`config.py` reads every one of these with sensible fallbacks — see its
`_p()` helper and the `TOPANEU_ROOT`/`TOPANEU_DATA`/`nnUNet_*` block at the
top of the file for the authoritative list.

### Expected raw data layout

Under `$TOPANEU_DATA`:

```
images/<case>_0000.nii.gz              CTA or MRA volume
location_masks/<case>.nii.gz           52-class aneurysm-location labels
vessel_masks/<case>.nii.gz             36-class vessel labels
location_mapping.json, vessel_mapping.json
```

`code/topaneu_rsna/labels.json` defines the exact 36 vessel names and 52
location names and their integer label values.

## Reproducing the pipeline

Each stage below is a dataset-build step (writes nnU-Net raw format) plus
the exact `nnUNetv2_*` commands. All trainer classes referenced here live
in `code/rsna2025_1st_place/nnUNet/nnunetv2/training/nnUNetTrainer/project_specific/rsna2025/`.

### 1. Model 1 — coarse vessel-group localization (Dataset301)

3 coarse vessel groups, 1mm isotropic, used to crop a region of interest
for the fine stage.

```bash
python -m topaneu_rsna.prep.build_nnunet_datasets --stage coarse
nnUNetv2_plan_and_preprocess -d 301 -pl nnUNetPlannerResEncMForcedLowres -c 3d_fullres --verify_dataset_integrity
nnUNetv2_train 301 3d_fullres all -p nnUNetResEncUNetMPlans -tr RSNA2025Trainer_moreDAv7
```

### 2. Model 2 — 36-class vessel segmentation (Dataset302)

**This is the model submitted to TopBrain TA36.** Full 3D resolution,
whole-head, native per-case grid — no cropping.

```bash
python -m topaneu_rsna.prep.build_nnunet_datasets --stage vessel
nnUNetv2_plan_and_preprocess -d 302 -pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity
nnUNetv2_train 302 3d_fullres all -p nnUNetResEncUNetMPlans -tr RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07
```

Trained fold "all" (no held-out split) deliberately — see `config.py`'s
`SEG_FOLD` constant and its accompanying comment for the reasoning.

Per-class Dice trajectory is visible directly in nnU-Net's own training
log (`Pseudo dice` printed each epoch); a pooled comparison across
different trainer/loss variants can be built with:

```bash
python -m topaneu_rsna.viz.visualize_vessel_overlap --n_cases 5
```

### 3. Binary aneurysm segmentation (Dataset304)

Whole-head, single channel, real 5-fold cross-validation. Aneurysm
location labels are collapsed to one binary foreground class; per-location
assignment happens downstream (step 4), not by this model.

```bash
python -m topaneu_rsna.prep.build_nnunet_datasets --stage aneurysm
nnUNetv2_plan_and_preprocess -d 304 -pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity
for FOLD in 0 1 2 3 4; do
    nnUNetv2_train 304 3d_fullres $FOLD -p nnUNetResEncUNetMPlans -tr RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07
done
```

Evaluate the pooled 5-fold result:

```bash
python -m topaneu_rsna.seg.evaluate_location --dataset 304 --out seg_eval_aneurysm.csv
```

### 4. Location-assignment rule engine

Given a binary aneurysm mask and a vessel mask, assigns each detected
instance to one of 52 named locations — geometry-based (nearest vessel,
junction/arc-fraction heuristics), not a trained model.

Build the bootstrap-voted location prior (resamples the training cohort
with replacement, rebuilding the cohort-statistic prior multiple times so
the final rule majority-votes across replicates rather than trusting one
run):

```bash
python -m topaneu_rsna.seg.build_vessel_location_prior --out vessel_location_prior.json
```

Run the rule engine end to end (real, non-oracle predictions from steps
2-3 as input):

```bash
python -m topaneu_rsna.seg.assign_location_rule \
    --binary_dataset 304 --folds 0 1 2 3 4 \
    --vessel_source pred --vessel_pred_dir /path/to/model2/predictions \
    --prior vessel_location_prior.json
```

Pass `--vessel_source gt` to instead use ground-truth vessel masks (an
oracle upper-bound run, useful for isolating how much error the vessel
model itself contributes vs. the rule engine).

## Key results

| Component | Dataset | Metric | Result |
|---|---|---|---|
| Model 2 (vessel segmentation) | 302 | mean Dice, 36 classes (training-time validation) | 0.8153 |
| Model 2 on TopBrain TA36 | — | mean Dice, final-test leaderboard | 0.763 |
| Model 2 on TopBrain TA36 | — | clDice (topology) | 0.834 |
| Binary aneurysm segmentation | 304 | mean Dice, real 5-fold CV | 0.5568 |

Two vessel classes (3rd-A2, 3rd-A3 — an accessory anterior cerebral
segment) score near-zero on the base Model 2 checkpoint, traced to real
case scarcity in the training cohort (present in only ~15% and ~27% of
cases respectively), not a modeling defect — see `config.py`'s comments
near `DS_VESSEL` and the class-balanced retrain variant in
`nnUNet/nnunetv2/training/nnUNetTrainer/project_specific/rsna2025/vessel_class_balanced.py`
for the fix attempted.

## Label convention

TA36's canonical 36-class label map
([`CoWBenchmark/TopBrain_Eval_Metrics`](https://github.com/CoWBenchmark/TopBrain_Eval_Metrics)'s
`MUL_CLASS_LABEL_MAP`) was compared value-for-value against this project's
own `labels.json` vessel list and is an exact match — same 36 names, same
order (1=BA … 36=L-ICA-C1-C5), including the same four infraclinoid
classes TA36 added (R/L-ICA-C6-C7, R/L-ICA-C1-C5) at the same positions.
Model 2's raw output is therefore a valid TA36 submission with no
relabeling step.

## Citation

This project builds on nnU-Net:

> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H.
> (2021). nnU-Net: a self-configuring method for deep learning-based
> biomedical image segmentation. *Nature Methods*, 18(2), 203-211.

See `code/rsna2025_1st_place/nnUNet/LICENSE` for nnU-Net's own license.
