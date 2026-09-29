"""
TopAneu port of the RSNA-2025 1st-place pipeline (uchiyama33/rsna2025_1st_place).

Every hyper-parameter here is taken from the author's released configs
(configs/experiment/251013-...-e25-w01_005_1-s128_256_256.yaml) unless marked
[TOPANEU], which flags a deliberate adaptation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _p(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default)).expanduser()


# --------------------------------------------------------------------------- paths
TOPANEU_ROOT = _p("TOPANEU_ROOT", "~/projects/def-punithak/abdul7/TopAneu")
EXP_ROOT     = _p("EXP_ROOT", str(TOPANEU_ROOT / "experiments/01_rsna_pipeline"))
CODE_ROOT    = _p("CODE_ROOT", str(EXP_ROOT / "code"))
DATA_ROOT    = _p("TOPANEU_DATA", str(TOPANEU_ROOT / "data"))
# TopBrain 2026 TA36 data release (imagesTr_topbrain + labelsTr_topbrain_v2_topaneu36class,
# same 36-class label convention Dataset302/Model 2 already uses) -- see setup_env.sh
TOPBRAIN_DATA_ROOT = _p("TOPBRAIN_DATA", str(TOPANEU_ROOT / "TopBrain_Data"))
LOG_ROOT     = _p("LOG_ROOT", str(EXP_ROOT / "logs"))
SCRATCH_ROOT = _p("SCRATCH_ROOT",
                  str(Path(os.environ.get("SCRATCH", "/tmp")) / "TopAneu/experiments/01_rsna_pipeline"))

nnUNet_raw          = _p("nnUNet_raw", str(SCRATCH_ROOT / "nnUNet_raw"))
nnUNet_preprocessed = _p("nnUNet_preprocessed", str(SCRATCH_ROOT / "nnUNet_preprocessed"))
nnUNet_results      = _p("nnUNet_results", str(SCRATCH_ROOT / "nnUNet_results"))

WORK            = SCRATCH_ROOT / "work"
COARSE_PRED_DIR = WORK / "coarse_pred"
COARSE_ROI_DIR  = WORK / "coarse_roi"
VESSEL_PRED_M2  = WORK / "vessel_pred_m2"
VESSEL_PRED_M3  = WORK / "vessel_pred_m3"
CLS_CACHE_DIR   = WORK / "cls_cache"
CLS_RESULTS_DIR = SCRATCH_ROOT / "cls_results"
VESSELNESS_DIR  = WORK / "vesselness"    # prep/compute_vesselness.py output, one <case>.nii.gz per case
# [TOPANEU] dedicated coarse-to-fine cascade for aneurysm segmentation itself (see
# DS_ANEURYSM_COARSE below) -- mirrors COARSE_PRED_DIR/COARSE_ROI_DIR/VESSEL_PRED_M2
# above, but driven by a coarse *aneurysm* localizer instead of Model 1's vessel-group
# one, so it gets its own working directories rather than overloading those.
ANEURYSM_COARSE_PRED_DIR = WORK / "aneurysm_coarse_pred"
ANEURYSM_COARSE_ROI_DIR  = WORK / "aneurysm_coarse_roi"
ANEURYSM_VESSEL_PRED_M2  = WORK / "aneurysm_vessel_pred_m2"
# [TOPANEU] vesselness maps already computed (3-day Frangi run) for the unrelated
# 01_rsna_pipeline/Dataset104_TopAneuLocationVesselness experiment, as
# imagesTr/<case>_0001.nii.gz next to the original imagesTr/<case>_0000.nii.gz --
# but on Dataset104's own native/un-resampled grid, not this pipeline's. Not used by
# default: build_aneurysm_vesselness_dataset.py computes vesselness fresh via
# prep/compute_vesselness.py (job 18) directly on this pipeline's own image grid,
# rather than trusting an unverified same-world-space assumption to resample this
# in. Kept only for the (unused-by-default) --legacy_vesselness_dir opt-in.
LEGACY_VESSELNESS_DIR = _p("LEGACY_VESSELNESS_DIR",
                           str(Path(os.environ.get("SCRATCH", "/tmp"))
                               / "TopAneu/experiments/01_rsna_pipeline/nnUNet_raw"
                                 "/Dataset104_TopAneuLocationVesselness/imagesTr"))

# ------------------------------------------------------- raw TopAneu layout (as shipped)
IMAGES_DIR        = DATA_ROOT / "images"          # <case>_0000.nii.gz
LOCATION_MASKS    = DATA_ROOT / "location_masks"  # <case>.nii.gz, aneurysm location classes
LOCATION_JSONS    = DATA_ROOT / "location_jsons"  # <case>.json
TYPE_MASKS        = DATA_ROOT / "type_masks"      # <case>.nii.gz, aneurysm type classes
VESSEL_MASKS      = DATA_ROOT / "vessel_masks"    # <case>.nii.gz, vessel-region classes
LOCATION_MAPPING  = DATA_ROOT / "location_mapping.json"
TYPE_MAPPING      = DATA_ROOT / "type_mapping.json"
VESSEL_MAPPING    = DATA_ROOT / "vessel_mapping.json"

IMAGE_SUFFIX = "_0000.nii.gz"
LABEL_SUFFIX = ".nii.gz"

# --------------------------------------------------------------------------- geometry
# array axis order is (z, y, x) everywhere in this codebase
COARSE_SPACING = (1.0, 1.0, 1.0)
COARSE_PATCH   = (128, 128, 128)
COARSE_ROI_MM  = (140.0, 140.0, 140.0)
FINE_SPACING   = (0.80, 0.45, 0.44)
FINAL_ROI_SIZE = (128, 256, 256)
# author's ROI dir was named "margin15_30": 15 mm in z, 30 mm in-plane
ROI_REFINE_MARGIN_MM = (15.0, 30.0, 30.0)

DBSCAN_EPS_MM      = 4.0
DBSCAN_MIN_SAMPLES = 20
DBSCAN_MAX_POINTS  = 30000
SW_OVERLAP_COARSE  = 0.2
SW_OVERLAP_FINE    = 0.3

# --------------------------------------------------------------------------- nnU-Net
DS_COARSE   = 301   # Dataset301_TopAneuVesselGroup  (Model 1)
DS_VESSEL   = 302   # Dataset302_TopAneuVessel       (Models 2 and 3)
DS_LOCATION = 303   # Dataset303_TopAneuLocation     (Task 2, 52-class -- paused, see memory)
DS_ANEURYSM = 304   # Dataset304_TopAneuAneurysm     (Task 2, binary, whole-head -- running)
# [TOPANEU] second Task 2 attempt, in parallel with DS_ANEURYSM/304 above (which was
# already running when this was added, so it gets its own dataset id rather than
# reusing/overwriting 304): same binary aneurysm/background target, but built by
# seg/build_aneurysm_roi_dataset.py instead of prep/build_nnunet_datasets.py --
# ROI-cropped to the same 128x256x256 box final_roi.py uses (Model 2's predicted
# vessel mask + margin) with Model 2's vessel mask as a second input channel, since
# aneurysms only occur on vessels. See jobs/15-17.
DS_ANEURYSM_ROI = 305   # Dataset305_TopAneuAneurysmROI (Task 2, binary, ROI + vessel channel)
# [TOPANEU] binary aneurysm/background again (same target as DS_ANEURYSM/304), but with a
# multi-scale Frangi vesselness response as a second input channel instead of a predicted
# vessel mask. Unlike DS_ANEURYSM_ROI/305 this is whole-head, not ROI-cropped, so it needs
# no Model 1/2 predictions (jobs 5/6). Channel 1 is computed fresh by
# prep/compute_vesselness.py (job 18, a hard prerequisite) directly on this pipeline's
# own image grid -- deliberately NOT reused/resampled from the separate, untracked
# 01_rsna_pipeline/Dataset104_TopAneuLocationVesselness experiment (see
# LEGACY_VESSELNESS_DIR below), since that would need trusting an unverified same-
# world-space assumption between two different image grids; recomputing (~3 days)
# was judged worth it for correctness. See jobs/18-21.
DS_ANEURYSM_VESSELNESS = 306   # Dataset306_TopAneuAneurysmVesselness (Task 2, binary, vesselness channel)
# [TOPANEU] revisiting the 52-class direct segmentation that collapsed as Dataset303,
# this time (a) conditioned on an existing vessel model's own prediction as a second
# input channel, since aneurysms only occur on vessels, and (b) initialized from that
# same model's trained weights (nnUNetv2_train -pretrained_weights) instead of random
# init -- the same transfer-learning lever that took the ROI classifier from 0.794 to
# 0.902 AUC when pretrained from Model 2. Two variants, built by
# seg/build_location_conditioned_dataset.py from job 05/06's *existing* outputs --
# neither needs new segmentation inference:
#   DS_LOCATION_M1COND -- whole-head, native grid. ch0=raw image, ch1=Model 1's
#     3-class vessel-group prediction (COARSE_PRED_DIR, already computed by job 05
#     for every case on this exact grid). Warm-start from Model 1's fold_all checkpoint.
#   DS_LOCATION_M2COND -- coarse-ROI-cropped grid (140mm cube, FINE_SPACING). ch0/ch1/
#     label are job 05/06's existing coarse-ROI image, Model 2's 36-class prediction,
#     and cropped location mask -- already grid-aligned. Warm-start from Model 2's
#     fold_all checkpoint.
# Channel-conditioning and warm-starting address different things than the earlier
# 303 postmortem's root cause (most patches contain at most one of 52 classes, too
# sparse for the loss to escape all-background) -- this doesn't fix that on its own,
# still worth pairing with class-balanced sampling if it doesn't fully resolve it.
DS_LOCATION_M1COND = 307   # Dataset307_TopAneuLocationM1Cond (Task 2, 52-class, +Model1 channel)
DS_LOCATION_M2COND = 308   # Dataset308_TopAneuLocationM2Cond (Task 2, 52-class, +Model2 channel, ROI)
# [TOPANEU] a dedicated coarse-to-fine cascade for aneurysm segmentation, mirroring the
# vessel Model1->Model2/3 pattern -- but built fresh for this task rather than reusing
# the vessel models' own localization (DS_ANEURYSM/304's and DS_ANEURYSM_ROI/305's
# ROI came from Model 1/2's vessel predictions, not a model that ever looked for
# aneurysms specifically at low resolution first).
#   DS_ANEURYSM_COARSE -- whole-head, ForcedLowres (1mm iso, 128^3 patch -- same
#     recipe as Model 1/DS_COARSE), binary aneurysm/background target. Its own
#     prediction (not ground truth) drives the crop for both fine variants below,
#     via seg/aneurysm_coarse_roi.py -- same self-consistent train/inference logic
#     as seg/coarse_roi.py. Whether that crop actually contains the true aneurysm is
#     checked separately and after the fact by seg/check_aneurysm_crop_coverage.py
#     (a CPU-only audit job) -- it doesn't feed back into the crop itself.
#   DS_ANEURYSM_FINE_RAW -- fine stage, cropped to that ROI, single channel (image
#     only). "Experiment A."
#   DS_ANEURYSM_FINE_VESSEL -- same crop, ch0=image, ch1=Model 2's vessel prediction
#     computed fresh on this new crop (ANEURYSM_VESSEL_PRED_M2 -- Model 2 was never
#     run whole-head or on this crop before, only on the older vessel-based one).
#     "Experiment B."
DS_ANEURYSM_COARSE      = 309   # Dataset309_TopAneuAneurysmCoarse
DS_ANEURYSM_FINE_RAW    = 310   # Dataset310_TopAneuAneurysmFineRaw
DS_ANEURYSM_FINE_VESSEL = 311   # Dataset311_TopAneuAneurysmFineVessel
# [TOPANEU] third variant of the 52-class location-conditioning experiment
# (see DS_LOCATION_M1COND/M2COND above), swapping the conditioning channel for
# the dataset's own ground-truth vessel_masks (36 classes) instead of a real
# model's prediction. This is an oracle/upper-bound run: no vessel inference
# and no pretrained-weight expansion/warm-start needed (the thing that made
# 307/308's own build+verify+train chain error-prone), so it isolates one
# question -- does a *perfect* vessel channel let the 52-class head escape the
# all-background collapse that random-init Dataset303 hit -- before spending
# effort on 307/308's noisier, model-predicted channel. Whole-head, native
# per-case grid, built by seg/build_location_conditioned_dataset.py
# --source gt_vessel. See jobs/14_segmentation_gtvessel_52class.
DS_LOCATION_GTVESSEL    = 312   # Dataset312_TopAneuLocationGTVessel
# [TOPANEU] Experiment A: Vessel-Aware Multi-Task Segmentation (Tapar, TopAneu
# 2026), adapted to this pipeline as a DECOUPLED two-model design rather than
# the paper's single joint dual-head network -- see jobs/29_expA_vessel_cond_seg's
# docstrings for the full rationale. This dataset is the SEGMENTATION half only
# (Task 2, 53-class = background + 52 locations), whole-head, native per-case
# grid, two input channels:
#   ch0 = raw image (IMAGES_DIR)
#   ch1 = Model 2's own 36-class vessel prediction, run whole-head for the
#         first time in this pipeline (every prior use of Model 2 was on a
#         coarse-ROI crop or was the oracle ground truth) -- see
#         seg/build_expA_vessel_cond_dataset.py and jobs/29's job 78.
# label = LOCATION_MASKS (52 classes + background), whole-head
# A fixed ~15% holdout (EXPA_HOLDOUT_JSON) is excluded from imagesTr/labelsTr
# entirely so this dataset's fold_all training still has one honest,
# never-trained-on set to evaluate on -- the paper's real center/modality-
# stratified 5-fold CV was judged too expensive for a 1-week budget (see the
# same jobs' docstrings), so fold_all + a manual holdout replaces it here,
# the same tradeoff already made for Models 1/2/3.
DS_VESSELCOND_SEG = 313   # Dataset313_TopAneuVesselCondSeg
# [TOPANEU] Experiment Delta: last-try binary aneurysm improvement, run on a
# SEPARATE server/checkout so it cannot affect the main submission pipeline.
# Dataset304's own dataset.json declares its single input channel as
# "CTA_MRA" (see build_nnunet_datasets.py) -- CTA and MRA are pooled into
# one nnU-Net z-score-normalized channel despite CTA being calibrated
# Hounsfield units and MRA being arbitrary, uncalibrated scanner intensity.
# Forcing one network to learn both intensity semantics at once is a
# plausible, previously-undiagnosed source of Dataset304's ~0.60 ceiling.
# This splits the SAME binary target into two modality-homogeneous datasets
# (same label rule as Dataset304: LOCATION_MASKS collapsed to binary), each
# trained as its own real 5-fold model, then pooled back into one combined
# Dice for a fair, same-cohort comparison against Dataset304's baseline. See
# prep/classify_modality.py (heuristic HU-based split -- no modality field
# exists anywhere in this dataset's own metadata, confirmed by grep) and
# jobs/34_modality_split_binary_aneurysm.
DS_ANEURYSM_CTA = 314   # Dataset314_TopAneuAneurysmCTA (binary, CTA-only cases)
DS_ANEURYSM_MRA = 315   # Dataset315_TopAneuAneurysmMRA (binary, MRA-only cases)
# [TOPANEU] TopBrain 2026 TA36 track: fine-tune Model 2 (Dataset302's own checkpoint,
# TRAINER_M2/PLANS_RESENC) on the organizers' own TopBrain_Data release -- same
# 36-class vessel target Dataset302 already predicts, just a second, independent
# cohort. Built by seg/build_topbrain_finetune_dataset.py from
# TOPBRAIN_DATA_ROOT/imagesTr_topbrain + labelsTr_topbrain_v2_topaneu36class (NOT the
# v1_ct/v1_mr label folders, which use different, per-modality, non-matching class
# counts). See jobs/41_topbrain_finetune.
DS_TOPBRAIN_FINETUNE = 316   # Dataset316_TopBrainVesselFinetune
# [TOPANEU] The one clean, controlled test of vessel-conditioning that was never
# actually run: identical to Dataset304 in every respect (same 417 cases, same
# whole-head native grid, same nnUNetResEncUNetMPlans plan reused via
# nnUNetv2_move_plans_between_datasets, same trainer, same real 5-fold split --
# copied byte-for-byte from Dataset304's own splits_final.json, not
# regenerated) except channel 1 = a binarized (vessel_pred > 0) real Model 2
# whole-head vessel prediction (VESSEL_PRED_M2_FULLHEAD, job 78's cache --
# same 0/1 "on some vessel" convention Dataset305 used, not Dataset313's raw
# 36-class channel). Every earlier attempt at this hypothesis (305: ROI-cropped
# AND vessel-conditioned at once; 306: vesselness filter instead of a real
# model; 311: crashed before epoch 1) changed something else alongside the
# channel, so none of them isolated the one variable this dataset isolates.
# See seg/build_aneurysm_vesselcond_dataset.py and jobs/43_vessel_cond_binary_controlled.
DS_ANEURYSM_VESSELCOND = 317   # Dataset317_TopAneuAneurysmVesselCond
# Same experiment, second variant: channel 1 comes from the class-balanced
# retrain (job 165, still training as of this writing) instead of the base
# Model 2 checkpoint. Kept as a SEPARATE dataset from 317, not a swap-in-place,
# so both vessel-channel sources stay comparable side by side rather than
# overwriting one another -- and because job 165's checkpoint is a moving
# target (checkpoint_best.pth changes every time a new EMA best is hit), this
# dataset's own build should be re-run if you want it to reflect a later
# checkpoint than whatever job 177 predicted with.
DS_ANEURYSM_VESSELCOND_CB = 318   # Dataset318_TopAneuAneurysmVesselCondCB
VESSEL_PRED_M2_CB_FULLHEAD = WORK / "vessel_pred_m2_cb_fullhead"
DS_NAMES = {DS_COARSE: "TopAneuVesselGroup", DS_VESSEL: "TopAneuVessel",
           DS_LOCATION: "TopAneuLocation", DS_ANEURYSM: "TopAneuAneurysm",
           DS_ANEURYSM_ROI: "TopAneuAneurysmROI",
           DS_ANEURYSM_VESSELNESS: "TopAneuAneurysmVesselness",
           DS_LOCATION_M1COND: "TopAneuLocationM1Cond",
           DS_LOCATION_M2COND: "TopAneuLocationM2Cond",
           DS_ANEURYSM_COARSE: "TopAneuAneurysmCoarse",
           DS_ANEURYSM_FINE_RAW: "TopAneuAneurysmFineRaw",
           DS_ANEURYSM_FINE_VESSEL: "TopAneuAneurysmFineVessel",
           DS_LOCATION_GTVESSEL: "TopAneuLocationGTVessel",
           DS_VESSELCOND_SEG: "TopAneuVesselCondSeg",
           DS_ANEURYSM_CTA: "TopAneuAneurysmCTA",
           DS_ANEURYSM_MRA: "TopAneuAneurysmMRA",
           DS_TOPBRAIN_FINETUNE: "TopBrainVesselFinetune",
           DS_ANEURYSM_VESSELCOND: "TopAneuAneurysmVesselCond",
           DS_ANEURYSM_VESSELCOND_CB: "TopAneuAneurysmVesselCondCB"}
# case -> "CTA" / "MRA" / "UNCERTAIN", written once by prep/classify_modality.py
MODALITY_JSON = CODE_ROOT / "topaneu_rsna" / "modality_split.json"
MODALITY_STATS_CSV = LOG_ROOT / "modality_classification_stats.csv"

PLANS_RESENC = "nnUNetResEncUNetMPlans"
# trainers bundled in the author's nnUNet fork
TRAINER_M1 = "RSNA2025Trainer_moreDAv7"
TRAINER_M2 = "RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07"   # backbone donor
TRAINER_M3 = "RSNA2025Trainer_moreDAv6_SkeletonRecallW3TverskyBeta07"
# [TOPANEU] Task 2 aneurysm segmentation: reuse M2's trainer -- aneurysms are small/
# rare positives like the vessel skeletons that trainer was tuned for, and its Tversky
# beta=0.7 already weights recall over precision, matching a task graded partly on
# false negatives (HD95 blows up on missed lesions).
#
# The first attempt trained this as a 52-class problem directly on location_masks
# (Dataset303/DS_LOCATION) and it collapsed to predicting all-background: with
# do_bg=False, most patches have ground truth for at most 1 of the 52 classes (most
# cases have a single aneurysm), so per-class signal was too sparse for the loss to
# ever push logits off the trivial "background everywhere" optimum. Dataset304/
# DS_ANEURYSM collapses all 52 location classes to one binary "aneurysm" foreground
# class instead, so every foreground-oversampled patch contributes real, consistent
# gradient for that one channel; the per-location class is assigned afterward by the
# existing ROI classifier (job 7), not by this segmentation model.
TRAINER_LOC = TRAINER_M2
SEG_FOLD = "all"

# [TOPANEU] Experiment A (see DS_VESSELCOND_SEG above): custom trainer
# implementing the paper's Sect 2.3 Stage-1 loss (Dice + CE + TopK(10%) +
# Focal(gamma=2, alpha=0.25)) on top of this pipeline's own proven anisotropic-
# patch augmentation trainer -- see
# code/rsna2025_1st_place/nnUNet/.../project_specific/rsna2025/expA_vessel_cond_seg.py
#
# RSNA2025Trainer_ExpA_VesselCondSeg's first run (jobs 3107515/3109070, 80
# epochs) collapsed to all-background (pseudo dice exactly 0.0/nan for all 52
# classes, every epoch) -- the same collapse Dataset303 hit, for the same
# reason: guaranteed-positive sampling alone doesn't fix per-class signal
# sparsity across the whole training run, only within one patch. Pointing
# this constant at the _ClassBalanced subclass instead, which adds
# class_balanced_focal.py's own second, previously-proven lever
# (case-level class-balanced sampling) on top -- see that class's own
# docstring. The original class is kept, not deleted, for the record.
TRAINER_EXPA_SEG = "RSNA2025Trainer_ExpA_VesselCondSeg_ClassBalanced"
# Job 78's whole-head (never-before-run) Model 2 vessel prediction, used as
# Dataset313's second input channel -- distinct from VESSEL_PRED_M2 above,
# which is only ever computed on the coarse-ROI crop.
VESSEL_PRED_M2_FULLHEAD = WORK / "vessel_pred_m2_fullhead"
# Deterministic ~15% case holdout, generated once by seg/build_expA_holdout.py
# and then reused unchanged by every later Experiment A job (dataset build,
# classifier training, consensus-fusion evaluation) -- same
# generated-artifact-under-code-root convention as PRIOR_PATH below.
EXPA_HOLDOUT_JSON = CODE_ROOT / "topaneu_rsna" / "expA_holdout_cases.json"
EXPA_HOLDOUT_FRAC = 0.15
# Cached per-case global-average-pooled encoder features (frozen Dataset313
# backbone), used to train Experiment A's small classification head without
# re-running the (expensive, whole-volume) encoder forward pass every epoch.
EXPA_FEATURE_CACHE = WORK / "expA_features"
# Job 83's TTA segmentation predictions on the holdout cases only (nnU-Net's
# own -o output dir), read back by seg/expA_consensus_fusion.py.
EXPA_SEG_PRED_HOLDOUT = WORK / "expA_seg_pred_holdout"

# [TOPANEU] Experiment B: standalone 3D crop classifier, completely
# independent of nnU-Net's training machinery and of Experiment A's
# (abandoned after its own collapse) segmentation attempt. Assigns each
# REAL, non-oracle predicted aneurysm instance (Dataset304's pooled 5-fold
# binary prediction, connected-component per instance) to one of 52
# locations + background, using a 3-channel crop (raw image, that same real
# binary channel, job 78's real whole-head Model 2 vessel channel) centered
# on the PREDICTED instance's own centroid -- not a ground-truth instance's
# centroid, which would create a train/inference distribution mismatch (see
# seg/build_expB_crop_dataset.py's own module docstring). Ground truth is
# used ONLY to look up each predicted instance's true label via overlap,
# exactly like assign_location_rule.py's own true_class field -- never as an
# input channel or as the basis for where/what gets cropped.
#
# Train/val split: reuses Dataset304's own real fold assignment (which
# fold's `validation/` directory a case's file appears under -- the same
# signal load_binary_pred_paths already relies on) rather than inventing a
# new split. EXPB_VAL_FOLD is held out entirely for evaluation; the other 4
# folds' cases are training data. A single fixed split, not a full 5-fold
# retrain, given this pipeline's time budget -- same class of tradeoff
# already made for Experiment A's fold_all + manual holdout.
EXPB_VAL_FOLD = 4
EXPB_CROP_CACHE = WORK / "expB_crops"                      # cached .npz crops, one per instance
EXPB_MANIFEST_CSV = LOG_ROOT / "task2_expB_crop_manifest.csv"
EXPB_CKPT_DIR = WORK / "expB_classifier"
EXPB_PREDICTIONS_CSV = LOG_ROOT / "task2_expB_classifier_predictions.csv"

# [TOPANEU] Experiment Alpha: INFERENCE-ONLY ablation on Dataset304's already-trained
# real 5-fold binary aneurysm model (job 13) -- (1) cross-fold ensembling, (2)
# test-time mirroring augmentation on/off, (3) connected-component volume-threshold
# postprocessing. Dataset304's training is never touched. See
# jobs/33_expAlpha_binary_postproc.
#
# IMPORTANT caveat (see that job's README): Dataset304 was trained with a real 5-fold
# CV over the FULL case pool, with no case held out of all 5 folds. That means every
# case is training data for 4 of the 5 folds and validation data for exactly 1. A
# fold-ensembled prediction on any of these cases is therefore NOT an honest unseen-
# data estimate for 4/5 of its own ensemble members -- there is no subset of cases in
# this pipeline that was never used to train ANY of Dataset304's 5 folds. The
# single-fold (own validation fold only) TTA-on/off comparison and the postprocessing
# sweep are NOT affected by this (each case is still scored only by the one fold that
# never trained on it), only the cross-fold ENSEMBLE number is; that step's own script
# labels its result accordingly rather than silently reporting a leaked/inflated Dice.
EXPALPHA_PRED_DIR = WORK / "expAlpha_binary_pred"                 # notta/, ensemble5/, postproc_best/
EXPALPHA_RESULTS_CSV = LOG_ROOT / "task2_expAlpha_binary_results.csv"


def seg_model_dir(ds_id: int, trainer: str) -> Path:
    return (nnUNet_results / f"Dataset{ds_id:03d}_{DS_NAMES[ds_id]}"
            / f"{trainer}__{PLANS_RESENC}__3d_fullres")


# --------------------------------------------------------------------------- labels
LABELS_JSON = CODE_ROOT / "topaneu_rsna" / "labels.json"


@dataclass
class LabelSpec:
    locations: list = field(default_factory=list)          # 52 aneurysm-location names
    vessels: list = field(default_factory=list)            # V vessel-region names
    types: list = field(default_factory=list)              # aneurysm type names
    location_to_vessel: dict = field(default_factory=dict)
    coarse_groups: dict = field(default_factory=dict)      # vessel name -> group 1..3

    @property
    def n_loc(self): return len(self.locations)
    @property
    def n_vessel(self): return len(self.vessels)
    @property
    def n_type(self): return len(self.types)
    @property
    def n_coarse_group(self): return len(set(self.coarse_groups.values()))

    def vessel_label_for_location(self) -> list:
        """Row i = vessel label value (1..V) whose mask feeds location token i."""
        v2i = {v: i + 1 for i, v in enumerate(self.vessels)}
        out = []
        for loc in self.locations:
            v = self.location_to_vessel.get(loc)
            if v is None or v not in v2i:
                raise KeyError(f"location '{loc}' has no valid vessel in labels.json")
            out.append(v2i[v])
        return out


def load_labels(path=None) -> LabelSpec:
    path = Path(path or LABELS_JSON)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python -m topaneu_rsna.prep.probe_dataset")
    d = json.loads(path.read_text())
    return LabelSpec(**{k: d[k] for k in
                        ("locations", "vessels", "types",
                         "location_to_vessel", "coarse_groups")})


# --------------------------------------------------------------- classifier config
@dataclass
class ClsConfig:
    # --- backbone: pretrained nnU-Net, decoder truncated by one stage
    num_truncate_stages: int = 1
    sphere_mid_channels: int = 32
    pretrained: bool = True              # ablation: 0.794 -> 0.902 with pretraining
    # --- pooling (author: m32g64)
    proj_mask_dim: int = 32
    proj_global_dim: int = 64
    branch_norm: bool = True
    use_encoder_global_feat: bool = True
    num_extra_mask_branches: int = 1     # Model 3 masks as a second branch
    # --- transformer
    transformer_embed_dim: int = 96
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.1
    # --- aux sphere task
    sphere_radius: int = 5
    ft_alpha: float = 0.3
    ft_beta: float = 0.7
    ft_gamma_pp: float = 2.0
    ft_gamma_focal: float = 1.33
    # --- loss weights (ablation: all-1.0 costs ~1.8 AUC points)
    w_loc: float = 0.1
    w_ap: float = 0.05
    w_sphere: float = 1.0
    w_type: float = 0.05                 # [TOPANEU] extra head, TopAneu ships type_masks
    # w_loc=0.1 was tuned to protect macro-AUC under plain (unweighted)
    # BCEWithLogitsLoss on the loc head -- job 88/89's official, presence-
    # based evaluation (evaluate_official_classification.py) showed that
    # setup gives recall 0.035 at the standard 0.5 threshold despite AUC
    # ~0.85-0.94: with 52 independent, heavily-imbalanced location labels,
    # unweighted BCE at low weight lets the model rank correctly while never
    # being pushed to place its decision boundary near 0.5 for the rare
    # positive class. "balanced_bce" reuses the SAME BalancedBCEWithLogitsLoss
    # already used for the sphere head (mean loss on positives + mean loss on
    # negatives, weighted equally regardless of class rarity) for the loc head
    # too -- default "bce" keeps existing runs' behavior identical.
    loc_loss: str = "bce"                # "bce" (original) or "balanced_bce"
    use_type_head: bool = True           # [TOPANEU]
    # --- optimisation
    epochs: int = 25
    batch_size: int = 1
    accumulate_grad_batches: int = 8     # effective batch 8
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 4
    warmup_start_lr: float = 1e-5
    eta_min: float = 1e-5
    amp: bool = True
    num_workers: int = 4
    ema: bool = True
    ema_decay: float = 0.995
    ema_update_after_step: int = 100
    # --- augmentation
    distort_limit: float = 0.1
    p_flip: float = 0.5
    p_affine: float = 0.6
    p_distort: float = 0.3
    p_lowres: float = 0.3
    rotate_deg: float = 10.0
    scale_range: float = 0.10
    shear_range: float = 0.10
    # --- cv
    n_folds: int = 5
    n_folds_ensemble: int = 4            # author averaged 4 of 5
    seed: int = 42


CLS = ClsConfig()
