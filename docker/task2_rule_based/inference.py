"""
TopAneu-26 Task 2 submission: real (non-oracle) rule-based location-assignment
pipeline. Implements the two functions templates/task2/main.py (from
https://github.com/Bangulli/TopAneu-26) imports and calls -- infer_ct for the
"head-ct-angiography" interface, infer_mr for "head-mr-angiography". Both
route to the same pipeline: Dataset304/302 were trained on a merged CTA+MRA
cohort, so there is currently only one model per stage, not per modality.

Pipeline per case:
  1. Binary aneurysm segmentation -- Dataset304, 5-fold ensemble (softmax
     averaged across folds 0-4, TTA on) -> binary foreground/background mask.
  2. Vessel segmentation -- Dataset302 "Model 2", fold_all (this model was
     never trained with a held-out split -- see this pipeline's own
     documented limitation) -> 36-class vessel mask. This is a REAL
     prediction, not the ground-truth oracle every evaluation number in this
     repo up to this point was computed against (assign_location_rule.py's
     --vessel_source pred option exists for exactly this reason but was
     never the default in any prior job).
  3. Connected-component label the binary mask -> one candidate per instance.
  4. The declared rule engine (assign_location_rule.py's own assign_case(),
     imported directly rather than re-implemented, so this can never
     silently drift from whatever the evaluated/tuned rule actually does)
     assigns each instance one of 52 locations using the REAL vessel mask
     above + the bundled, pre-computed vessel_location_prior.json.
  5. Paint the final uint8 mask: 0 = background, 1-52 = location classes,
     matching this repo's own loc_value convention (labels.json's
     "locations" list, 1-indexed) and the challenge's own 0-52 output
     labeling.

Model weights are expected at $MODEL_ROOT (default /opt/ml/model, matching
grand-challenge's own convention for a separately-uploaded model resource --
see this directory's README for the exact expected layout and how to test
this locally before uploading anything).

NOT YET VALIDATED end-to-end -- see the README's "before you trust this"
checklist. In particular the rule engine has never been run against a real
(non-oracle) vessel prediction before; this container is also the first
time that gets tested.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from topaneu_rsna import config as C
from topaneu_rsna.seg.assign_location_rule import (
    PRIOR_PATH, assign_case, build_vessel_to_locations)

MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/opt/ml/model"))
DATASET304_DIR = MODEL_ROOT / "dataset304_binary_aneurysm"
DATASET302_DIR = MODEL_ROOT / "dataset302_vessel"

# Rule parameters: identical to jobs 62/65's own values (Experiment Gamma's
# current best), NOT re-tuned for real-vessel input -- see the README for
# why that's a real, open question worth checking, not assumed fine.
TAU_MM = 4.0
JUNCTION_TAU_MM = 2.0
JUNCTION_OVERRIDE_MM = 1.5
ARC_AMBIGUOUS_MARGIN = 0.03
MIN_VOXELS = 3

_state = {}  # lazy-initialized: binary_predictor, vessel_predictor, spec, vessel_to_locations, loc_value, name_to_id, prior


def _lazy_init():
    """Load both nnU-Net models + the rule engine's small JSON tables once,
    on first call rather than at import time -- keeps a bad model path or
    missing file from crashing before main.py's own error reporting kicks in,
    and avoids paying model-load cost twice if a container instance somehow
    imports this module more than once."""
    if _state:
        return

    if not DATASET304_DIR.exists():
        raise FileNotFoundError(
            f"{DATASET304_DIR} missing -- see this directory's README for the "
            f"expected model bundle layout under $MODEL_ROOT.")
    if not DATASET302_DIR.exists():
        raise FileNotFoundError(
            f"{DATASET302_DIR} missing -- see this directory's README for the "
            f"expected model bundle layout under $MODEL_ROOT.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    binary_predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    binary_predictor.initialize_from_trained_model_folder(
        str(DATASET304_DIR), use_folds=(0, 1, 2, 3, 4), checkpoint_name="checkpoint_final.pth")

    vessel_predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    vessel_predictor.initialize_from_trained_model_folder(
        str(DATASET302_DIR), use_folds=("all",), checkpoint_name="checkpoint_final.pth")

    spec = C.load_labels()
    prior = json.loads(PRIOR_PATH.read_text())

    _state.update(
        binary_predictor=binary_predictor,
        vessel_predictor=vessel_predictor,
        spec=spec,
        vessel_to_locations=build_vessel_to_locations(spec),
        loc_value={loc: i + 1 for i, loc in enumerate(spec.locations)},
        name_to_id={v: i + 1 for i, v in enumerate(spec.vessels)},
        prior=prior,
    )


def _sitk_to_nnunet_input(img: sitk.Image):
    """(1, Z, Y, X) float array + the {'spacing': (z,y,x)} properties dict
    predict_single_npy_array requires -- same (z,y,x) axis convention this
    whole pipeline's utils/io.py already uses everywhere else (array axis
    order (z,y,x), spacing reversed from SimpleITK's native (x,y,z))."""
    arr = sitk.GetArrayFromImage(img).astype(np.float32)[None]
    spacing_zyx = tuple(img.GetSpacing()[::-1])
    return arr, {"spacing": spacing_zyx}


def _run_pipeline(img: sitk.Image) -> sitk.Image:
    _lazy_init()
    s = _state

    arr, props = _sitk_to_nnunet_input(img)
    spacing_zyx = props["spacing"]

    binmask = s["binary_predictor"].predict_single_npy_array(arr, props) > 0
    vessel_map = s["vessel_predictor"].predict_single_npy_array(arr, props)

    final, _instances = assign_case(
        binmask, vessel_map, spacing_zyx,
        s["spec"].vessels, s["vessel_to_locations"], s["prior"],
        s["name_to_id"], s["loc_value"],
        tau_mm=TAU_MM, junction_tau_mm=JUNCTION_TAU_MM,
        junction_override_mm=JUNCTION_OVERRIDE_MM,
        arc_ambiguous_margin=ARC_AMBIGUOUS_MARGIN, min_voxels=MIN_VOXELS,
        gt=None)

    out = sitk.GetImageFromArray(final.astype(np.uint8))
    out.CopyInformation(img)   # spacing/origin/direction must match the input exactly
    return out


def infer_ct(img: sitk.Image) -> sitk.Image:
    return _run_pipeline(img)


def infer_mr(img: sitk.Image) -> sitk.Image:
    return _run_pipeline(img)
