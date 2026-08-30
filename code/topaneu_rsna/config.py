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
DS_LOCATION = 303   # Dataset303_TopAneuLocation     (Task 2: aneurysm location segmentation)
DS_NAMES = {DS_COARSE: "TopAneuVesselGroup", DS_VESSEL: "TopAneuVessel",
           DS_LOCATION: "TopAneuLocation"}

PLANS_RESENC = "nnUNetResEncUNetMPlans"
# trainers bundled in the author's nnUNet fork
TRAINER_M1 = "RSNA2025Trainer_moreDAv7"
TRAINER_M2 = "RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07"   # backbone donor
TRAINER_M3 = "RSNA2025Trainer_moreDAv6_SkeletonRecallW3TverskyBeta07"
# [TOPANEU] Task 2 (aneurysm location segmentation): reuse M2's trainer -- aneurysms
# are small/rare positives like the vessel skeletons that trainer was tuned for, and
# its Tversky beta=0.7 already weights recall over precision, which matches a task
# graded partly on false negatives (HD95 blows up on missed lesions).
TRAINER_LOC = TRAINER_M2
SEG_FOLD = "all"


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
