#!/bin/bash

# ============================================================
# TopAneu - RSNA-SELF Pipeline Environment
# ============================================================

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

export TOPANEU_ROOT="$HOME/projects/def-punithak/abdul7/TopAneu"

export EXP_ROOT="$TOPANEU_ROOT/experiments/02_rsna_self_pipeline"
export CODE_ROOT="$EXP_ROOT/code"

# Master dataset - DO NOT MODIFY
export TOPANEU_DATA="$TOPANEU_ROOT/data"

# TopBrain 2026 TA36 data release (imagesTr_topbrain + labelsTr_topbrain_v2_topaneu36class,
# same 36-class label convention Model 2/Dataset302 already uses) - DO NOT MODIFY
export TOPBRAIN_DATA="$TOPANEU_ROOT/TopBrain_Data"

# ---------- Persistent logs ----------
export LOG_ROOT="$EXP_ROOT/logs"


# ------------------------------------------------------------
# Scratch
# ------------------------------------------------------------

export SCRATCH_ROOT="$SCRATCH/TopAneu/experiments/02_rsna_self_pipeline"

# Large nnUNet working directories live on scratch
export nnUNet_raw="$SCRATCH_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$SCRATCH_ROOT/nnUNet_preprocessed"
export nnUNet_results="$SCRATCH_ROOT/nnUNet_results"

# ------------------------------------------------------------
# Python
# ------------------------------------------------------------

export PYTHONPATH="$CODE_ROOT/rsna2025_1st_place/nnUNet:$PYTHONPATH"
# ------------------------------------------------------------
# Virtual environment
# ------------------------------------------------------------

source "$TOPANEU_ROOT/venv/bin/activate"

# ------------------------------------------------------------
# Create directories
# ------------------------------------------------------------

mkdir -p "$SCRATCH_ROOT"
mkdir -p "$nnUNet_raw"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

mkdir -p "$EXP_ROOT/logs"

# ------------------------------------------------------------
# Information
# ------------------------------------------------------------

echo "=============================================="
echo "TopAneu RSNA Pipeline Environment"
echo "=============================================="

echo "TOPANEU_ROOT        = $TOPANEU_ROOT"
echo "EXP_ROOT            = $EXP_ROOT"
echo "CODE_ROOT           = $CODE_ROOT"
echo "TOPANEU_DATA        = $TOPANEU_DATA"
echo "TOPBRAIN_DATA        = $TOPBRAIN_DATA"

echo ""
echo "SCRATCH_ROOT        = $SCRATCH_ROOT"
echo "nnUNet_raw          = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results      = $nnUNet_results"
echo "LOG_ROOT            = $LOG_ROOT"

echo ""
echo "Python:"
which python

echo ""
echo "nnUNet:"
which nnUNetv2_train 2>/dev/null || echo "not installed"

echo ""
echo "Scratch:"
df -h "$SCRATCH_ROOT"

echo "=============================================="
