# Based verbatim on https://github.com/Bangulli/TopAneu-26's own
# templates/task2/Dockerfile (fetched directly, not reproduced from memory) --
# everything above the "--- added for this pipeline ---" marker is their
# original template, unchanged, so this stays compatible with however
# grand-challenge actually runs it.

FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime AS example_algorithm_amd64
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# --- added for this pipeline ---
#
# The trained Dataset304/302 checkpoints were saved by custom trainer classes
# (e.g. RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07) that only
# exist in this pipeline's own vendored nnU-Net fork, not in a plain PyPI
# nnunetv2 install -- nnUNetPredictor looks the trainer class up by name from
# the checkpoint at load time, so it must be importable here. Bundled via
# PYTHONPATH exactly like this pipeline's own setup_env.sh already does on
# the cluster, rather than `pip install`ed, for the same reason (its
# console-script entry points aren't needed here, only the plain package
# import nnUNetPredictor itself uses).
COPY --chown=user:user nnUNet_src /opt/app/nnUNet_src
ENV PYTHONPATH="/opt/app/nnUNet_src:${PYTHONPATH}"

# The rule engine itself (assign_location_rule.py's assign_case()/
# build_vessel_to_locations(), imported directly rather than re-implemented,
# so this can never silently drift from the actual tuned/evaluated rule) plus
# its small config/label/prior JSON tables.
COPY --chown=user:user topaneu_rsna /opt/app/code/topaneu_rsna
ENV CODE_ROOT="/opt/app/code"
ENV PYTHONPATH="/opt/app/code:${PYTHONPATH}"

# Model weights are NOT copied into the image -- grand-challenge uploads
# these as a separate "model" resource and mounts it at /opt/ml/model at
# runtime (see this directory's README for the exact expected layout, and
# how to fake this mount for local testing via do_test_run.sh).
ENV MODEL_ROOT="/opt/ml/model"

# --- end additions ---

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user main.py /opt/app/

ENTRYPOINT ["python", "main.py"]
