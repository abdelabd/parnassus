#!/bin/bash
# Optuna hyperparameter search for tune_cms_fullsim.
# ONE Optuna study; each trial's fit runs data-parallel across N GPUs (DDP),
# launched with torchrun (forks N ranks on this node -- no SLURM job step).
# Run on a node that has N GPUs (e.g. an interactive salloc shell), then:
#   ./src/parnassus/torch_delphes/run_optuna.sh        # 4 GPUs (default)
#   ./src/parnassus/torch_delphes/run_optuna.sh 2      # 2 GPUs
#
# Edit the CONFIG block below. Two presets are provided -- switch by commenting.
set -euo pipefail

N_GPUS="${1:-4}"
PYTHON=/global/cfs/cdirs/m3246/Runze/MCGen/envs/parnassus/bin/python

# # ===== preset: pseudodata (active) =====
# ROOT_FILE=/global/cfs/cdirs/m3246/diff_delphes/cms_pseudodata_100k.root
# N_EVENTS=-1
# OUTPUT_BASE=doc/pseudodata_results
# STUDY_NAME=pseudo_100k

# ===== preset: full CMS sim (uncomment this, comment the block above) =====
ROOT_FILE=/global/cfs/cdirs/m3246/diff_delphes/train_1000.root
N_EVENTS=100000
OUTPUT_BASE=doc/figure_fullsim_results
STUDY_NAME=fullsim_100k

# ===== shared knobs =====
N_STEPS=100
N_TRIALS=50          # trials ADDED per run (re-run to add 50 more; resumes the study below)
LOSS=wasserstein_1d  # can be "soft_hist"
PID_WEIGHTING=sqrt_fraction
OPTUNA_CONFIG=src/parnassus/torch_delphes/param_configs/optuna_config.yaml
# Warm start: trial 0 of a FRESH study starts from these known-good defaults
# (values outside the optuna_config search ranges are clamped; lr / batch_size /
# lr_scale are still sampled). Ignored when the study below is resumed.
INIT_CONFIG=src/parnassus/torch_delphes/param_configs/cms_target_default.yaml
HISTORY_PATH="$OUTPUT_BASE/all_optuna.json"
# Persistent study: re-running this script RESUMES it and adds N_TRIALS more trials
# (TPE keeps the earlier results; round_<n>/ dirs continue). To start over, delete
# this file or change STUDY_NAME above.
STORAGE="sqlite:///$OUTPUT_BASE/study.db"

# ===== launch =====
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # repo root (the paths above are relative to it)
export PYTHONUNBUFFERED=1       # stream the live log
export OMP_NUM_THREADS=8        # CPU threads per rank (lower if the N ranks contend for cores)

# N ranks on this node (rank 0 owns the study); each rank uses cuda:LOCAL_RANK.
"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$N_GPUS" \
    -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
    --root-file "$ROOT_FILE" \
    --optuna-config "$OPTUNA_CONFIG" \
    --init-config "$INIT_CONFIG" \
    --n-events "$N_EVENTS" \
    --n-steps "$N_STEPS" \
    --n-trials "$N_TRIALS" \
    --loss "$LOSS" \
    --pid-weighting "$PID_WEIGHTING" \
    --output-base "$OUTPUT_BASE" \
    --history-path "$HISTORY_PATH" \
    --storage "$STORAGE" \
    --study-name "$STUDY_NAME"
