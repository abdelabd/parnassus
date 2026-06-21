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

# ===== preset: pseudodata (active) =====
ROOT_FILE=/global/cfs/cdirs/m3246/Runze/MCGen/data/cms_pseudodata_100k.root
OUTPUT_BASE=doc/pseudodata_results
STUDY_NAME=pseudo_100k

# ===== preset: full CMS sim (uncomment this, comment the block above) =====
# ROOT_FILE=/global/cfs/cdirs/m3246/Runze/MCGen/data/train_1000.root
# OUTPUT_BASE=doc/fullsim_results
# STUDY_NAME=fullsim_100k

# ===== shared knobs =====
N_EVENTS=-1
N_STEPS=100
N_TRIALS=50
LOSS=wasserstein_1d
PID_WEIGHTING=sqrt_fraction
OPTUNA_CONFIG=src/parnassus/torch_delphes/param_configs/optuna_config.yaml
HISTORY_PATH="$OUTPUT_BASE/all_optuna.json"

# ===== launch =====
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # repo root (the paths above are relative to it)
export PYTHONUNBUFFERED=1       # stream the live log
export OMP_NUM_THREADS=8        # CPU threads per rank (lower if the N ranks contend for cores)

# N ranks on this node (rank 0 owns the study); each rank uses cuda:LOCAL_RANK.
"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$N_GPUS" \
    -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
    --root-file "$ROOT_FILE" \
    --optuna-config "$OPTUNA_CONFIG" \
    --n-events "$N_EVENTS" \
    --n-steps "$N_STEPS" \
    --n-trials "$N_TRIALS" \
    --loss "$LOSS" \
    --pid-weighting "$PID_WEIGHTING" \
    --output-base "$OUTPUT_BASE" \
    --history-path "$HISTORY_PATH" \
    --study-name "$STUDY_NAME"
