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
# Use the ACTIVATED environment by default (override with PYTHON=... ./run_optuna.sh).
# A hardcoded path here silently ignores `source parnassus_env/bin/activate` and dies
# with "No module named 'parnassus'" if that env lacks an editable install.
PYTHON="${PYTHON:-$(command -v python)}"

# # ===== preset: pseudodata (active) =====
# ROOT_FILE=/global/cfs/cdirs/m3246/diff_delphes/cms_pseudodata_100k.root
# N_EVENTS=-1
# OUTPUT_BASE=doc/pseudodata_results
# STUDY_NAME=pseudo_100k

# ===== preset: full CMS sim (uncomment this, comment the block above) =====
ROOT_FILE=${ROOT_FILE:-/global/cfs/cdirs/m3246/diff_delphes/cms_pseudodata_100k.root}
N_EVENTS=${N_EVENTS:-100_000}
OUTPUT_BASE=${OUTPUT_BASE:-doc/figures/optuna_adam_w1d_pseudo_100k}
STUDY_NAME=${STUDY_NAME:-fullsim_100k}

# ===== shared knobs =====
N_STEPS=${N_STEPS:-100}
N_TRIALS=${N_TRIALS:-50}          # trials ADDED per run (re-run to add 50 more; resumes the study below)
LOSS=wasserstein_1d  # can be "soft_hist"
PID_WEIGHTING=sqrt_fraction
OPTUNA_CONFIG=src/parnassus/torch_delphes/param_configs/optuna_config.yaml
# Comet: one experiment PER TRIAL, named "<COMET_NAME>_<trial number>"
# (optuna_adam_0, optuna_adam_1, ...). Needs COMET_API_KEY exported (workspace
# from COMET_WORKSPACE); without it the search still runs, just unlogged.
COMET_NAME="${COMET_NAME:-optuna_adam}"
HISTORY_PATH="$OUTPUT_BASE/all_optuna.json"
# Persistent study: re-running this script RESUMES it and adds N_TRIALS more trials
# (TPE keeps the earlier results; round_<n>/ dirs continue). To start over, delete
# this file or change STUDY_NAME above.
STORAGE="sqlite:///$OUTPUT_BASE/study.db"

# ===== train/val split =====
# The split is CONTIGUOUS in file order, so by default train/val/test are blocks
# of the ROOT file. Set SHUFFLE_SPLIT=1 to permute the event order first, which
# matters whenever the file is ordered -- e.g. a merged sample (the generation
# array's per-seed parts are concatenated) or a multi-gun merge, where the val
# block would otherwise cover a different phase space than the train block and
# the trial scores stop being comparable. Off by default so an existing study
# resumed by this script keeps the split its earlier trials were scored on.
SHUFFLE_SPLIT=${SHUFFLE_SPLIT:-0}
SHUFFLE_SPLIT_SEED=${SHUFFLE_SPLIT_SEED:-0}   # separate from the fit seed; fixes the split

SPLIT_ARGS=()
if [[ "$SHUFFLE_SPLIT" != "0" ]]; then
    SPLIT_ARGS+=(--shuffle-split --shuffle-split-seed "$SHUFFLE_SPLIT_SEED")
fi

# ===== launch =====
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # repo root (the paths above are relative to it)
export PYTHONUNBUFFERED=1       # stream the live log
export OMP_NUM_THREADS=8        # CPU threads per rank (lower if the N ranks contend for cores)

# N ranks on this node (rank 0 owns the study); each rank uses cuda:LOCAL_RANK.
# "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$N_GPUS" \
srun "$PYTHON" \
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
    --comet-name "$COMET_NAME" \
    --storage "$STORAGE" \
    --study-name "$STUDY_NAME" \
    ${SPLIT_ARGS[@]+"${SPLIT_ARGS[@]}"}
