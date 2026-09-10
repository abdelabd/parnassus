#!/bin/bash
#SBATCH -A m3246_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 03:00:00
#SBATCH -J seq_survival
#
# Sequential closure on the survival-labeled samples (EFF_LOSS_PLAN.md steps 4/7).
# The BCE efficiency loss is on automatically: run_sequential.sh passes
# --mode delphes and --eff-loss defaults to bce there.
#
#   TAG=truth_matched_survival     -> doc/figure_sequential_truth_matched_survival
#   TAG=hungarian_matched_survival -> doc/figure_sequential_hungarian_matched_survival
#
# Submit:   TAG=truth_matched_survival sbatch --export=ALL run_sequential_survival.sbatch
# Or, on an interactive allocation (salloc --no-shell ... -> JID):
#   TAG=... srun --jobid=$JID -N1 -n1 --gpus-per-node=4 -c 128 \
#       bash src/parnassus/torch_delphes/slurm_scripts/run_sequential_survival.sbatch
set -euo pipefail

REPO="${REPO:-/pscratch/sd/a/aelabd/parnassus}"
TAG="${TAG:?set TAG=truth_matched_survival or TAG=hungarian_matched_survival}"

export SAMPLE_DIR="${SAMPLE_DIR:-/global/cfs/cdirs/m3246/diff_delphes/allsamples}"
export SAMPLE_PATTERN="pseudo_data_200k_param_config_all_%s_${TAG}.root"
export OUT_BASE="$REPO/doc/figure_sequential_${TAG}"
case "$TAG" in
    truth_matched_survival)     export COMET_NAME_PREFIX="tms" ;;
    hungarian_matched_survival) export COMET_NAME_PREFIX="hms" ;;
    *)                          export COMET_NAME_PREFIX="$TAG" ;;
esac
export COMET_API_KEY="${COMET_API_KEY:-dP7SQEk285l0DsZvRZgPM4cR7}"
export COMET_WORKSPACE="${COMET_WORKSPACE:-abdelabd}"

mkdir -p "$OUT_BASE"
# Reproducibility (torch_delphes/CLAUDE.md rule 5).
{
    echo "# Reproduce this run:"
    echo "cd $REPO && git checkout $(git -C "$REPO" rev-parse HEAD)"
    echo "TAG=$TAG SAMPLE_DIR=$SAMPLE_DIR sbatch --export=ALL $REPO/src/parnassus/torch_delphes/slurm_scripts/run_sequential_survival.sbatch"
    echo "# SAMPLE_PATTERN=$SAMPLE_PATTERN  OUT_BASE=$OUT_BASE  COMET_NAME_PREFIX=$COMET_NAME_PREFIX"
} > "$OUT_BASE/reproduce.sh"

cd "$REPO"
# Optional pipelining: STAGES = space-separated stage YAMLs (default: all four);
# FROM_HISTORY = a history.json to start the first listed stage from (see
# run_sequential.sh). Lets stages 1-2 run before the dijet sample exists, with
# stages 3-4 chained behind the dijet merge via FROM_HISTORY=stage2's history.
# shellcheck disable=SC2086
FROM_HISTORY="${FROM_HISTORY:-}" \
    bash src/parnassus/torch_delphes/full_phasespace_tuning/run_sequential.sh ${STAGES:-}
