#!/bin/bash
#SBATCH -A m3246
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -J hungmatch
#
# Build the *_hungarian_matched_survival.root sample set from the
# *_truth_matched_survival.root set (EFF_LOSS_PLAN.md steps 5-6): replace
# truth_survived by per-event Hungarian truth<->reco matching labels, keep the
# generator labels as truth_survived_generator, and write the per-species
# confusion matrix sidecars (the matcher label-noise measurement).
#
# Submit after the truth_matched_survival merges:
#   sbatch --dependency=afterok:<merge jids> hungarian_match_samples.sbatch
set -euo pipefail

REPO="${REPO:-/pscratch/sd/a/aelabd/parnassus}"
DIR="${DIR:-/global/cfs/cdirs/m3246/diff_delphes/allsamples}"
# shellcheck disable=SC1091
source "$REPO/parnassus_env/bin/activate"
cd "$REPO"

for p in muongun electrongun ksgun dijet; do
    in="$DIR/pseudo_data_200k_param_config_all_${p}_truth_matched_survival.root"
    out="$DIR/pseudo_data_200k_param_config_all_${p}_hungarian_matched_survival.root"
    echo "[hungmatch] $p"
    python -m parnassus.torch_delphes.hungarian_survival_matching \
        --input "$in" --output "$out"
done
echo "[hungmatch] DONE"
