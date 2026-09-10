#!/bin/bash
# Regenerate every tuning pseudodata sample WITH the per-truth-particle survival
# labels (truth_survived / truth_in_tracker / truth_eff_region — see
# EFF_LOSS_PLAN.md), as *_truth_matched_survival.root next to the originals.
# Original samples are never touched. This file doubles as the reproduce.sh for
# the sample set (CLAUDE.md rule 5); a copy is placed in the output area.
#
# Run on a Perlmutter login node from anywhere:
#   bash src/parnassus/torch_delphes/slurm_scripts/submit_truth_matched_survival_samples.sh
#
# Generated (branch BCE_eff, commit recorded per-file in the .provenance.json
# sidecars): 2026-09-09 overnight, by Claude on the user's behalf.
set -euo pipefail

export REPO="/pscratch/sd/a/aelabd/parnassus"
export ENV_PREFIX="$REPO/parnassus_env"
SUB="$REPO/src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh"
CFG="$REPO/src/parnassus/torch_delphes/param_configs"
DD="/global/cfs/cdirs/m3246/diff_delphes"
TAG="truth_matched_survival"

# ---- The four sequential-closure samples (param_config_all truth card),
#      matching Runze's 2026-08-18 production: guns 2x100k, dijet 20x10k. ----
for p in muongun electrongun ksgun; do
    OUTBASE="$DD/allsamples" \
    MERGED_NAME="pseudo_data_200k_param_config_all_${p}_${TAG}.root" \
        bash "$SUB" --config "$CFG/param_config_all.yaml" --process "$p" \
        --n_events 200000 --n_tasks 2
done
OUTBASE="$DD/allsamples" \
MERGED_NAME="pseudo_data_200k_param_config_all_dijet_${TAG}.root" \
    bash "$SUB" --config "$CFG/param_config_all.yaml" --process dijet \
    --n_events 200000 --n_tasks 20

# ---- HZZ4l closure sample for compare_sample.py (100k, default 10 tasks). ----
OUTBASE="$DD/allsamples" \
MERGED_NAME="pseudo_data_100k_param_config_all_HZZ4l_${TAG}.root" \
    bash "$SUB" --config "$CFG/param_config_all.yaml" --process HZZ4l \
    --n_events 100000 --n_tasks 10

# ---- The single-stage samples (torch_delphes/CLAUDE.md section 3). ----
OUTBASE="$DD" \
MERGED_NAME="pseudo_data_200k_param_config_muons_muongun_${TAG}.root" \
    bash "$SUB" --config "$CFG/param_config_muons.yaml" --process muongun \
    --n_events 200000 --n_tasks 2
OUTBASE="$DD" \
MERGED_NAME="pseudo_data_200k_param_config_chads_ksgun_${TAG}.root" \
    bash "$SUB" --config "$CFG/param_config_chads.yaml" --process ksgun \
    --n_events 200000 --n_tasks 2
