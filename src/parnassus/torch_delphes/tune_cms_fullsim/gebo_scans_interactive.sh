#!/bin/bash
# Launch all 12 gebo_optuna_search.py scans (2 losses x 3 trust regions x 2
# grad modes) INTERACTIVELY, one per GPU, packed into an already-active
# salloc allocation of 3 nodes x 4 GPUs (12 GPUs total).
#
# First get the allocation (NOT done by this script):
#
#   salloc -C gpu -q interactive -N 3 --gpus-per-node=4 -t 04:00:00 -A m3246_g
#
# then, from inside that allocation's shell:
#
#   cd /pscratch/sd/a/aelabd/parnassus
#   ./src/parnassus/torch_delphes/tune_cms_fullsim/gebo_scans_interactive.sh
#
# Re-running this script (in a fresh salloc, after the previous one timed
# out) resumes every scan: each scan's Optuna study lives in a persistent
# sqlite DB under its own doc/gebo_optuna_scans/<scan_name>/, and each
# trial's GEBO/L-BFGS progress is checkpointed on disk (see
# gebo_optuna_search.py's module docstring for the two-level resume design).
#
# Override the per-scan trial budget / wall-clock cap via env vars:
#   N_TRIALS=40 TIME_BUDGET_HOURS=3.5 ./gebo_scans_interactive.sh
#
# Per-trial GEBO / L-BFGS knobs (also overridable via env vars):
#   GEBO_TIME_LIMIT_HOURS wall-clock cap on the GEBO stage of each trial --
#                         GEBO stops early (even short of its sampled
#                         n_iterations_gebo target, see configs/gebo_meta_search.yaml)
#                         once this elapses, and the trial still proceeds to
#                         L-BFGS with whatever GEBO found so far (default 2.0)
#   LBFGS_N_EVENTS        events the L-BFGS stage evaluates against, overriding
#                         whatever n_events GEBO's trial happened to sample
#                         (default 20000)

set -euo pipefail

N_TRIALS="${N_TRIALS:-40}"
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-3.9}"   # leave margin under a 4h salloc
GEBO_TIME_LIMIT_HOURS="${GEBO_TIME_LIMIT_HOURS:-2.0}"
LBFGS_N_EVENTS="${LBFGS_N_EVENTS:-20000}"

source /pscratch/sd/a/aelabd/parnassus/parnassus_env/bin/activate
cd /pscratch/sd/a/aelabd/parnassus/src

# export COMET_API_KEY="..."       # uncomment + fill in (or make sure it's already exported)
# export COMET_WORKSPACE="..."

export N_TRIALS TIME_BUDGET_HOURS GEBO_TIME_LIMIT_HOURS LBFGS_N_EVENTS  # read by gebo_scans_run_on_node.sh

mkdir -p logs

LOSSES=(wasserstein_1d soft_hist)
TRUST_REGIONS=(cosine adaptive none)
GRAD_MODES=(grad no_grad)

echo "[launch] N_TRIALS=${N_TRIALS} TIME_BUDGET_HOURS=${TIME_BUDGET_HOURS} GEBO_TIME_LIMIT_HOURS=${GEBO_TIME_LIMIT_HOURS} LBFGS_N_EVENTS=${LBFGS_N_EVENTS}"
echo "[launch] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<not in a Slurm allocation>}"

# This QOS/partition allows only ONE srun job step per node at a time within
# an allocation (verified empirically: a second `srun --nodelist=<already-
# stepped node> --exact ...` is rejected outright -- "Job request does not
# match any supported policy" -- regardless of --gpus-per-task / -G /
# --gpu-bind / --overlap). So there is exactly ONE srun step per node here
# (using the whole node), and gebo_scans_run_on_node.sh fans that single step
# out into 4 plain background processes pinned via CUDA_VISIBLE_DEVICES --
# ordinary Unix process management, not more Slurm steps, so the per-node
# step-count limit doesn't apply to it.
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?not inside a Slurm allocation -- salloc first, see the header comment}")
echo "[launch] nodes: ${NODES[*]}"
if [ "${#NODES[@]}" -lt 3 ]; then
  echo "[launch] WARNING: expected 3 nodes, got ${#NODES[@]} (${NODES[*]}) -- allocation may be smaller than 3x4." >&2
fi

COMBOS=()
for loss in "${LOSSES[@]}"; do
  for tr in "${TRUST_REGIONS[@]}"; do
    for grad in "${GRAD_MODES[@]}"; do
      COMBOS+=("${loss}" "${tr}" "${grad}")
    done
  done
done  # 36 tokens = 12 (loss, tr, grad) triplets, 4 per node below

SCRIPT_DIR="/pscratch/sd/a/aelabd/parnassus/src/parnassus/torch_delphes/tune_cms_fullsim"
for node_idx in 0 1 2; do
  node="${NODES[$node_idx]}"
  start=$(( node_idx * 12 ))
  node_combo_args=("${COMBOS[@]:$start:12}")
  echo "[launch] node=${node}: ${node_combo_args[*]}"
  srun --nodes=1 --ntasks=1 --nodelist="${node}" --exact \
    bash "${SCRIPT_DIR}/gebo_scans_run_on_node.sh" "${node_combo_args[@]}" &
done

echo "[launch] all 3 node steps launched (4 scans each); waiting ..."
wait
echo "[launch] all 12 scans finished (or hit their time budget)."
