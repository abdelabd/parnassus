#!/bin/bash
# Run 4 gebo_optuna_search.py scans concurrently on the CURRENT node, one per
# GPU (CUDA_VISIBLE_DEVICES=0..3, plain background processes -- no further
# srun/Slurm involvement).
#
# Invoked as (see gebo_scans_interactive.sh / gebo_scans_submit.sh):
#   srun --nodes=1 --ntasks=1 --nodelist=<node> --exact \
#     bash gebo_scans_run_on_node.sh <loss1> <tr1> <grad1> <loss2> <tr2> <grad2> ... (4 triplets, 12 args)
#
# N_TRIALS / TIME_BUDGET_HOURS are read from the environment (srun propagates
# the caller's environment by default). GEBO_N_ITERATIONS / GEBO_TIME_LIMIT_HOURS
# / LBFGS_N_EVENTS are likewise read from the environment but default here (so
# this script works whether or not its caller set them -- gebo_scans_submit.sh
# does not currently export them).
#
# Why this exists: this cluster's interactive/urgent GPU QOS allows only ONE
# srun job step per node at a time within an allocation -- verified
# empirically: a second `srun --nodelist=<already-stepped node> --exact ...`
# is rejected outright ("Job request does not match any supported policy"),
# regardless of --gpus-per-task / -G / --gpu-bind / --overlap. So instead of
# 12 srun steps (1 per GPU, the naive approach -- only 1 in 4 concurrent
# attempts per node ever succeeded, the rest sat forever retrying "Requested
# nodes are busy"), there is exactly ONE srun step per node (using the WHOLE
# node, all 4 GPUs visible within it), and this script fans that single step
# out into 4 ordinary background processes -- ordinary Unix process
# management, which the per-node step-count policy has no say over.

set -euo pipefail

GEBO_N_ITERATIONS="${GEBO_N_ITERATIONS:-300}"
GEBO_TIME_LIMIT_HOURS="${GEBO_TIME_LIMIT_HOURS:-2.0}"
LBFGS_N_EVENTS="${LBFGS_N_EVENTS:-20000}"

mkdir -p logs

gpu=0
while [ "$#" -gt 0 ]; do
  loss="$1"; tr="$2"; grad="$3"; shift 3
  scan_name="${loss}__${tr}__${grad}"
  logfile="logs/${scan_name}.log"
  echo "[launch] $(hostname): ${scan_name} -> CUDA_VISIBLE_DEVICES=${gpu} -> ${logfile}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m parnassus.torch_delphes.tune_cms_fullsim.gebo_optuna_search \
    --loss "${loss}" --trust-region "${tr}" --grad-mode "${grad}" \
    --n-trials "${N_TRIALS}" --time-budget-hours "${TIME_BUDGET_HOURS}" \
    --gebo-n-iterations "${GEBO_N_ITERATIONS}" --gebo-time-limit-hours "${GEBO_TIME_LIMIT_HOURS}" \
    --lbfgs-n-events "${LBFGS_N_EVENTS}" \
    > "${logfile}" 2>&1 &
  gpu=$(( gpu + 1 ))
done

echo "[launch] $(hostname): all $((gpu)) scans launched on this node; waiting ..."
wait
echo "[launch] $(hostname): all scans on this node finished (or hit their time budget)."
