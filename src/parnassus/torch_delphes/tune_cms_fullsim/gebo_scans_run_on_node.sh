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
# the caller's environment by default). GEBO_TIME_LIMIT_HOURS / FINETUNE_N_EVENTS
# / FINETUNE / SCAN_ROOT are likewise read from the environment but default here
# (so this script works whether or not its caller set them -- gebo_scans_submit.sh
# does not currently export them all). GEBO's own iteration budget is no longer a
# fixed knob here -- it is sampled PER TRIAL as n_iterations_gebo in
# configs/gebo_meta_search.yaml, capped by GEBO_TIME_LIMIT_HOURS.
#
# FINETUNE selects which optimizer polishes GEBO's best point: "lbfgs" (bounded
# L-BFGS-B on GEBO's own objective) or "adam" (the real tune_cms_fullsim
# training loop, warm-started from that point -- see optuna_gebo_adam.sh). The
# two produce incomparable trial scores, so SCAN_ROOT keeps their studies apart.
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

GEBO_TIME_LIMIT_HOURS="${GEBO_TIME_LIMIT_HOURS:-2.0}"
FINETUNE="${FINETUNE:-lbfgs}"
# Accept the old LBFGS_N_EVENTS spelling so existing launchers keep working.
FINETUNE_N_EVENTS="${FINETUNE_N_EVENTS:-${LBFGS_N_EVENTS:-20000}}"
FINETUNE_N_STEPS="${FINETUNE_N_STEPS:-200}"
SCAN_ROOT="${SCAN_ROOT:-}"   # empty -> gebo_optuna_search.py's per-finetune default
# Per-pipeline log dir, so an Adam scan does not overwrite the L-BFGS scan's
# logs/<scan_name>.log (the two share scan names by design).
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "${LOG_DIR}"

gpu=0
while [ "$#" -gt 0 ]; do
  loss="$1"; tr="$2"; grad="$3"; shift 3
  scan_name="${loss}__${tr}__${grad}"
  logfile="${LOG_DIR}/${scan_name}.log"
  echo "[launch] $(hostname): ${scan_name} (finetune=${FINETUNE}) -> CUDA_VISIBLE_DEVICES=${gpu} -> ${logfile}"
  output_base_args=()
  if [ -n "${SCAN_ROOT}" ]; then
    output_base_args=(--output-base "${SCAN_ROOT}/${scan_name}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python -m parnassus.torch_delphes.tune_cms_fullsim.gebo_optuna_search \
    --loss "${loss}" --trust-region "${tr}" --grad-mode "${grad}" \
    --n-trials "${N_TRIALS}" --time-budget-hours "${TIME_BUDGET_HOURS}" \
    --gebo-time-limit-hours "${GEBO_TIME_LIMIT_HOURS}" \
    --finetune "${FINETUNE}" \
    --finetune-n-events "${FINETUNE_N_EVENTS}" \
    --finetune-n-steps "${FINETUNE_N_STEPS}" \
    "${output_base_args[@]}" \
    > "${logfile}" 2>&1 &
  gpu=$(( gpu + 1 ))
done

echo "[launch] $(hostname): all $((gpu)) scans launched on this node; waiting ..."
wait
echo "[launch] $(hostname): all scans on this node finished (or hit their time budget)."
