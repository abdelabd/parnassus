#!/bin/bash
# Launch all 12 Optuna+GEBO+ADAM scans (2 losses x 3 trust regions x 2 grad
# modes) INTERACTIVELY, one per GPU, packed into an already-active salloc
# allocation of 3 nodes x 4 GPUs (12 GPUs total).
#
# This is gebo_scans_interactive.sh with the fine-tuning stage swapped: instead
# of polishing GEBO's best point with bounded L-BFGS-B on GEBO's own
# full-dataset objective, each trial warm-starts the REAL tune_cms_fullsim Adam
# training loop from it -- stochastic mini-batches, a 70/20 train/val split,
# per-group Adam learning rates, ReduceLROnPlateau decay and early stopping --
# and is scored by that fit's best VALIDATION loss.
#
# In other words this is run_optuna.sh's Optuna+Adam pipeline with GEBO in place
# of Optuna as the thing that picks Adam's 66 parameter initializations. The
# outer Optuna still tunes Adam's lr / batch_size / per-group lr_scale over the
# same ranges optuna_config.yaml uses -- see the `adam:` section of
# configs/gebo_meta_search.yaml -- it just no longer samples the inits.
#
# Because an Adam trial's score is a validation loss and an L-BFGS trial's is a
# full-dataset training objective, the two are NOT comparable, so this script
# keeps its studies in a separate root (SCAN_ROOT, default doc/gebo_adam_scans)
# and its logs in a separate directory (LOG_DIR, default logs_adam). Your
# existing 12 L-BFGS studies under doc/gebo_optuna_scans/ are untouched.
#
# First get the allocation (NOT done by this script):
#
#   salloc -C gpu -q interactive -t 240 --nodes 3 --ntasks-per-node=4 --gpus-per-node=4 -A m3246_g
#
# then, from inside that allocation's shell:
#
#   cd /pscratch/sd/a/aelabd/parnassus
#   source parnassus_env/bin/activate
#   export COMET_API_KEY=...   COMET_WORKSPACE=...
#   ./src/parnassus/torch_delphes/tune_cms_fullsim/optuna_gebo_adam.sh
#
# Re-running this script (in a fresh salloc, after the previous one timed out)
# resumes every scan, exactly as the L-BFGS version does: each scan's Optuna
# study lives in a persistent sqlite DB under its own
# doc/gebo_adam_scans/<scan_name>/, and each trial's GEBO progress is
# checkpointed on disk (see gebo_optuna_search.py's module docstring for the
# two-level resume design). NOTE the one asymmetry with the L-BFGS pipeline: the
# Adam stage has no mid-fit checkpoint of its own, so a trial killed partway
# through Adam redoes that fit from GEBO's best point on resume (a completed
# Adam stage, i.e. one that wrote adam/gebo_summary.json, is never redone).
#
# Comet: each trial's Adam metrics are logged into the SAME experiment as that
# round's GEBO stage, suffixed "_adam" (val_loss_adam, param/<name>_adam, ...),
# so one Comet run shows the whole round end to end.
#
# Override the per-scan trial budget / wall-clock cap via env vars:
#   N_TRIALS=40 TIME_BUDGET_HOURS=3.5 ./optuna_gebo_adam.sh
#
# Per-trial GEBO / Adam knobs (also overridable via env vars):
#   GEBO_TIME_LIMIT_HOURS wall-clock cap on the GEBO stage of each trial --
#                         GEBO stops early (even short of its sampled
#                         n_iterations_gebo target, see configs/gebo_meta_search.yaml)
#                         once this elapses, and the trial still proceeds to
#                         Adam with whatever GEBO found so far (default 2.0)
#   FINETUNE_N_EVENTS     events the Adam stage trains on, overriding whatever
#                         n_events GEBO's trial happened to sample; split 70/20
#                         into train/val (default 20000)
#   FINETUNE_N_STEPS      max Adam epochs per trial (default 200; early stopping
#                         on val_loss usually ends a fit sooner)
#   SCAN_ROOT             root for the 12 studies (default doc/gebo_adam_scans)
#   LOG_DIR               per-scan log directory (default logs_adam)

set -euo pipefail

N_TRIALS="${N_TRIALS:-40}"
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-3.9}"   # leave margin under a 4h salloc
GEBO_TIME_LIMIT_HOURS="${GEBO_TIME_LIMIT_HOURS:-2.0}"
FINETUNE_N_EVENTS="${FINETUNE_N_EVENTS:-100_000}"
FINETUNE_N_STEPS="${FINETUNE_N_STEPS:-200}"
SCAN_ROOT="${SCAN_ROOT:-runs/gebo_adam_scans}"
LOG_DIR="${LOG_DIR:-logs_adam}"
FINETUNE=adam   # the whole point of this script; use gebo_scans_interactive.sh for L-BFGS

source /pscratch/sd/a/aelabd/parnassus/parnassus_env/bin/activate
cd /pscratch/sd/a/aelabd/parnassus/src

# export COMET_API_KEY="..."       # uncomment + fill in (or make sure it's already exported)
# export COMET_WORKSPACE="..."

# read by gebo_scans_run_on_node.sh
export N_TRIALS TIME_BUDGET_HOURS GEBO_TIME_LIMIT_HOURS
export FINETUNE FINETUNE_N_EVENTS FINETUNE_N_STEPS SCAN_ROOT LOG_DIR

mkdir -p "${LOG_DIR}"

LOSSES=(wasserstein_1d soft_hist)
TRUST_REGIONS=(cosine adaptive none)
GRAD_MODES=(grad no_grad)

echo "[launch] finetune=${FINETUNE} N_TRIALS=${N_TRIALS} TIME_BUDGET_HOURS=${TIME_BUDGET_HOURS}"
echo "[launch] GEBO_TIME_LIMIT_HOURS=${GEBO_TIME_LIMIT_HOURS} FINETUNE_N_EVENTS=${FINETUNE_N_EVENTS} FINETUNE_N_STEPS=${FINETUNE_N_STEPS}"
echo "[launch] SCAN_ROOT=${SCAN_ROOT}  LOG_DIR=${LOG_DIR}"
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
