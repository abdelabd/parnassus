#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes 3
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4
#SBATCH -t 24:00:00
#SBATCH -A m3246_g
#SBATCH -J gebo_scans

#SBATCH --mail-user=aelabd2@uw.edu
#SBATCH --mail-type=ALL

# Batch submission for all 12 gebo_optuna_search.py scans (2 losses x 3 trust
# regions x 2 grad modes), one per GPU across 3 nodes x 4 GPUs.
#
#   sbatch src/parnassus/torch_delphes/tune_cms_fullsim/gebo_scans_submit.sh
#
# Safe to resubmit after this job times out / gets killed: each scan's Optuna
# study lives in a persistent sqlite DB under its own
# doc/gebo_optuna_scans/<scan_name>/, and every trial's GEBO/L-BFGS progress
# is checkpointed on disk, so a fresh submission of this SAME script resumes
# every scan where it left off (see gebo_optuna_search.py's module docstring
# for the two-level resume design: study-level via persistent storage,
# trial-level via Optuna's heartbeat + RetryFailedTrialCallback).
#
# Override the per-scan trial budget / wall-clock cap at submit time, e.g.:
#   sbatch --export=ALL,N_TRIALS=80,TIME_BUDGET_HOURS=47 gebo_scans_submit.sh
# (also bump #SBATCH -t accordingly if you do).

set -euo pipefail

N_TRIALS="${N_TRIALS:-40}"
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-23}"   # leave ~1h margin under the 24h -t above

cd /pscratch/sd/a/aelabd/parnassus
source parnassus_env/bin/activate
cd src

# export COMET_API_KEY="..."       # uncomment + fill in (or make sure it's already exported)
# export COMET_WORKSPACE="..."

export N_TRIALS TIME_BUDGET_HOURS  # read by gebo_scans_run_on_node.sh

mkdir -p logs

LOSSES="wasserstein_1d soft_hist"
TRUST_REGIONS="cosine adaptive none"
GRAD_MODES="grad no_grad"
SCRIPT_DIR="/pscratch/sd/a/aelabd/parnassus/src/parnassus/torch_delphes/tune_cms_fullsim"

echo "[launch] N_TRIALS=${N_TRIALS} TIME_BUDGET_HOURS=${TIME_BUDGET_HOURS}"
echo "[launch] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<none>}"

# This QOS/partition allows only ONE srun job step per node at a time within
# an allocation (verified empirically: a second `srun --nodelist=<already-
# stepped node> --exact ...` is rejected outright -- "Job request does not
# match any supported policy" -- regardless of --gpus-per-task / -G /
# --gpu-bind / --overlap). So there is exactly ONE srun step per node here
# (using the whole node), and gebo_scans_run_on_node.sh fans that single step
# out into 4 plain background processes pinned via CUDA_VISIBLE_DEVICES --
# ordinary Unix process management, not more Slurm steps, so the per-node
# step-count limit doesn't apply to it.
# (POSIX sh has no arrays, so nodes/combos are picked out of newline lists
# with sed rather than indexed directly.)
NODES=$(scontrol show hostnames "${SLURM_JOB_NODELIST:?not inside a Slurm allocation}")
n_nodes=$(printf '%s\n' "$NODES" | wc -l)
echo "[launch] nodes: $(printf '%s ' $NODES)"
if [ "$n_nodes" -lt 3 ]; then
  echo "[launch] WARNING: expected 3 nodes, got ${n_nodes} -- allocation may be smaller than 3x4." >&2
fi

COMBOS=""
for loss in $LOSSES; do
  for tr in $TRUST_REGIONS; do
    for grad in $GRAD_MODES; do
      COMBOS="${COMBOS}${loss} ${tr} ${grad}
"
    done
  done
done  # 12 newline-separated "loss tr grad" lines, 4 assigned to each node below

node_idx=1
for start in 1 5 9; do
  end=$(( start + 3 ))
  node=$(printf '%s\n' "$NODES" | sed -n "${node_idx}p")
  node_combo_args=$(printf '%s' "$COMBOS" | sed -n "${start},${end}p" | tr '\n' ' ')
  echo "[launch] node=${node}: ${node_combo_args}"
  srun --nodes=1 --ntasks=1 --nodelist="${node}" --exact \
    bash "${SCRIPT_DIR}/gebo_scans_run_on_node.sh" ${node_combo_args} &
  node_idx=$(( node_idx + 1 ))
done

echo "[launch] all 3 node steps launched (4 scans each); waiting ..."
wait
echo "[launch] all 12 scans finished (or hit their time budget)."
