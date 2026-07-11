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

set -euo pipefail

N_TRIALS="${N_TRIALS:-40}"
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-3.5}"   # leave margin under a 4h salloc

source /pscratch/sd/a/aelabd/parnassus/parnassus_env/bin/activate
cd /pscratch/sd/a/aelabd/parnassus/src

# export COMET_API_KEY="..."       # uncomment + fill in (or make sure it's already exported)
# export COMET_WORKSPACE="..."

mkdir -p logs

LOSSES=(wasserstein_1d soft_hist)
TRUST_REGIONS=(cosine adaptive none)
GRAD_MODES=(grad no_grad)

echo "[launch] N_TRIALS=${N_TRIALS} TIME_BUDGET_HOURS=${TIME_BUDGET_HOURS}"
echo "[launch] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<not in a Slurm allocation>}"

for loss in "${LOSSES[@]}"; do
  for tr in "${TRUST_REGIONS[@]}"; do
    for grad in "${GRAD_MODES[@]}"; do
      scan_name="${loss}__${tr}__${grad}"
      logfile="logs/${scan_name}.log"
      echo "[launch] ${scan_name}  -> ${logfile}"
      # -G 1 --exact: carve exactly 1 GPU out of the allocation for this step;
      # Slurm packs the 12 concurrent steps 4-per-node across the 3 nodes and
      # sets CUDA_VISIBLE_DEVICES correctly inside each step automatically.
      srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 --exact \
        python -m parnassus.torch_delphes.tune_cms_fullsim.gebo_optuna_search \
          --loss "${loss}" --trust-region "${tr}" --grad-mode "${grad}" \
          --n-trials "${N_TRIALS}" --time-budget-hours "${TIME_BUDGET_HOURS}" \
          > "${logfile}" 2>&1 &
    done
  done
done

echo "[launch] all 12 scans launched; waiting ..."
wait
echo "[launch] all 12 scans finished (or hit their time budget)."
