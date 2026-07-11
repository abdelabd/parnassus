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

mkdir -p logs

LOSSES="wasserstein_1d soft_hist"
TRUST_REGIONS="cosine adaptive none"
GRAD_MODES="grad no_grad"

echo "[launch] N_TRIALS=${N_TRIALS} TIME_BUDGET_HOURS=${TIME_BUDGET_HOURS}"
echo "[launch] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<none>}"

for loss in $LOSSES; do
  for tr in $TRUST_REGIONS; do
    for grad in $GRAD_MODES; do
      scan_name="${loss}__${tr}__${grad}"
      logfile="logs/${scan_name}.log"
      echo "[launch] ${scan_name} -> ${logfile}"
      # -G 1 --exact: carve exactly 1 GPU out of the allocation per step;
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
