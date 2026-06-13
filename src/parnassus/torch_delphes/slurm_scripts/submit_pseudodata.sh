#!/bin/bash
# Fan out generate_pseudodata across a SLURM job array (NERSC Perlmutter, CPU
# partition), then submit an auto-dependent merge job that concatenates the
# per-seed ROOT files into one and deletes the parts.
#
# USAGE (run on a login node, from anywhere):
#   bash src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh
#
# Override any config via the environment, e.g. a cheap dry run:
#   N_EVENTS_PER_JOB=100 NJOBS=2 OUTBASE=/tmp/mcgen_test \
#       bash src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh
#
# Each array task generates N_EVENTS_PER_JOB events with seed = SLURM_ARRAY_TASK_ID
# (1..NJOBS), giving NJOBS x N_EVENTS_PER_JOB events total (default 10 x 10000 = 100k).
# The GPU is used for only ~12s of a ~13-min job, so generation runs on the CPU
# partition with --device cpu.

set -euo pipefail

# ---- Config (override via environment) ----
REPO="${REPO:-/global/u2/m/mukyu/MCGen/torch_delphes/parnassus}"
ENV_PREFIX="${ENV_PREFIX:-/global/cfs/cdirs/m3246/Runze/MCGen/envs/parnassus}"
OUTBASE="${OUTBASE:-/global/cfs/cdirs/m3246/Runze/MCGen/data}"
PARTS_DIR="${PARTS_DIR:-$OUTBASE/parts}"
LOGDIR="${LOGDIR:-$OUTBASE/logs}"
N_EVENTS_PER_JOB="${N_EVENTS_PER_JOB:-10000}"
PT_HAT_MIN="${PT_HAT_MIN:-100}"
N_WORKERS="${N_WORKERS:-32}"
NJOBS="${NJOBS:-10}"
MERGED_NAME="${MERGED_NAME:-cms_pseudodata_100k.root}"

# Exported so the array tasks and the merge job inherit them (sbatch propagates
# the submitting environment by default, SLURM_EXPORT_ENV=ALL).
export REPO ENV_PREFIX OUTBASE PARTS_DIR LOGDIR \
    N_EVENTS_PER_JOB PT_HAT_MIN N_WORKERS NJOBS MERGED_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[submit] config:"
echo "  OUTBASE=$OUTBASE"
echo "  PARTS_DIR=$PARTS_DIR"
echo "  LOGDIR=$LOGDIR"
echo "  NJOBS=$NJOBS  N_EVENTS_PER_JOB=$N_EVENTS_PER_JOB  (total $((NJOBS * N_EVENTS_PER_JOB)) events)"
echo "  N_WORKERS=$N_WORKERS  PT_HAT_MIN=$PT_HAT_MIN  MERGED_NAME=$MERGED_NAME"

# ---- Build/activate the uv env once here on the login node, so the NJOBS array
#      tasks don't each fire a concurrent `uv sync` (CFS/GPFS flock contention). ----
echo "[submit] sourcing setup.sh to ensure the uv env is built ..."
# shellcheck disable=SC1091
source "$REPO/setup.sh"

mkdir -p "$PARTS_DIR" "$LOGDIR"

# ---- Submit the generation array. The heredoc is QUOTED ('EOF') so $SLURM_* and
#      $config references stay literal and are resolved at runtime from the
#      inherited environment. --array and the log paths are passed as CLI args so
#      this shell expands $LOGDIR while SLURM expands %A/%a. ----
ARRAY_JID=$(sbatch --parsable \
    --array="1-${NJOBS}" \
    -o "$LOGDIR/gen_%A_%a.out" \
    -e "$LOGDIR/gen_%A_%a.err" \
    <<'EOF'
#!/bin/bash
#SBATCH -A m3246
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 00:59:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -J genpseudo
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/global/cfs/cdirs/m3246/Runze/MCGen/envs/parnassus}"
PARTS_DIR="${PARTS_DIR:-/global/cfs/cdirs/m3246/Runze/MCGen/data/parts}"
N_EVENTS_PER_JOB="${N_EVENTS_PER_JOB:-10000}"
N_WORKERS="${N_WORKERS:-32}"
PT_HAT_MIN="${PT_HAT_MIN:-100}"

# shellcheck disable=SC1091
source "$ENV_PREFIX/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

OUT="$PARTS_DIR/cms_pseudodata_seed${SLURM_ARRAY_TASK_ID}.root"
echo "[gen] seed=$SLURM_ARRAY_TASK_ID  n_events=$N_EVENTS_PER_JOB  -> $OUT"

python -m parnassus.torch_delphes.generate_pseudodata \
    --output "$OUT" \
    --n-events "$N_EVENTS_PER_JOB" \
    --n-workers "$N_WORKERS" \
    --pt-hat-min "$PT_HAT_MIN" \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --device cpu
EOF
)
echo "[submit] generation array job id: $ARRAY_JID  (tasks 1-$NJOBS)"

# ---- Submit the merge with an afterok dependency on the whole array. It runs
#      only if ALL array tasks succeed. If a task fails, rerun that seed and then
#      `sbatch merge_pseudodata.sbatch` by hand (or change afterok -> afterany to
#      merge whatever did succeed). ----
MERGE_JID=$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JID}" \
    -o "$LOGDIR/merge_%j.out" \
    -e "$LOGDIR/merge_%j.err" \
    "$SCRIPT_DIR/merge_pseudodata.sbatch")
echo "[submit] merge job id: $MERGE_JID  (afterok:$ARRAY_JID)"
echo "[submit] merged output will be: $OUTBASE/$MERGED_NAME"
echo "[submit] watch with:  squeue -u \"\$USER\" -j ${ARRAY_JID},${MERGE_JID}"
