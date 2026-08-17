#!/bin/bash
# Fan out generate_pseudodata across a SLURM job array (NERSC Perlmutter, CPU
# partition), then submit an auto-dependent merge job that concatenates the
# per-seed ROOT files into one and deletes the parts.
#
# USAGE (run on a login node, from anywhere):
#   bash src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh \
#       --config <param_config.yaml> [--n_events 100000] [--n_tasks 10]
#
# --config is REQUIRED: the (possibly partial) generation param config, passed
# to generate_pseudodata --param-config; its basename names the output,
#   $OUTBASE/pseudo_data_<total>_<config-stem>.root   (e.g. pseudo_data_100k_param_config_chadtrkeff.root)
# --n_events: TOTAL number of events (default 100000); --n_tasks: number of array
# tasks (default 10). n_events must be divisible by n_tasks (each task gets n_events/n_tasks).
#
# Override any other config via the environment, e.g. a cheap dry run:
#   OUTBASE=/tmp/mcgen_test \
#       bash src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh --config my.yaml --n_events 200 --n_tasks 2
#
# Each array task generates N_EVENTS_PER_JOB events with seed = SLURM_ARRAY_TASK_ID
# (1..NJOBS). The seed drives BOTH the Pythia workers (disjoint Random:seed ranges
# per task, so every task produces distinct events) and the target card's smearing.
# The GPU is used for only ~12s of a ~13-min job, so generation runs on the CPU
# partition with --device cpu.

set -euo pipefail

# ---- CLI: --config <yaml> (required), --n_events <total>, --n_tasks <n> ----
USAGE="usage: $0 --config <param_config.yaml> [--n_events <total, default 100000>] [--n_tasks <n, default 10>]"
PARAM_CONFIG=""
TOTAL=100000
NJOBS=10
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)             PARAM_CONFIG="${2:-}"; shift 2 ;;
        --n_events|--n-events) TOTAL="${2:-}"; shift 2 ;;
        --n_tasks|--n-tasks)   NJOBS="${2:-}"; shift 2 ;;
        *) echo "$USAGE  (unknown arg: $1)" >&2; exit 2 ;;
    esac
done
if [[ -z "$PARAM_CONFIG" || ! -f "$PARAM_CONFIG" ]]; then
    echo "$USAGE  (--config missing or not a file: '${PARAM_CONFIG}')" >&2
    exit 2
fi
if ! [[ "$TOTAL" =~ ^[0-9]+$ && "$NJOBS" =~ ^[0-9]+$ ]] || (( NJOBS < 1 || TOTAL % NJOBS != 0 )); then
    echo "$USAGE  (--n_events=$TOTAL must be a positive multiple of --n_tasks=$NJOBS)" >&2
    exit 2
fi
PARAM_CONFIG="$(realpath "$PARAM_CONFIG")"
CFG_TAG="$(basename "$PARAM_CONFIG" .yaml)"
N_EVENTS_PER_JOB=$((TOTAL / NJOBS))

# ---- Config (override via environment) ----
REPO="${REPO:-/global/u2/m/mukyu/MCGen/torch_delphes/parnassus}"
ENV_PREFIX="${ENV_PREFIX:-/global/cfs/cdirs/m3246/Runze/MCGen/envs/parnassus}"
OUTBASE="${OUTBASE:-/global/cfs/cdirs/m3246/diff_delphes}"
PARTS_DIR="${PARTS_DIR:-$OUTBASE/parts}"
LOGDIR="${LOGDIR:-$OUTBASE/logs}"
PT_HAT_MIN="${PT_HAT_MIN:-100}"
N_WORKERS="${N_WORKERS:-32}"
# Merged filename: pseudo_data_<total>_<config-stem>.root, total as "100k" when
# it is a whole number of thousands.
if (( TOTAL % 1000 == 0 )); then TOTAL_TAG="$((TOTAL / 1000))k"; else TOTAL_TAG="$TOTAL"; fi
MERGED_NAME="${MERGED_NAME:-pseudo_data_${TOTAL_TAG}_${CFG_TAG}.root}"
# Per-seed part files share the merged stem, so several configs can coexist in PARTS_DIR.
PART_PREFIX="${MERGED_NAME%.root}"

# Exported so the array tasks and the merge job inherit them (sbatch propagates
# the submitting environment by default, SLURM_EXPORT_ENV=ALL).
export REPO ENV_PREFIX OUTBASE PARTS_DIR LOGDIR \
    N_EVENTS_PER_JOB PT_HAT_MIN N_WORKERS NJOBS MERGED_NAME PART_PREFIX PARAM_CONFIG

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[submit] config:"
echo "  PARAM_CONFIG=$PARAM_CONFIG"
echo "  OUTBASE=$OUTBASE"
echo "  PARTS_DIR=$PARTS_DIR"
echo "  LOGDIR=$LOGDIR"
echo "  NJOBS=$NJOBS  N_EVENTS_PER_JOB=$N_EVENTS_PER_JOB  (total $TOTAL events)"
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
PARTS_DIR="${PARTS_DIR:-/global/cfs/cdirs/m3246/diff_delphes/parts}"
N_EVENTS_PER_JOB="${N_EVENTS_PER_JOB:-10000}"
N_WORKERS="${N_WORKERS:-32}"
PT_HAT_MIN="${PT_HAT_MIN:-100}"
PART_PREFIX="${PART_PREFIX:-pseudo_data}"
PARAM_CONFIG="${PARAM_CONFIG:?PARAM_CONFIG must be exported by submit_pseudodata.sh}"

# shellcheck disable=SC1091
source "$ENV_PREFIX/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

OUT="$PARTS_DIR/${PART_PREFIX}_seed${SLURM_ARRAY_TASK_ID}.root"
echo "[gen] seed=$SLURM_ARRAY_TASK_ID  n_events=$N_EVENTS_PER_JOB  config=$PARAM_CONFIG  -> $OUT"

python -m parnassus.torch_delphes.generate_pseudodata \
    --output "$OUT" \
    --n-events "$N_EVENTS_PER_JOB" \
    --n-workers "$N_WORKERS" \
    --pt-hat-min "$PT_HAT_MIN" \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --param-config "$PARAM_CONFIG" \
    --device cpu \
    --debug
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
