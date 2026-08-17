#!/bin/bash
# Fan out generate_pseudodata across a SLURM job array (NERSC Perlmutter, CPU
# partition), then submit an auto-dependent merge job that concatenates the
# per-seed ROOT files into one and deletes the parts.
#
# USAGE (run on a login node, from anywhere):
#   bash src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh \
#       --config <param_config.yaml> [--process dijet|HZZ4l|muongun] [--n_events 100000] [--n_tasks 10] [--debug]
#
# --config is REQUIRED: the (possibly partial) generation param config, passed
# to generate_pseudodata --param-config; its basename and the process name the output,
#   $OUTBASE/pseudo_data_<total>_<config-stem>_<process>[_debug].root   (e.g. pseudo_data_100k_param_config_chadtrkeff_dijet.root)
# --process: physics process (generate_pseudodata --process; default dijet).
# --n_events: TOTAL number of events (default 100000); --n_tasks: number of array
# tasks (default 10). n_events must be divisible by n_tasks (each task gets n_events/n_tasks).
# --debug: also write the ~400 per-module intermediate branches (generate_pseudodata
# --debug; needed only for plot_fit_results --debug). OFF by default: it is a
# single-threaded uproot write that costs ~120 ms/event (~20 min per 10k events, 7x file size).
#
# Override any other config via the environment, e.g. a cheap dry run:
#   OUTBASE=/tmp/mcgen_test \
#       bash src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh --config my.yaml --n_events 200 --n_tasks 2
#
# Each array task generates N_EVENTS_PER_JOB events with seed = SLURM_ARRAY_TASK_ID
# (1..NJOBS). The seed drives BOTH the Pythia workers (disjoint Random:seed ranges
# per task, so every task produces distinct events) and the target card's smearing.
# Generation runs on the CPU partition (--device cpu; the torch forward is a few
# min per 10k events). Measured per-task cost (dijet): ~0.11 s/event without --debug
# (~1.5 s/event/core Pythia+HepMC on 32 cores + serial read/torch/write), ~0.22 s/event
# with --debug -- pick n_events/n_tasks to fit TIME_LIMIT (default 30 min: ~15k events, ~8k with --debug).
# The muon gun has no hard process and generates in seconds.

set -euo pipefail

# ---- CLI: --config <yaml> (required), --process <name>, --n_events <total>, --n_tasks <n> ----
USAGE="usage: $0 --config <param_config.yaml> [--process dijet|HZZ4l|muongun] [--n_events <total, default 100000>] [--n_tasks <n, default 10>] [--debug]"
PARAM_CONFIG=""
PROCESS="dijet"
DEBUG_FLAG=""
TOTAL=100000
NJOBS=10
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)             PARAM_CONFIG="${2:-}"; shift 2 ;;
        --process)            PROCESS="${2:-}"; shift 2 ;;
        --n_events|--n-events) TOTAL="${2:-}"; shift 2 ;;
        --n_tasks|--n-tasks)   NJOBS="${2:-}"; shift 2 ;;
        --debug)               DEBUG_FLAG="--debug"; shift ;;
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
if ! [[ "$PROCESS" =~ ^(dijet|HZZ4l|muongun)$ ]]; then
    echo "$USAGE  (unknown --process: '$PROCESS')" >&2
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
PT_HAT_MIN="${PT_HAT_MIN:-}"           # empty = generate_pseudodata default (100 for dijet, none otherwise)
N_WORKERS="${N_WORKERS:-32}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"   # per array task (sbatch -t)
# Merged filename: pseudo_data_<total>_<config-stem>_<process>[_debug].root, total as
# "100k" when it is a whole number of thousands.
if (( TOTAL % 1000 == 0 )); then TOTAL_TAG="$((TOTAL / 1000))k"; else TOTAL_TAG="$TOTAL"; fi
MERGED_NAME="${MERGED_NAME:-pseudo_data_${TOTAL_TAG}_${CFG_TAG}_${PROCESS}${DEBUG_FLAG:+_debug}.root}"
# Per-seed part files share the merged stem, so several configs can coexist in PARTS_DIR.
PART_PREFIX="${MERGED_NAME%.root}"

# Exported so the array tasks and the merge job inherit them (sbatch propagates
# the submitting environment by default, SLURM_EXPORT_ENV=ALL).
export REPO ENV_PREFIX OUTBASE PARTS_DIR LOGDIR \
    N_EVENTS_PER_JOB PT_HAT_MIN N_WORKERS NJOBS MERGED_NAME PART_PREFIX PARAM_CONFIG PROCESS DEBUG_FLAG

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[submit] config:"
echo "  PARAM_CONFIG=$PARAM_CONFIG  PROCESS=$PROCESS"
echo "  OUTBASE=$OUTBASE"
echo "  PARTS_DIR=$PARTS_DIR"
echo "  LOGDIR=$LOGDIR"
echo "  NJOBS=$NJOBS  N_EVENTS_PER_JOB=$N_EVENTS_PER_JOB  (total $TOTAL events)"
echo "  N_WORKERS=$N_WORKERS  PT_HAT_MIN=${PT_HAT_MIN:-<default>}  MERGED_NAME=$MERGED_NAME  DEBUG=${DEBUG_FLAG:-off}"
# Rough runtime estimate (measured 2026-08-16: ~0.11 s/event, ~0.22 s/event with --debug).
SEC_PER_100EV=11; [[ -n "$DEBUG_FLAG" ]] && SEC_PER_100EV=22
echo "  TIME_LIMIT=$TIME_LIMIT per task  (estimated ~$(( N_EVENTS_PER_JOB * SEC_PER_100EV / 6000 + 1 )) min/task)"

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
    -t "$TIME_LIMIT" \
    -o "$LOGDIR/gen_%A_%a.out" \
    -e "$LOGDIR/gen_%A_%a.err" \
    <<'EOF'
#!/bin/bash
#SBATCH -A m3246
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -J genpseudo
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/global/cfs/cdirs/m3246/Runze/MCGen/envs/parnassus}"
PARTS_DIR="${PARTS_DIR:-/global/cfs/cdirs/m3246/diff_delphes/parts}"
N_EVENTS_PER_JOB="${N_EVENTS_PER_JOB:-10000}"
N_WORKERS="${N_WORKERS:-32}"
PT_HAT_MIN="${PT_HAT_MIN:-}"
PROCESS="${PROCESS:-dijet}"
PART_PREFIX="${PART_PREFIX:-pseudo_data}"
PARAM_CONFIG="${PARAM_CONFIG:?PARAM_CONFIG must be exported by submit_pseudodata.sh}"

# shellcheck disable=SC1091
source "$ENV_PREFIX/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

OUT="$PARTS_DIR/${PART_PREFIX}_seed${SLURM_ARRAY_TASK_ID}.root"
echo "[gen] seed=$SLURM_ARRAY_TASK_ID  n_events=$N_EVENTS_PER_JOB  process=$PROCESS  config=$PARAM_CONFIG  -> $OUT"

python -m parnassus.torch_delphes.generate_pseudodata \
    --output "$OUT" \
    --n-events "$N_EVENTS_PER_JOB" \
    --n-workers "$N_WORKERS" \
    --process "$PROCESS" \
    ${PT_HAT_MIN:+--pt-hat-min "$PT_HAT_MIN"} \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --param-config "$PARAM_CONFIG" \
    --device cpu \
    ${DEBUG_FLAG:-}
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
