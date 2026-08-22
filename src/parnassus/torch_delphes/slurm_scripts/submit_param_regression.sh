#!/bin/bash
# Parameter-regression seed scan for ONE particle gun: a SLURM job array with one task per
# torch seed (SLURM_ARRAY_TASK_ID == optuna_search --seed, default 0-4), each on one shared
# A100 (NERSC Perlmutter, -C gpu -q shared) for TIME_LIMIT. Same login-node wrapper +
# quoted-heredoc pattern as submit_pseudodata.sh.
#
# USAGE (login node, from anywhere):
#   bash src/parnassus/torch_delphes/slurm_scripts/submit_param_regression.sh [--gun] <muongun|ksgun|electrongun|dijet>
#   (the config keys muons|chads|electrons|dijets are accepted as aliases; the results dir
#   always uses the process name, e.g. --gun muons -> results/muongun_<seed>)
#   for g in muongun ksgun electrongun dijet; do bash .../submit_param_regression.sh $g; done   # all four
#   SEEDS=3 bash .../submit_param_regression.sh dijet                                           # redo one seed
#   TIME_LIMIT=02:00:00 bash .../submit_param_regression.sh dijet                               # more wall time
#
# Each task fits the gun's 200k pseudodata
#   $OUTBASE/pseudo_data_200k_param_config_<key>_<gun>.root   (key = muons|chads|electrons|dijets)
# with optuna_config_<key>.yaml (single enqueued trial) into $RESULTS/<gun>_<seed>/
# (round_0/history.json, all_optuna.json), then runs plot_parameter_regression (CPU, seconds)
# for the per-seed sanity PDF $RESULTS/<gun>_<seed>/plots/params_reg.pdf. Aggregate with
#   python -m parnassus.torch_delphes.plotting_scripts.plot_parameter_summary --results $RESULTS
#
# --seed is the torch training seed (train/val event split, shuffle order, smearing noise);
# with --n-trials 1 it is the only randomness. history.json is written only when the fit
# ENDS, so a task killed at TIME_LIMIT loses its seed. Measured single-GPU wall times:
# ~10-16 min for the guns, 37 min for dijet (36 of max 100 epochs) -> TIME_LIMIT=02:00:00
# for dijet if a seed times out.

set -euo pipefail

USAGE="usage: $0 [--gun] <muongun|ksgun|electrongun|dijet>  (config keys muons|chads|electrons|dijets also accepted)"
if [[ "${1:-}" == "--gun" || "${1:-}" == "--GUN" ]]; then shift; fi
GUN="${1:-}"
# Process name (results-dir prefix, as plot_parameter_summary expects) <-> config key.
case "$GUN" in
    muongun|muons)         GUN=muongun;     KEY=muons ;;
    ksgun|chads)           GUN=ksgun;       KEY=chads ;;
    electrongun|electrons) GUN=electrongun; KEY=electrons ;;
    dijet|dijets)          GUN=dijet;       KEY=dijets ;;
    *) echo "$USAGE  (unknown gun: '$GUN')" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then echo "$USAGE  (unexpected args: ${*:2})" >&2; exit 2; fi

# ---- Config (override via environment) ----
REPO="${REPO:-/global/u2/m/mukyu/MCGen/torch_delphes/parnassus}"
ENV_PREFIX="${ENV_PREFIX:-/global/cfs/cdirs/m3246/Runze/MCGen/envs/parnassus}"
OUTBASE="${OUTBASE:-/global/cfs/cdirs/m3246/diff_delphes}"
RESULTS="${RESULTS:-$OUTBASE/results}"
LOGDIR="${LOGDIR:-$RESULTS/logs}"
SEEDS="${SEEDS:-0-4}"                  # sbatch --array spec; task id == --seed
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"   # per array task (sbatch -t)
N_EVENTS="${N_EVENTS:-200000}"
N_STEPS="${N_STEPS:-100}"
CFG_DIR="src/parnassus/torch_delphes/param_configs"   # relative to $REPO (the task cd's there)
ROOT_FILE="$OUTBASE/pseudo_data_200k_param_config_${KEY}_${GUN}.root"
OPTUNA_CFG="$CFG_DIR/optuna_config_${KEY}.yaml"
TRUTH_CFG="$CFG_DIR/param_config_${KEY}.yaml"
for f in "$ROOT_FILE" "$REPO/$OPTUNA_CFG" "$REPO/$TRUTH_CFG"; do
    if [[ ! -f "$f" ]]; then echo "[submit] missing input: $f" >&2; exit 2; fi
done

# Exported so the array tasks inherit them (sbatch propagates the submitting environment).
export REPO ENV_PREFIX RESULTS GUN ROOT_FILE OPTUNA_CFG TRUTH_CFG N_EVENTS N_STEPS

echo "[submit] gun=$GUN (key=$KEY)  seeds=$SEEDS  TIME_LIMIT=$TIME_LIMIT  N_EVENTS=$N_EVENTS  N_STEPS=$N_STEPS"
echo "  ROOT_FILE=$ROOT_FILE"
echo "  OPTUNA_CFG=$OPTUNA_CFG  TRUTH_CFG=$TRUTH_CFG"
echo "  output: $RESULTS/${GUN}_<seed>/   logs: $LOGDIR/${GUN}_<seed>.{out,err}"

# ---- Build/activate the uv env once here on the login node, so the array tasks don't each
#      fire a concurrent `uv sync` (CFS flock contention; see submit_pseudodata.sh). ----
echo "[submit] sourcing setup.sh to ensure the uv env is built ..."
# shellcheck disable=SC1091
source "$REPO/setup.sh"

mkdir -p "$RESULTS" "$LOGDIR"

# ---- Submit the array. The heredoc is QUOTED ('EOF') so $SLURM_* stays literal; --array, -t,
#      -J and the log paths are sbatch CLI args so this shell expands $GUN/$LOGDIR while SLURM
#      expands %a (== the seed). ----
JID=$(sbatch --parsable \
    --array="$SEEDS" \
    -t "$TIME_LIMIT" \
    -J "reg_${GUN}" \
    -o "$LOGDIR/${GUN}_%a.out" \
    -e "$LOGDIR/${GUN}_%a.err" \
    <<'EOF'
#!/bin/bash
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
set -euo pipefail

REPO="${REPO:?exported by submit_param_regression.sh}"
ENV_PREFIX="${ENV_PREFIX:?exported by submit_param_regression.sh}"
RESULTS="${RESULTS:?exported by submit_param_regression.sh}"
GUN="${GUN:?exported by submit_param_regression.sh}"
ROOT_FILE="${ROOT_FILE:?exported by submit_param_regression.sh}"
OPTUNA_CFG="${OPTUNA_CFG:?exported by submit_param_regression.sh}"
TRUTH_CFG="${TRUTH_CFG:?exported by submit_param_regression.sh}"
N_EVENTS="${N_EVENTS:-200000}"
N_STEPS="${N_STEPS:-100}"

cd "$REPO"   # the config paths are relative to the repo root
# shellcheck disable=SC1091
source "$ENV_PREFIX/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

SEED="$SLURM_ARRAY_TASK_ID"
OUT="$RESULTS/${GUN}_${SEED}"
echo "[reg] gun=$GUN seed=$SEED -> $OUT"
nvidia-smi -L || echo "[reg] no GPU visible"

python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
    --root-file "$ROOT_FILE" \
    --optuna-config "$OPTUNA_CFG" \
    --n-events "$N_EVENTS" \
    --n-steps "$N_STEPS" \
    --n-trials 1 \
    --loss wasserstein_1d \
    --output-base "$OUT" \
    --history-path "$OUT/all_optuna.json" \
    --mode delphes \
    --seed "$SEED"

python -m parnassus.torch_delphes.plotting_scripts.plot_parameter_regression \
    --workspace "$OUT" \
    --truth-config "$TRUTH_CFG"
echo "[reg] DONE gun=$GUN seed=$SEED"
EOF
)
echo "[submit] job array id: $JID  (seeds $SEEDS)"
echo "[submit] watch with:  squeue -u \"\$USER\" -j $JID"
