#!/bin/bash
# Sequential full-phase-space fit: one tuning run per stage YAML, each starting from the
# previous stage's final parameter values (frozen and fitted alike) with only that
# stage's block trainable. See README.md.
#
# USAGE (compute node with GPUs; from anywhere):
#   bash src/parnassus/torch_delphes/full_phasespace_tuning/run_sequential.sh [stage.yaml ...]
# Default stages: stage1_muons stage2_chads stage3_calo stage4_electrons
# (the files next to this script). Resume from a later stage by listing only the
# remaining YAMLs and pointing FROM_HISTORY at the previous stage's history.json.
#
# Environment overrides:
#   SAMPLE_DIR      dir of the pseudo-data files   (default /global/cfs/cdirs/m3246/diff_delphes/allsamples)
#   SAMPLE_PATTERN  printf pattern with %s = process (default pseudo_data_200k_param_config_all_%s.root)
#   OUT_BASE        output base dir                (default doc/figure_sequential, relative to $REPO)
#   N_STEPS         epochs per stage               (default 100; the CLI's global batch is 4096)
#   N_EVENTS        events per stage               (default -1 = all; small values for a dry run)
#   NPROC           GPUs; >1 launches torchrun     (default 4; per-rank batch = 4096/NPROC)
#   PID_WEIGHTING   --pid-weighting of the per-species shape terms (default fraction: in every
#                   stage the fitted species is the abundant one, so this keeps its terms at full
#                   weight and mutes the stray-species noise/floor; count/pair/log_ht untouched)
#   EARLY_STOP      early-stopping patience in epochs, 0 = off (default 10: the val loss of every
#                   stage sits on a floor from the frozen species, so late epochs only track noise)
#   PICK            best|last epoch carried to the next stage (default best = the early-stopping
#                   checkpoint, i.e. what params_reg.pdf marks; last = the final epoch)
#   PLOT            1 = plot each stage (params_reg.pdf + plot_fit_results figures; default 1)
#   PLOT_N_EVENTS   cap on events used by plot_fit_results (default: full validation split)
#   TRUTH_CONFIG    truth reference for the plots  (default param_configs/param_config_all.yaml)
#   FROM_HISTORY    history.json to start the FIRST listed stage from (default: card defaults)
#   EXTRA_ARGS      extra tuning-CLI args appended to every stage
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../.." && pwd)}"
SAMPLE_DIR="${SAMPLE_DIR:-/global/cfs/cdirs/m3246/diff_delphes/allsamples}"
SAMPLE_PATTERN="${SAMPLE_PATTERN:-pseudo_data_200k_param_config_all_%s.root}"
OUT_BASE="${OUT_BASE:-$REPO/doc/figure_sequential}"
N_STEPS="${N_STEPS:-100}"
N_EVENTS="${N_EVENTS:--1}"
NPROC="${NPROC:-4}"
PID_WEIGHTING="${PID_WEIGHTING:-fraction}"
EARLY_STOP="${EARLY_STOP:-10}"
PICK="${PICK:-best}"
PLOT="${PLOT:-1}"
TRUTH_CONFIG="${TRUTH_CONFIG:-$REPO/src/parnassus/torch_delphes/param_configs/param_config_all.yaml}"
FROM_HISTORY="${FROM_HISTORY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ $# -gt 0 ]]; then
    STAGES=("$@")
else
    STAGES=("$HERE"/stage1_muons.yaml "$HERE"/stage2_chads.yaml "$HERE"/stage3_calo.yaml \
            "$HERE"/stage4_electrons.yaml)
fi

# shellcheck disable=SC1091
source "$REPO/setup.sh"
MK="python -m parnassus.torch_delphes.full_phasespace_tuning.make_stage_config"
if (( NPROC > 1 )); then
    LAUNCH="torchrun --standalone --nproc-per-node=$NPROC -m"
else
    LAUNCH="python -m"
fi

echo "[seq] REPO=$REPO  SAMPLE_DIR=$SAMPLE_DIR  OUT_BASE=$OUT_BASE"
echo "[seq] N_STEPS=$N_STEPS  N_EVENTS=$N_EVENTS  NPROC=$NPROC  PID_WEIGHTING=$PID_WEIGHTING  EARLY_STOP=$EARLY_STOP  PICK=$PICK  PLOT=$PLOT  FROM_HISTORY=${FROM_HISTORY:-<card defaults>}"
mkdir -p "$OUT_BASE"

PREV="$FROM_HISTORY"
for stage in "${STAGES[@]}"; do
    name="$($MK --stage "$stage" --show name)"
    process="$($MK --stage "$stage" --show process)"
    extra="$($MK --stage "$stage" --show extra_args)"
    # shellcheck disable=SC2059
    root_file="$SAMPLE_DIR/$(printf "$SAMPLE_PATTERN" "$process")"
    # Optuna-workspace layout (<stage>/round_0/{materialized_config.yaml,history.json,...})
    # so plot_parameter_regression / plot_fit_results run on it unchanged.
    outdir="$OUT_BASE/$name"
    rdir="$outdir/round_0"
    mkdir -p "$rdir"
    [[ -f "$root_file" ]] || { echo "[seq] $name: sample not found: $root_file" >&2; exit 1; }

    echo "[seq] ===== $name  (process=$process, sample=$root_file) ====="
    $MK --stage "$stage" ${PREV:+--from-history "$PREV"} --pick "$PICK" --out "$rdir/materialized_config.yaml"

    # shellcheck disable=SC2086
    $LAUNCH parnassus.torch_delphes.tune_cms_fullsim \
        --root-file "$root_file" \
        --param-config "$rdir/materialized_config.yaml" \
        --lr 1 \
        --loss wasserstein_1d --mode delphes --pid-weighting "$PID_WEIGHTING" \
        --early-stopping-patience "$EARLY_STOP" --lr-scheduler-patience 0 \
        --n-events "$N_EVENTS" --n-steps "$N_STEPS" \
        --history-path "$rdir/history.json" \
        --intermediate-plot-dir "$rdir/intermediate_plots" \
        $extra $EXTRA_ARGS 2>&1 | tee "$outdir/train.log"

    [[ -f "$rdir/history.json" ]] || { echo "[seq] $name: no history.json written" >&2; exit 1; }
    PREV="$rdir/history.json"

    if [[ "$PLOT" == "1" ]]; then
        # params_reg.pdf (fitted vs truth per block) + the observable / drift figures.
        { python -m parnassus.torch_delphes.plotting_scripts.plot_parameter_regression \
              --workspace "$outdir" --truth-config "$TRUTH_CONFIG" && \
          python -m parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results \
              --history "$rdir/history.json" \
              --root-file "$root_file" \
              --output-dir "$outdir/plots" \
              --truth-config "$TRUTH_CONFIG" ${PLOT_N_EVENTS:+--n-events-for-plots "$PLOT_N_EVENTS"}; } \
            2>&1 | tee "$outdir/plot.log" || echo "[seq] $name: plotting failed (fit result kept)" >&2
    fi
done

# The final fitted card: every parameter at its last-stage value, all frozen.
$MK --from-history "$PREV" --pick "$PICK" --out "$OUT_BASE/fitted_config.yaml"
echo "[seq] done. Final fitted card: $OUT_BASE/fitted_config.yaml"
