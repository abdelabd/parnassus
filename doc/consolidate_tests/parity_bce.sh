#!/bin/bash
# CONSOLIDATE_MODES_PLAN.md 1c.1 — BCE parity: consolidate_modes --existence bce
# vs the BCE_eff tip (4773249) with the champion flags. Short runs (50k events,
# 10 epochs, no early stop, no plots), same defaults/seeds; compare history.json
# trajectories per stage.
set -euo pipefail
MAIN=/pscratch/sd/a/aelabd/parnassus
WT=/pscratch/sd/a/aelabd/worktrees/bce_eff_ref
COMMON="N_EVENTS=50000 N_STEPS=10 EARLY_STOP=0 PLOT=0 NPROC=4"
SAMPLES="SAMPLE_PATTERN=pseudo_data_200k_param_config_all_%s_hungarian_matched_survival.root"

echo "=== REF: BCE_eff@4773249 (worktree) ==="
cd "$WT"
env $COMMON $SAMPLES \
  PYTHONPATH="$WT/src" \
  OUT_BASE="$MAIN/doc/consolidate_tests/bce_ref" \
  EXTRA_ARGS="--calo-bce --calo-bce-grads detach --calo-bce-threshold self_consistent --calo-count-weight 0" \
  bash "$WT/src/parnassus/torch_delphes/full_phasespace_tuning/run_sequential.sh"

echo "=== NEW: consolidate_modes --existence bce ==="
cd "$MAIN"
env $COMMON $SAMPLES \
  OUT_BASE="$MAIN/doc/consolidate_tests/bce_new" \
  EXTRA_ARGS="--existence bce" \
  bash "$MAIN/src/parnassus/torch_delphes/full_phasespace_tuning/run_sequential.sh"
echo "=== PAIR DONE ==="
