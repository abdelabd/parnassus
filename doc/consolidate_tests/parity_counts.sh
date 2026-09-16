#!/bin/bash
# CONSOLIDATE_MODES_PLAN.md 1c.1 — counts parity: consolidate_modes --existence counts
# vs the diff_delphes tip (f777dd5) bare defaults, unlabeled samples. Short runs (50k events,
# 10 epochs, no early stop, no plots), same defaults/seeds; compare history.json
# trajectories per stage.
set -euo pipefail
MAIN=/pscratch/sd/a/aelabd/parnassus
WT=/pscratch/sd/a/aelabd/worktrees/diff_delphes_ref
COMMON="N_EVENTS=50000 N_STEPS=10 EARLY_STOP=0 PLOT=0 NPROC=4"
SAMPLES=""

echo "=== REF: diff_delphes@f777dd5 (worktree) ==="
cd "$WT"
env $COMMON $SAMPLES \
  PYTHONPATH="$WT/src" \
  OUT_BASE="$MAIN/doc/consolidate_tests/counts_ref" \
  EXTRA_ARGS="" \
  bash "$WT/src/parnassus/torch_delphes/full_phasespace_tuning/run_sequential.sh"

echo "=== NEW: consolidate_modes --existence counts ==="
cd "$MAIN"
env $COMMON $SAMPLES \
  OUT_BASE="$MAIN/doc/consolidate_tests/counts_new" \
  EXTRA_ARGS="--existence counts" \
  bash "$MAIN/src/parnassus/torch_delphes/full_phasespace_tuning/run_sequential.sh"
echo "=== PAIR DONE ==="
