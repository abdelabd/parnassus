#!/bin/bash
# CONSOLIDATE_MODES_PLAN.md 2c — fullsim parity: consolidate_modes --mode fullsim
# vs diff_delphes_runze_cmssinglejet (4650b39), the user's reference command
# shortened to 50k events / 10 steps, world_size=1 (single GPU) so the DDP
# fixes consolidate_modes carries cannot bite.
set -euo pipefail
export OMP_NUM_THREADS=8
MAIN=/pscratch/sd/a/aelabd/parnassus
WT=/pscratch/sd/a/aelabd/worktrees/cmssinglejet_ref
ARGS="--root-file /global/cfs/cdirs/m3246/diff_delphes/cms_opendata_zenodo/train_1000.root \
  --n-events 50000 --n-steps 10 --n-trials 1 --loss wasserstein_1d \
  --mode fullsim --reco-pt-cut 5 --pid-weighting fraction"

echo "=== REF: cmssinglejet@4650b39 (worktree) ==="
cd "$WT"
PYTHONPATH="$WT/src" python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
  --optuna-config "$WT/src/parnassus/torch_delphes/param_configs_fullsim/optuna_config_ptbins.yaml" \
  --output-base "$MAIN/doc/consolidate_tests/fullsim_ref" \
  --history-path "$MAIN/doc/consolidate_tests/fullsim_ref/all_optuna.json" \
  $ARGS

echo "=== NEW: consolidate_modes --mode fullsim ==="
cd "$MAIN"
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
  --optuna-config "$MAIN/src/parnassus/torch_delphes/param_configs_fullsim/optuna_config_ptbins.yaml" \
  --output-base "$MAIN/doc/consolidate_tests/fullsim_new" \
  --history-path "$MAIN/doc/consolidate_tests/fullsim_new/all_optuna.json" \
  $ARGS
echo "=== PAIR DONE ==="
