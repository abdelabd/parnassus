# torch_delphes — working notes

## 1. Interactive allocations (preferred for testing/debugging)

We like to test and debug in interactive mode. Request an allocation and set up the
environment, then run commands inside it with `srun`.


1x4 (single node, 4 GPUs):

```bash
salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node=4 --gpus-per-node=4 -A m3246_g
cd parnassus
source parnassus_env/bin/activate
export COMET_API_KEY="dP7SQEk285l0DsZvRZgPM4cR7"
export COMET_WORKSPACE="abdelabd"
```

1x1 (single node, single GPU):

```bash
salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node=1 --gpus-per-node=1 -A m3246_g
cd parnassus
source parnassus_env/bin/activate
export COMET_API_KEY="dP7SQEk285l0DsZvRZgPM4cR7"
export COMET_WORKSPACE="abdelabd"
```

More GPUs is typically better. **But** changes to the loss function sometimes break
multi-GPU or multi-node training (or both) — when debugging a loss change, verify at
1x1 first, then 1x4.

**NN inference belongs on a GPU**: if salloc resources are available (i.e. the
user's 2 interactive GPU allocations aren't already in use), run the HZZ4l eval
(`compare_sample.py --device cuda`, or its auto-detect) — and in general ANY
neural-network inference — on CUDA/GPU rather than CPU. A 100k-event HZZ4l
`compare_sample` pass is ~40 min on login CPUs vs ~a minute on an A100; tack such
evals onto the tail of an existing allocation when one is open.

**Pipeline whenever possible**: when a run depends on another job's output
(generation -> merge -> preprocessing -> training), submit the whole chain up front
with `sbatch --dependency=afterok:<jid>[:<jid>...]` instead of waiting and
submitting by hand — queue wait times overlap, nothing sits idle overnight, and a
failed upstream job cleanly holds its dependents. Example chain: sample-generation
merges -> `hungarian_match_samples.sbatch` -> `run_sequential_survival.sbatch`.

Allocation tips:

- Add `--no-shell` to `salloc` to allocate without blocking, then drive the job with
  `srun --jobid=<id> ...`.
- The `interactive` QOS caps concurrent jobs per user (~2); plan around that.
- Login nodes have one shared A100 and no SLURM env — real runs need an allocation.
- `/tmp` is node-local on NERSC. Anything a compute node must read (worktrees,
  configs, outputs) belongs on `/pscratch`.
- Keep one Optuna study per node at `world_size=4`. `world_size` changes DDP batch
  semantics and therefore the loss, so studies at different world sizes are not
  comparable. Use extra nodes for parallel studies/seeds, not one bigger study.

## 2. Full closure training

The full "closure" training (testing on pseudodata so we can check parameter
convergence):

```bash
bash src/parnassus/torch_delphes/full_phasespace_tuning/run_sequential.sh
```

- GPU usage: `NPROC` (default 4) sets the GPU count; `>1` launches
  `torchrun --standalone`, which is **single-node only** — a multi-node allocation
  does not speed up one sequential run. Per-rank batch is `4096/NPROC`.
- Runtime: on a single GPU, roughly 1–2 hours total (~30 minutes per stage); on a
  1x4 node, roughly 1–1.5 hours total (~20 minutes per stage). Don't bother with 4x4:
  the speedup over 1x4 has been negligible, and `torchrun --standalone` can't use the
  extra nodes anyway.
- Caveat for `NPROC=1`: the dijet (calo) stage needs >= 2 ranks to fit in memory
  (per the README), so a pure single-GPU run may OOM at stage 3.
- Plots: the script only produces **per-stage** figures (`params_reg.pdf` +
  `plot_fit_results` observables under `<stage>/plots/`) and the final
  `fitted_config.yaml` card. The overall closure plot is a separate manual step —
  run `compare_sample.py` on an independent sample against the final card
  (CPU-only, ~20 s per 5k gun events; one PDF: target vs initial vs tuned,
  per-species kinematics, pair-mass responses, m_ee / m_mumu / m_4l):

```bash
python -m parnassus.torch_delphes.full_phasespace_tuning.compare_sample \
    --sample /global/cfs/cdirs/m3246/diff_delphes/allsamples/pseudo_data_100k_param_config_all_HZZ4l.root \
    --fitted-config doc/figure_sequential/fitted_config.yaml
```
- Resume from a later stage by listing only the remaining stage YAMLs and pointing
  `FROM_HISTORY` at the previous stage's `history.json`, e.g.:

```bash
D=src/parnassus/torch_delphes/full_phasespace_tuning
FROM_HISTORY=doc/figure_sequential/stage2_chads/round_0/history.json \
    bash $D/run_sequential.sh $D/stage3_calo.yaml $D/stage4_electrons.yaml
```

## 3. Smaller training runs

Muons-only (the first step of the sequential tuning):

```bash
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search --root-file /global/cfs/cdirs/m3246/diff_delphes/pseudo_data_200k_param_config_muons_muongun.root --optuna-config src/parnassus/torch_delphes/param_configs/optuna_config_muons.yaml --n-events 200000 --n-steps 100 --n-trials 1 --loss wasserstein_1d --output-base doc/figure_pseudodata_muongun --history-path doc/figure_pseudodata_muongun/all_optuna.json --mode delphes
```

Charged-hadrons (kaon gun):

```bash
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search --root-file /global/cfs/cdirs/m3246/diff_delphes/pseudo_data_200k_param_config_chads_ksgun.root --optuna-config src/parnassus/torch_delphes/param_configs/optuna_config_chads.yaml --n-events 200000 --n-steps 100 --n-trials 1 --loss wasserstein_1d --output-base doc/figure_pseudodata_k0sgun --history-path doc/figure_pseudodata_k0sgun/all_optuna.json --mode delphes
```

## 4. Full "fullsim" training

Training on real full-simulation data. **Requires switching to the
`diff_delphes_runze_cmssinglejet` branch.**

```bash
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search --root-file /global/cfs/cdirs/m3246/diff_delphes/cms_opendata_zenodo/train_1000.root --optuna-config src/parnassus/torch_delphes/param_configs_fullsim/optuna_config_ptbins.yaml --n-events 100000 --n-steps 100 --n-trials 1 --loss wasserstein_1d --output-base doc/figure_fullsim/1000_frac_pt5 --history-path doc/figure_fullsim/1000_frac_pt5/all_optuna.json --mode fullsim --reco-pt-cut 5 --pid-weighting fraction
```

## 5. Reproducibility

If you submit runs yourself, you **must** document the exact commands/configs used for
that run — preferably as a `reproduce.sh` in that run's output directory.

## Efficiency loss (BCE vs counts)

- `--eff-loss {counts,bce}`: how the tracking `eff_logits` are fitted. `bce` (the
  default in `--mode delphes`) is the per-particle survival BCE
  (`EFF_LOSS_PLAN.md`); it needs samples with the survival-label branches — the
  `*_truth_matched_survival.root` set (and the matcher-labeled
  `*_hungarian_matched_survival.root` set) next to the originals. `counts` is the
  legacy expected-count chi^2 and the fullsim-mode default. Knobs: `--bce-weight`,
  `--bce-weighting {pooled,per_species}`.
- Old samples without labels + `--eff-loss bce` = a hard error telling you to
  regenerate (`slurm_scripts/submit_truth_matched_survival_samples.sh`).

## Run outputs

- Outputs land wherever `--output-base` / `--history-path` / `OUT_BASE` point in the
  commands above (`doc/figure_*` by convention).
- For an example of reading `all_optuna.json`, see
  `parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results`.

## Environment

- Always `source parnassus_env/bin/activate`; never invoke the venv python by absolute
  path.
- The editable install is a plain `.pth` file pointing at the main checkout's `src`
  (no import hook), so `PYTHONPATH=<worktree>/src` correctly shadows it when running
  from a git worktree. Assert on `parnassus.__file__` to confirm which checkout is live.

## Tests

- `pytest src/parnassus/tests/test_tune_cms_fullsim.py` is fully green (47/47)
  since 2026-09-10: the pseudodata fixture was regenerated with `truth_pdgid` and
  the survival-label branches. Treat any failure there as a real regression.
- `test_torch_delphes_learnable.py`, `test_loss_ddp_gather.py`,
  `test_survival_labels.py` and `test_bce_eff_loss.py` run fully green and are the
  gates for card, DDP-gather, and BCE-efficiency-loss work.
- Pre-existing failures elsewhere (verified on the unmodified tree, 2026-09-09,
  unrelated to this work): test_nn (1), test_readers (3), test_optuna_search (5),
  test_param_config (1), test_parnassus (2).
