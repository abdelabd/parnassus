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
merges -> the sequential closure (`run_sequential.sh`).

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

Training on real full-simulation data. Since the `consolidate_modes` branch
this no longer needs a branch switch: `--mode fullsim` selects the 12-bin
chad-efficiency layout (`eff_binning=ptbins12`) and counts-based existence
terms automatically; add `--existence bce` for the track survival BCE (Hungarian
labels built at load time; the tower BCE stays delphes-only, so the calo count
terms stay on). The legacy
`diff_delphes_runze_cmssinglejet` branch remains as the historical reference.

```bash
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search --root-file /global/cfs/cdirs/m3246/diff_delphes/cms_opendata_zenodo/train_1000.root --optuna-config src/parnassus/torch_delphes/param_configs_fullsim/optuna_config_ptbins.yaml --n-events 100000 --n-steps 100 --n-trials 1 --loss wasserstein_1d --output-base doc/figure_fullsim/1000_frac_pt5 --history-path doc/figure_fullsim/1000_frac_pt5/all_optuna.json --mode fullsim --reco-pt-cut 5 --pid-weighting fraction
```

## 5. Baseline runs (regenerate per code release)

When a new code baseline is established ("release"), the user regenerates the
four reference baselines THEMSELVES with exactly these commands (each on an
interactive GPU node, from the repo root; see §1 for the salloc). Convention:
output dirs as named below; add a `reproduce.sh` copy of the command in each.

```bash
D=src/parnassus/torch_delphes/full_phasespace_tuning

# (a) BCE, calo-joint + live gradients -> doc/pseudo_seq_bce_live
OUT_BASE=doc/pseudo_seq_bce_live \
COMET_NAME_PREFIX=pseudo_bce_live \
EXTRA_ARGS="--existence bce --calo-bce-grads live" \
bash $D/run_sequential.sh $D/stage1_muons.yaml $D/stage2_chads.yaml $D/stage3_calo_joint.yaml $D/stage4_electrons.yaml

# (b) BCE, calo + detach (the champion) -> doc/pseudo_seq_bce_det
OUT_BASE=doc/pseudo_seq_bce_det \
COMET_NAME_PREFIX=pseudo_bce_det \
EXTRA_ARGS="--existence bce" \
bash $D/run_sequential.sh

# (c) counts -> doc/pseudo_seq_counts
OUT_BASE=doc/pseudo_seq_counts \
COMET_NAME_PREFIX=pseudo_counts \
EXTRA_ARGS="--existence counts" \
bash $D/run_sequential.sh

# (d) fullsim -> doc/fullsim_counts
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
  --root-file /global/cfs/cdirs/m3246/diff_delphes/cms_opendata_zenodo/train_1000.root \
  --optuna-config src/parnassus/torch_delphes/param_configs_fullsim/optuna_config_ptbins.yaml \
  --n-events 100000 --n-steps 100 --n-trials 1 --loss wasserstein_1d \
  --output-base doc/fullsim_counts --history-path doc/fullsim_counts/all_optuna.json \
  --mode fullsim --reco-pt-cut 5 --pid-weighting fraction
```

For the delphes closures (a)-(c), the HZZ4l closure PDF is the optional tail
step (on the same allocation):

```bash
python -m parnassus.torch_delphes.full_phasespace_tuning.compare_sample \
  --sample /global/cfs/cdirs/m3246/diff_delphes/allsamples/pseudo_data_100k_param_config_all_HZZ4l_truth_matched_survival.root \
  --fitted-config <OUT_BASE>/fitted_config.yaml \
  --output <OUT_BASE>/distributions_HZZ4l.pdf --device cuda
```

## 6. Reproducibility

If you submit runs yourself, you **must** document the exact commands/configs used for
that run — preferably as a `reproduce.sh` in that run's output directory.

## Efficiency loss (BCE vs counts)

- `--existence {counts,bce}` (delphes mode): the umbrella toggle over the
  existence-term family (CONSOLIDATE_MODES_PLAN.md). `counts` = the legacy
  count-term losses (diff_delphes behavior); `bce` = the validated BCE champion
  (`--eff-loss bce --calo-bce --calo-count-weight 0`; marginal conditioning and
  self-consistent thresholds are hardcoded). It fills defaults only — explicit
  knobs win — and omitting it keeps the legacy per-knob defaults.
- `--eff-loss {counts,bce}`: how the tracking `eff_logits` are fitted. `bce` (the
  default in `--mode delphes`) is the per-particle survival BCE
  (`EFF_LOSS_PLAN.md`) on the PLAIN samples: the survival labels are built in
  memory right before training by Hungarian truth<->reco matching
  (`data._build_survival_labels`, deltaR gate 0.05; ~1 min per 200k dijet events,
  wall time printed as `[hungarian] ...`). No preprocessed sample set is needed.
  `counts` is the legacy expected-count chi^2 and the fullsim-mode default. Knobs:
  `--bce-weight`, `--bce-weighting {pooled,per_species}`, `--matching
  {hungarian,nn}` (the assignment rule behind the labels: one-to-one Hungarian,
  default, or diff_delphes_luigi's reco-claims-nearest-truth; both per charged
  class within the 0.05 gate; pass via `EXTRA_ARGS` in run_sequential.sh).
- One efficiency function: the card forward exports, per track, the survival
  probability it evaluated on the smeared pre-mask kinematics (the
  `TrackSurvivalExport`, keyed by input row); the BCE is taken between that and
  the matched label of the same truth row. Loss and forward cannot disagree, the
  muon > 1 TeV roll-off is fitted exactly, and `rate_raw` gets a gradient there
  (pinned only by the cards).

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
- `test_torch_delphes_learnable.py`, `test_loss_ddp_gather.py` and
  `test_bce_eff_loss.py` run fully green and are the gates for card, DDP-gather,
  and BCE-efficiency-loss (incl. Hungarian matching) work.
- Pre-existing failures elsewhere (verified on the unmodified tree, 2026-09-09,
  unrelated to this work): test_nn (1), test_readers (3), test_optuna_search (5),
  test_param_config (1), test_parnassus (2).
