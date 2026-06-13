# `torch_delphes` — Differentiable Detector Tuning

This package is a **differentiable detector-tuning harness**. The idea:

1. Pick a set of "true" detector-response parameters and use them to generate CMS-like
   **pseudodata** (truth particles + the reconstructed particle-flow objects they would produce).
2. Hand a *learnable* copy of the same detector model a config whose parameters are
   **deliberately off-truth**, and fit it back to the pseudodata.
3. Check that the optimizer **recovers the true parameters** and that the predicted observables
   match the target.

Because every step of the detector simulation is implemented in PyTorch, the whole chain
(truth → detector response → observables → loss) is differentiable, so the detector parameters
can be tuned with Adam.

The user-facing workflow is three commands — **generate → tune → plot** — described below.

> **Running the commands.** All entry points are Python modules and are meant to be run with
> [`uv`](https://docs.astral.sh/uv/), so the canonical form is
> `uv run python -m parnassus.torch_delphes.<entrypoint> ...`, executed from the repository root
> (`torch_delphes/torch_delphes`). If you have an activated environment (e.g. via a private
> `setup.sh`) you can drop the `uv run` prefix and just call `python -m ...`. All relative paths
> in the examples (`src/...`, `doc/...`) are **relative to the repo root**.

---

## Quickstart

Run these three commands in order, from the repo root:

```bash
# 1. Generate 5000 events of CMS pseudodata (truth response = cms_target_default.yaml)
uv run python -m parnassus.torch_delphes.generate_pseudodata \
    --output src/parnassus/tests/benchmark_data/cms_pseudodata.root \
    --n-events 5000 \
    --n-workers 32 \
    --pt-hat-min 100 \
    --seed 1

# 2. Fit a learnable card whose charged-hadron scale starts off-truth, recovering the truth
uv run python -m parnassus.torch_delphes.tune_cms_fullsim \
    --root-file src/parnassus/tests/benchmark_data/cms_pseudodata.root \
    --n-events -1 \
    --n-steps 120 \
    --lr 1e-2 \
    --history-path doc/fit_results/all66_history.json \
    --param-config src/parnassus/torch_delphes/param_configs/debug_train_chad_scale_barrel.yaml

# 3. Produce validation plots (loss curve, parameter drift vs truth, observable overlays)
uv run python -m parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results \
    --history doc/fit_results/all66_history.json \
    --root-file src/parnassus/tests/benchmark_data/cms_pseudodata.root \
    --output-dir doc/figures \
    --truth-config src/parnassus/torch_delphes/param_configs/cms_target_default.yaml
```

---

## Architecture at a glance

The same detector model — `CMSEnergyFlowDefault` (from [defaults/](defaults/)) — plays two roles:

- **Target card** (used in generation): loaded with the *truth* config, **frozen** (`requires_grad=False`,
  eval mode). It defines the ground-truth detector response that maps truth particles to the
  `pflow_*` branches of the pseudodata.
- **Trainee card** (used in tuning): loaded with a config whose `value`s start off-truth and whose
  `trainable` parameters are optimized. The fit pushes the trainee's parameters toward the truth.

A single **param-config YAML** drives both cards (which parameters exist, their physical values,
which are trainable, and their per-parameter learning rate). The data flows:

```
truth particles ──▶ detector card ──▶ EFlowObjects ──▶ observables (pt, eta, log_E, HT, …)
                                                              │
                  target observables (from ROOT) ◀───────────┴──▶ sliced-Wasserstein loss ──▶ Adam
```

Key files in this directory:

| File / dir | Role |
|---|---|
| [generate_pseudodata.py](generate_pseudodata.py) | Step 1 — Pythia8 + frozen target card → pseudodata ROOT file |
| [tune_cms_fullsim/](tune_cms_fullsim/) | Step 2 — fit the trainee card to the pseudodata |
| [tune_cms_fullsim/cli.py](tune_cms_fullsim/cli.py) | tuning entry point (`python -m parnassus.torch_delphes.tune_cms_fullsim`) |
| [tune_cms_fullsim/training.py](tune_cms_fullsim/training.py) | Adam loop, scheduler, early stopping |
| [tune_cms_fullsim/loss.py](tune_cms_fullsim/loss.py) | per-event sliced-Wasserstein loss (+ soft-hist diagnostic) |
| [tune_cms_fullsim/data.py](tune_cms_fullsim/data.py) | ROOT loading, padding, train/val split, observable extraction |
| [tune_cms_fullsim/plot_fit_results.py](tune_cms_fullsim/plot_fit_results.py) | Step 3 — validation plots |
| [param_config.py](param_config.py) | YAML loader, raw↔physical transforms, `select_trainable` |
| [param_configs/](param_configs/) | shipped configs + [make_default_configs.py](param_configs/make_default_configs.py) |
| [defaults/](defaults/) | the `CMSEnergyFlowDefault` detector card definition |

---

## Step 1 — Generate pseudodata

`uv run python -m parnassus.torch_delphes.generate_pseudodata`
([generate_pseudodata.py](generate_pseudodata.py))

Generates Pythia8 13 TeV hard-QCD dijet events, turns the stable truth particles into the
class-based representation the trainee will see (via `load_truth_events`, which keeps the
target-card input identical to the trainee input — the "truth-input fidelity" fix), passes them
through the **frozen target card**, and writes both the truth particles and the reconstructed
particle-flow objects to a ROOT file. All `--n-events` events are generated up front in one
parallel pass across `--n-workers` Pythia8 CPU processes into a single HepMC3 file, then streamed
in `--batch-size` chunks through the target card before the ROOT file is written in one go.

### Flags

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--output` | path | `src/parnassus/tests/benchmark_data/cms_pseudodata.root` | Output ROOT file. |
| `--n-events` | int | `20000` | Exact number of events to generate. |
| `--n-workers` | int | `None` (all CPU cores, capped at `--n-events`) | Parallel Pythia8 **CPU** processes for event generation. Independent of `--device`. |
| `--pt-hat-min` | float | `100.0` | Pythia8 `PhaseSpace:pTHatMin` (GeV) — minimum hard-scatter pT. Lower → softer, higher-multiplicity events. |
| `--batch-size` | int | `512` | Events per TorchDelphes forward pass in phase 2 (memory knob; does not change the output). |
| `--seed` | int | `1` | **Torch** RNG seed for the target card's stochastic smearing / Gumbel-ST in phase 2. (Pythia per-worker seeds are assigned internally by `HepMC3Generator`.) |
| `--device` | str | `None` (auto: `cuda` if a GPU is available, else `cpu`) | Device for the **phase-2 TorchDelphes forward pass only**. Pythia generation (phase 1) is always CPU-parallel. |
| `--cmnd-file` | path | shipped `qcd_dijet.cmnd` | Base Pythia `.cmnd` configuration; `--pt-hat-min` is appended as an override. |
| `--work-dir` | path | `None` (temp dir, auto-removed) | Scratch directory for intermediate HepMC files / per-job logs. |
| `--keep-hepmc` | flag | `False` | Keep the intermediate HepMC files instead of deleting the auto-created temp dir (no effect with `--work-dir`). |
| `--param-config` | path | `param_configs/cms_target_default.yaml` | YAML whose physical `value` fields define the **ground-truth** detector response written into the `pflow_*` branches. This is the truth you later try to recover. |
| `--debug` | flag | `False` | Also write ~150 intermediate per-module branches (`<ModuleName>.<Var>`); large files, leave off for plain training. |

### Output

A ROOT file with a TTree named `event_tree` and 8 jagged (per-event, variable-length) branches:

- **Truth (generator-level stable particles):** `truth_pt`, `truth_eta`, `truth_phi`, `truth_class`
- **PFlow (reconstructed targets):** `pflow_pt`, `pflow_eta`, `pflow_phi`, `pflow_class`

The `*_class` branches use the 5-class encoding:

| class | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| particle | charged hadron | electron | muon | neutral hadron | photon |

### Batch generation on SLURM (large samples)

A single run takes ~13 min for 10k events, so a large training sample is built by fanning
generation out across a SLURM job array (NERSC Perlmutter, project `m3246`) and merging the
per-seed files. Two scripts in [slurm_scripts/](slurm_scripts/) do this:

| Script | Role |
|---|---|
| [slurm_scripts/submit_pseudodata.sh](slurm_scripts/submit_pseudodata.sh) | Login-node driver: submits an `NJOBS`-task array (seed = array index) on the **CPU** partition, each generating `N_EVENTS_PER_JOB` events, then submits the merge job with `--dependency=afterok`. |
| [slurm_scripts/merge_pseudodata.sbatch](slurm_scripts/merge_pseudodata.sbatch) | The merge job: uproot-concatenates the per-seed files into one, validates the total event count, then deletes the parts. |

Run from a login node (defaults give 10 × 10000 = 100k events):

```bash
bash src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh
```

Everything is configurable via environment variables (shown with defaults):

| Var | Default | Meaning |
|---|---|---|
| `OUTBASE` | `/global/cfs/cdirs/m3246/Runze/MCGen/data` | Output base. Parts go in `$OUTBASE/parts/`, logs in `$OUTBASE/logs/`, merged file in `$OUTBASE/`. **Must be a shared filesystem** (CFS / scratch), not node-local `/tmp`. |
| `NJOBS` | `10` | Number of array tasks (= number of seeds, `1..NJOBS`). |
| `N_EVENTS_PER_JOB` | `10000` | Events per task (total = `NJOBS × N_EVENTS_PER_JOB`). |
| `N_WORKERS` | `32` | Pythia CPU workers per task (matches the job's `-c 32`). |
| `PT_HAT_MIN` | `100` | Pythia `PhaseSpace:pTHatMin`. |
| `MERGED_NAME` | `cms_pseudodata_100k.root` | Final merged filename under `$OUTBASE`. |

The jobs run on the **CPU partition** with `--device cpu`: the GPU is used for only ~12s (the
TorchDelphes forward) of a ~13-min job, so a GPU node would sit idle. The driver runs `setup.sh`
once up front so the array tasks activate the prebuilt uv env directly instead of each firing a
concurrent `uv sync`.

> **Merge is concatenation-safe.** The ROOT files carry no persisted event-number branch — event
> identity is the TTree entry index — so appending entries yields globally unique indices with no
> duplication. The merge verifies the merged entry count equals the sum of the parts before
> deleting them (so a bad merge leaves the parts intact).

> **Failed tasks.** The `afterok` dependency runs the merge only if **all** tasks succeed. If one
> fails, rerun that seed (`sbatch --array=<i> ...`) and then submit the merge by hand
> (`sbatch slurm_scripts/merge_pseudodata.sbatch`), or switch the dependency to `afterany` to merge
> whatever did succeed.

Cheap dry run (2 jobs × 100 events on scratch):

```bash
N_EVENTS_PER_JOB=100 NJOBS=2 OUTBASE=$SCRATCH/mcgen_dryrun \
    bash src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh
```

---

## Step 2 — Tune (fit the trainee card)

`uv run python -m parnassus.torch_delphes.tune_cms_fullsim`
([tune_cms_fullsim/cli.py](tune_cms_fullsim/cli.py))

Loads the pseudodata, builds a trainee `CMSEnergyFlowDefault` initialized from `--param-config`,
selects the trainable parameters, and runs Adam to minimize the distance between the trainee's
predicted observables and the full-sim targets. Uses a `ReduceLROnPlateau` scheduler
(halves the LR after 4 stale epochs) and early stopping (patience 10). DDP-aware: under SLURM
`srun` it shards across ranks, and only rank 0 logs, plots, and writes history.

### Flags

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--param-config` | path | **required** | YAML config (see [Param-config reference](#param-config-reference)). Its `value`s **initialize** every parameter, `trainable` selects the optimized subset (per element), and `lr_scale` sets each parameter's effective LR. |
| `--root-file` | path | **required** | Full-sim pseudodata ROOT file (Step 1 output). Must exist; the script errors if it is missing (no synthetic fallback). |
| `--n-events` | int | `-1` | Number of events to load. **`-1` = all events** in the tree. |
| `--n-steps` | int | `200` | Number of Adam steps (epochs); each step is one pass over the training data. |
| `--lr` | float | `1e-2` | **Global** learning-rate magnitude. The *effective* Adam LR of each parameter is `--lr × lr_scale` (see below). |
| `--seed` | int | `0` | Torch RNG seed and train/validation split seed. |
| `--history-path` | path | `None` | If set, write the full training history (loss trajectory + per-epoch parameter snapshots) to this JSON. **Required input for Step 3.** |
| `--intermediate-plot-dir` | str | `doc/figures/intermediate_plots` | Directory for per-epoch diagnostic PDFs (`intermediate_epoch_<step>.pdf`), one observable per page, overlaying target / epoch-0 / current prediction. Pass `""` to disable. |
| `--plot-every` | int | `1` | Save intermediate plots every N epochs (`1` = every epoch). The final / early-stopped epoch is always plotted. |

### The `--lr × lr_scale` rule

`--lr` is a single global knob; each parameter's real Adam learning rate is `--lr` multiplied by
that parameter's `lr_scale` from the YAML. `select_trainable` ([param_config.py](param_config.py))
groups parameters by their distinct effective LR and builds one Adam param-group per group.

**Worked example** — with `--lr 1e-2`:

| Parameter | `lr_scale` in YAML | Effective Adam LR |
|---|---|---|
| `...scale_raw[0]` (a momentum/energy scale) | `1.0` | `1e-2` |
| `...resolution_module.a_raw[0]` (a resolution coeff) | `0.1` | `1e-3` |

Resolution coefficients default to `lr_scale 0.1` because they pass through a `softplus`, which is
far more step-sensitive than the `tanh`/`sigmoid` used for scales and efficiencies.

Partially-trainable vectors (some elements trainable, some frozen) are handled with a gradient mask
hook so frozen elements receive zero gradient.

### History JSON

When `--history-path` is set, the written JSON has three top-level keys:

- `metadata` — run-level scalars: `n_events`, `n_steps`, `lr`, `param_config`, the distinct
  `param_group_lrs`, the list of `trainable_params`, and `world_size`.
- `history` — one entry per epoch (`epoch_<step>`) with `step`, `train_loss`, `val_loss`, and a
  `parameters` snapshot (physical, post-transform values keyed `name[i]`).
- `best_result` — the epoch with the minimum validation loss, including its final parameter values.

---

## Step 3 — Validation plots

`uv run python -m parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results`
([tune_cms_fullsim/plot_fit_results.py](tune_cms_fullsim/plot_fit_results.py))

Reads the history JSON and the pseudodata, restores the trainee card at its initial and best-epoch
parameters, and writes a set of validation PDFs.

### Flags

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--history` | path | **required** | The history JSON written by Step 2's `--history-path`. |
| `--root-file` | path | `src/parnassus/tests/benchmark_data/cms_pseudodata.root` | Full-sim pseudodata for the observable overlays. |
| `--output-dir` | path | `doc/figures` | Directory for the output PDFs (created if missing). |
| `--n-events-for-plots` | int | `400` | How many events to load for the observable histograms. |
| `--seed` | int | `0` | RNG seed for evaluating the (stochastic) trainee card. |
| `--truth-config` | path | `param_configs/cms_target_default.yaml` | The **generation/truth** config whose physical `value`s are drawn as the truth reference lines on the parameter-drift plots. **Must be the generation config, not a training config** — a trained parameter's `value` in a training config is its off-truth *start*, not its truth. The script prints a warning if the config you pass has any `trainable: true` entries. |

### Outputs (PDFs in `--output-dir`)

| File | Shows |
|---|---|
| `loss_trajectory.pdf` | Train (blue) and validation (orange) loss vs step, log-y, with the best epoch marked. |
| `param_drift_scales.pdf` | Trajectories of the scale parameters (charged-hadron pT scale, ECal scale, HCal scale) vs the truth reference line. |
| `param_drift_other.pdf` | Trajectories of representative non-scale parameters (a resolution coeff, a tracking-efficiency logit, the K0S hadron fraction) vs truth. |
| `observable_pt.pdf` | PF-object pT distribution: target / trainee-at-init / trainee-at-best. |
| `observable_eta.pdf` | PF-object η distribution (target / init / best). |
| `observable_ht.pdf` | Per-event scalar HT (target / init / best). |
| `observable_log_ht.pdf` | log(HT) — the form actually used in the loss. |
| `observable_multiplicity.pdf` | PF objects per event (target / init / best). |

---

## Param-config reference

A param-config is a flat YAML mapping, one entry per scalar:

```yaml
ChargedHadronMomentumSmearing.resolution_module.scale_raw[0]:
  value: 1.25        # PHYSICAL (post-transform) value
  trainable: false   # whether Adam optimizes this scalar
  lr_scale: 1.0      # per-parameter LR multiplier (effective LR = --lr * lr_scale)
```

- **Key** — matches the model's `named_parameters()` name, with a `[i]` suffix for vector elements
  (e.g. detector regions: index 0 = barrel, 1 = endcap, 2 = forward).
- **`value`** — the *physical* number (after the transform below), not the raw parameter.
- **`trainable`** — per-element; `select_trainable` enables gradients only for `true` entries.
- **`lr_scale`** — per-parameter learning-rate multiplier.

### Raw ↔ physical transforms

The optimizer works on unconstrained "raw" parameters; the config `value` is the physical value
after the transform. Defined in [param_config.py](param_config.py):

| Parameter kind | Suffix | Transform (raw → physical) | Range |
|---|---|---|---|
| Scale | `.scale_raw` | `1 + 0.3·tanh(raw)` | open `(0.7, 1.3)` |
| Efficiency / fraction | `.eff_logits`, `_logit` | `sigmoid(raw)` | `(0, 1)` |
| Resolution / rate | `.a_raw`, `.b_raw`, `.rate_raw`, calo `resolution_func` | `softplus(raw)` | `> 0` |

> ⚠️ **Scale NaN caveat.** A scale `value` must stay strictly inside `(0.7, 1.3)`. The exact
> boundary maps to `atanh(±1) = ±∞`, which produces NaN. Keep trainable scale starts comfortably
> inside the interval (e.g. `0.71`, not `0.70`).

### Parameter groups

Parameters cover, by particle type and detector region: tracking efficiencies (`eff_logits`,
`rate_raw`), momentum smearing per species (`a_raw`/`b_raw`/`scale_raw`), hadron fractions
(`chad_logit`, `k0s_logit`, `lambda_logit`), and ECal/HCal energy scales and resolution functions.

### Truth vs debug config

The two shipped configs differ in exactly one parameter block —
`ChargedHadronMomentumSmearing.resolution_module.scale_raw[0..2]`:

| Config | `value` | `trainable` |
|---|---|---|
| [param_configs/cms_target_default.yaml](param_configs/cms_target_default.yaml) (truth) | `1.25` | `false` |
| [param_configs/debug_train_chad_scale_barrel.yaml](param_configs/debug_train_chad_scale_barrel.yaml) (fit) | `0.71` | `true` |

So the debug config starts the charged-hadron momentum scale at `0.71` (off-truth, but inside the
valid interval) and asks the fit to recover the true `1.25`. To create or regenerate configs
programmatically, see [param_configs/make_default_configs.py](param_configs/make_default_configs.py).

---

## Outputs & paths

| Artifact | Default location | Produced by |
|---|---|---|
| Pseudodata ROOT | `src/parnassus/tests/benchmark_data/cms_pseudodata.root` | Step 1 (`--output`) |
| Large pseudodata ROOT (SLURM) | `/global/cfs/cdirs/m3246/Runze/MCGen/data/cms_pseudodata_100k.root` | Step 1 batch (`slurm_scripts/`) |
| Training history JSON | (set via `--history-path`, e.g. `doc/fit_results/all66_history.json`) | Step 2 |
| Per-epoch diagnostic PDFs | `doc/figures/intermediate_plots/intermediate_epoch_<step>.pdf` | Step 2 (`--intermediate-plot-dir`) |
| Validation PDFs | `doc/figures/*.pdf` | Step 3 (`--output-dir`) |

---

## Notes & gotchas

- **Run from the repo root.** All the relative paths above (`src/...`, `doc/...`) are resolved
  relative to the working directory; run the commands from `torch_delphes/torch_delphes`.
- **`uv run` vs activated env.** Commands are shown with `uv run`; if you have an environment
  activated (e.g. a private `setup.sh`) you can drop the prefix.
- **Use the *generation* config for `--truth-config` in Step 3** — passing a training config makes
  the truth reference lines wrong (they'd show the off-truth start value). The plotter warns you.
- **Scale parameters must stay inside `(0.7, 1.3)`** or the `tanh` transform produces NaN.
- **DDP:** under `srun` only rank 0 writes logs, plots, and the history JSON.
