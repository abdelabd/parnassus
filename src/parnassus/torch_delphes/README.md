# `torch_delphes` — differentiable Delphes in PyTorch

A PyTorch re-implementation of the Delphes CMS detector chain whose response parameters
(tracking efficiencies, momentum resolutions, calorimeter scales and resolutions, …) are
`nn.Parameter`s, so they can be **fitted by gradient descent**.

The package is used as a closure test:

1. **Generate pseudodata** — choose "truth" parameter values, run Pythia8 events through the
   frozen card, save truth particles + reconstructed particle-flow objects.
2. **Regress the parameters** — start a learnable copy of the card at the CMS defaults and fit it
   to the pseudodata.
3. **Plot** — check that the fitted values land on the truth.

## Layout

| Path | What |
|---|---|
| [defaults/CMSDefault.py](defaults/CMSDefault.py) | `CMSEnergyFlowDefault` — the chain: propagator → tracking (smearing + efficiency) → calorimeters → PF merger |
| [learnable.py](learnable.py) | learnable wrappers exposing the 68 card scalars as parameters |
| [param_config.py](param_config.py), [param_configs/](param_configs/) | YAML format for values / trainable masks; shipped truth and search configs |
| [generate_pseudodata.py](generate_pseudodata.py), [processes/](processes/) | Pythia8 → frozen card → ROOT; one `.cmnd` per process |
| [tune_cms_fullsim/](tune_cms_fullsim/) | the fit: data loading, loss, Adam loop, `optuna_search` driver |
| [plotting_scripts/](plotting_scripts/) | parameter-regression and distribution plots |
| [slurm_scripts/](slurm_scripts/) | Perlmutter job arrays for generation and the seed scan |
| [full_phasespace_tuning/](full_phasespace_tuning/) | sequential all-parameter fit (has its own README) |
| [validation/](validation/) | comparison of the torch chain against C++ Delphes |

---

## 1. Sample generation

### Processes

| `--process` | Sample | Constrains |
|---|---|---|
| `muongun` | one J/ψ or Z → μμ per event | muon efficiency + momentum resolution |
| `electrongun` | one J/ψ or Z → ee per event | electron efficiency + resolution |
| `ksgun` | one K_S → π⁺π⁻ per event | charged-hadron efficiency + resolution |
| `dijet` | 13 TeV QCD dijets (p̂_T > 20 GeV) | ECal / HCal scales and resolutions |
| `HZZ4l` | VBF H → ZZ → 4ℓ | independent closure sample |

### Local run

```bash
python -m parnassus.torch_delphes.generate_pseudodata \
    --process muongun --n-events 5000 --n-workers 32 --seed 1 \
    --param-config src/parnassus/torch_delphes/param_configs/param_config_muons.yaml \
    --output /path/to/out.root
```

`--param-config` is the **truth**. It may be partial: listed scalars take the given value, everything
else stays at the CMS default. Pythia8 runs on `--n-workers` CPU processes; the detector pass uses a
GPU if one is visible (`--device`).

Output: a ROOT file with tree `event_tree` and jagged branches `truth_{pt,eta,phi,class,pdgid}` and
`pflow_{pt,eta,phi,class}`; class = 0 charged hadron, 1 electron, 2 muon, 3 neutral hadron, 4 photon.

### Large samples on SLURM

[slurm_scripts/submit_pseudodata.sh](slurm_scripts/submit_pseudodata.sh) fans generation out over a
CPU job array and merges the parts (run from a login node):

```bash
bash src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh \
    --config src/parnassus/torch_delphes/param_configs/param_config_muons.yaml \
    --process muongun --n_events 200000 --n_tasks 10
# -> /global/cfs/cdirs/m3246/diff_delphes/pseudo_data_200k_param_config_muons_muongun.root
```

The merged file is `pseudo_data_<N>k_<config stem>_<process>.root` under `OUTBASE`
(default `/global/cfs/cdirs/m3246/diff_delphes`; `_debug` is appended with `--debug`). `--n_events` must
be a multiple of `--n_tasks`. Env overrides: `OUTBASE`, `N_WORKERS` (32), `TIME_LIMIT` (`00:30:00` per task), `PT_HAT_MIN`.

### The four regression samples

| Key | Process | Truth config | Sample |
|---|---|---|---|
| `muons` | `muongun` | `param_config_muons.yaml` | `pseudo_data_200k_param_config_muons_muongun.root` |
| `electrons` | `electrongun` | `param_config_electrons.yaml` | `pseudo_data_200k_param_config_electrons_electrongun.root` |
| `chads` | `ksgun` | `param_config_chads.yaml` | `pseudo_data_200k_param_config_chads_ksgun.root` |
| `dijets` | `dijet` | `param_config_dijets.yaml` | `pseudo_data_200k_param_config_dijets_dijet.root` |

Each truth config perturbs its block by ±10–50 % from the CMS defaults; the values and defaults are
commented inline in the files.

---

## 2. Parameter regression

### One fit

A fit runs through the Optuna driver with a single trial: `--n-trials 1` runs only the pinned "seed"
trial — the learning rates from `optuna_config_<key>.yaml`, every fitted scalar starting at the CMS
default. Muon example (swap `muons` / `muongun` using the table above for the other guns):

```bash
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
    --root-file /global/cfs/cdirs/m3246/diff_delphes/pseudo_data_200k_param_config_muons_muongun.root \
    --optuna-config src/parnassus/torch_delphes/param_configs/optuna_config_muons.yaml \
    --n-events 200000 --n-steps 100 --n-trials 1 \
    --loss wasserstein_1d --mode delphes \
    --output-base doc/figure_pseudodata_muongun \
    --history-path doc/figure_pseudodata_muongun/all_optuna.json

python -m parnassus.torch_delphes.plotting_scripts.plot_parameter_regression \
    --workspace doc/figure_pseudodata_muongun \
    --truth-config src/parnassus/torch_delphes/param_configs/param_config_muons.yaml
```

- `--mode delphes` — no acceptance cuts, no charged-hadron truncation, photon merger off: the trainee
  must reproduce Delphes as-is. `fullsim` turns those on for fits against CMS full simulation.
- `--loss wasserstein_1d` — per-species 1-D Wasserstein distances of the object kinematics, plus
  object-count and pair-mass response terms.
- `optuna_config_<key>.yaml` — sets which scalars are trainable (`parameters:` mask) and the per-group
  Adam learning rates (`search.lr.<group>.init`).

Outputs (with the paths used above — `all_optuna.json` goes to `--history-path`, `plots/` to the plot
script's `--workspace`):

```
round_0/materialized_config.yaml   start values, trainable mask and lrs of the trial
round_0/history.json               per-epoch train/val loss + parameter snapshots, best_result
round_0/intermediate_plots/        observable overlays every --plot-every epochs
all_optuna.json                    copy of the best trial's history.json
plots/params_reg.pdf               fitted value vs epoch, one page per parameter block, truth dashed
```

Useful knobs: `--seed` (train/val split, shuffle order, smearing noise), `--n-steps`,
`--early-stopping-patience` (10; `<= 0` off), `--pid-weighting {equal,fraction,sqrt_fraction}`,
`--no-pair-mass`. `MCGEN_LOSS_DEBUG=1` prints the per-term loss breakdown.

Multi-GPU: launch the same module with `torchrun --standalone --nproc-per-node=N -m ...` (or
`srun -n N`); the fit is data-parallel and `search.global_batch_size` is split across ranks.
[run_optuna.sh](run_optuna.sh) wraps this for a full TPE study with a persistent sqlite storage.

### Seed scan on SLURM

[slurm_scripts/submit_param_regression.sh](slurm_scripts/submit_param_regression.sh) repeats the fit
above for seeds 0–4, one shared A100 per seed (run from a login node):

```bash
for g in muons electrons chads dijets; do
    bash src/parnassus/torch_delphes/slurm_scripts/submit_param_regression.sh --GUN $g
done
# -> /global/cfs/cdirs/m3246/diff_delphes/results/<process>_<seed>/{round_0/, all_optuna.json, plots/params_reg.pdf}
```

Env overrides: `SEEDS` (`0-4`, sbatch array spec — `SEEDS=3` reruns one seed), `TIME_LIMIT`
(`01:00:00`; use `02:00:00` for dijet), `RESULTS`, `N_EVENTS`, `N_STEPS`. `history.json` is written
only when a fit ends, so a task killed by the time limit loses its seed.

### Plots

Summary over seeds — one page per gun, one row per parameter, mean ± std of (fit − truth)/truth; the
grey block at the bottom holds parameters that are pinned or expected to be unconstrained:

```bash
python -m parnassus.torch_delphes.plotting_scripts.plot_parameter_summary \
    --results /global/cfs/cdirs/m3246/diff_delphes/results \
    --output doc/figure_pseudodata_summary/params_summary.pdf      # --xlim 0.5
```

Paper-style example (muon p_T scales vs epoch + loss curve):

```bash
python -m parnassus.torch_delphes.plotting_scripts.plot_example_regression \
    --workspace /global/cfs/cdirs/m3246/diff_delphes/results/muongun_0 \
    --truth-config src/parnassus/torch_delphes/param_configs/param_config_muons.yaml
# -> <workspace>/plots/example_reg.pdf
```

Distributions of one fit (target vs initial vs fitted per species and observable, plus pair-mass
response):

```bash
python -m parnassus.torch_delphes.plotting_scripts.plot_distribution \
    --workspace doc/figure_pseudodata_muongun --sample <the pseudodata ROOT file>
# -> <workspace>/plots/distributions.pdf
```

### What to expect

Scales, efficiencies and the `a` resolution terms recover; the `b` terms only partially (weak lever)
and the calorimeter `c_E` terms not at all — the summary draws both in the grey block, together with
the muon `rate_raw` / `eff_logits[2,5]` (> 1 TeV bins), which these samples cannot constrain and
which stay pinned.

---

## 3. Full phase-space tuning (sequential)

Recovers every constrained block of the card from the `param_config_all` samples one block at a time
(muons → charged hadrons → calorimeters → electrons), each stage starting from the previous stage's
result. Details in [full_phasespace_tuning/README.md](full_phasespace_tuning/README.md).

```bash
OUT_BASE=doc/figure_sequential \
    bash src/parnassus/torch_delphes/full_phasespace_tuning/run_sequential.sh   # NPROC=1 for one GPU
# -> doc/figure_sequential/<stage>/{round_0/, plots/params_reg.pdf}, doc/figure_sequential/fitted_config.yaml

python -m parnassus.torch_delphes.full_phasespace_tuning.compare_sample \
    --sample /global/cfs/cdirs/m3246/diff_delphes/pseudo_data_100k_param_config_all_HZZ4l.root \
    --fitted-config doc/figure_sequential/fitted_config.yaml
# -> doc/figure_sequential/distributions_<sample stem>.pdf
```

`NPROC` = GPUs per stage (default 4, `torchrun` when > 1; the sub-README notes the dijet stage needs
≥ 2 to fit in memory). Other knobs: `N_STEPS`, `N_EVENTS`, `SAMPLE_DIR`. To resume at a later stage,
set `FROM_HISTORY` to the previous stage's `history.json` and pass the remaining stage YAMLs as
arguments.

---

## 4. Manual single fit

`python -m parnassus.torch_delphes.tune_cms_fullsim --param-config <yaml> --root-file <root>
--history-path <json>` runs one Adam fit from a **full** param config (every card scalar listed with a
`value`; `trainable` / `lr_scale` optional; effective lr = `--lr × lr_scale`, `--lr` default `1e-2`).
`python -m parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results --history <json> --root-file <root>
--truth-config <yaml>` plots it (loss curve, parameter drift, observable overlays). The regression
workflow above is the usual route.

---

## Param-config reference

```yaml
MuonMomentumSmearing.resolution_module.scale_raw[0]:
  value: 1.15        # physical value (after the transform below)
  trainable: true    # default false
  lr_scale: 1.0      # optional; default 1.0 (scales, efficiencies, fractions, rate_raw) or 0.1 (a/b, calo resolution)
```

Keys are the card's `named_parameters()` names; `[i]` selects a vector element — |η| regions for the
track `a_raw` / `b_raw` / `scale_raw` (0: < 0.5, 1: 0.5–1.5, 2: 1.5–2.5), the ECal scale (0 barrel,
1 endcap, 2 forward) and the HCal scale (0 central, 1 forward); efficiency bins as laid out in
`learnable.py`.
[param_configs/cms_target_default.yaml](param_configs/cms_target_default.yaml) lists all 68 scalars
at the CMS defaults.

| Kind | Name | raw → physical | Allowed |
|---|---|---|---|
| scale | `scale_raw` | `1 + 0.3·tanh(raw)` | open (0.7, 1.3); a boundary value is rejected by the loader |
| efficiency / fraction | `eff_logits`, `*_logit` | `lo + (hi − lo)·sigmoid(raw)` | (0, 1); `k0s_logit` (0.1, 0.5), `photon_logit` (0.8, 1.0), `k0l_logit` (0, 0.4) |
| resolution / rate | `a_raw`, `b_raw`, `rate_raw`, `resolution_func.*` | `softplus(raw)` | > 0 |

Two config flavours:

- **Truth** (`param_config_<key>.yaml`) — partial, only the perturbed scalars. Used by
  `generate_pseudodata --param-config` and by every `--truth-config`.
- **Search** (`optuna_config_<key>.yaml`) — `search:` (lr ranges with `init`, `global_batch_size`),
  `constants:` (values fixed per trial or sampled by TPE) and `parameters:` (`{key: {trainable: bool}}`,
  unlisted = trainable). Fitted scalars start at the CMS default.

## Gotchas

- Pass the **truth** config to `--truth-config`, never a fit config.
- `--mode` must match the sample: pseudodata from this package → `delphes`; CMS full simulation →
  `fullsim`. An Optuna study is locked to one mode — use a new `--study-name` to switch.
- `--seed` also chooses the train/val split, so seeds are not comparable at the event level.
- On compute nodes run the tests with `env python -m pytest` (avoids a uv cache lock error).
