# Parnassus-P

## Installation

Requirements: bash, git and curl. `uv` and a matching Python (3.12) are fetched
automatically if missing.

```bash
git clone <this repo> && cd parnassus
source setup.sh
```

`source setup.sh` (not `bash setup.sh`) installs the pinned environment from `uv.lock`
and activates it. On first use it asks for four directories and saves them in the
git-ignored `.config`:

| key | meaning |
|---|---|
| `ENV_PREFIX` | this clone's virtualenv (unique per clone; several GB — on NERSC use `$SCRATCH` or CFS, not `$HOME`) |
| `UV_CACHE` | uv download cache (can be shared by all your clones) |
| `UV_PYTHON_DIR` | where uv puts the Python it downloads if none matches on `PATH` |
| `SAMPLE_DIR` | where `prepare_zenodo_samples.sh` puts the paper's samples (4.2 GB, see Data) |

Press Enter to accept a default. Later `source setup.sh` calls reuse `.config` and ask only
for keys that are missing; edit a value, or delete its line to be asked again. The Slurm scripts under
`src/parnassus/torch_delphes/` source `setup.sh` themselves and so use the same `.config`.

## Data

The samples of the paper are on Zenodo: [record 22071385](https://zenodo.org/records/22071385)
(DOI 10.5281/zenodo.22071385, CC-BY-4.0; 14 files, 4.2 GB). Download them all with

```bash
bash prepare_zenodo_samples.sh
```

The files go to `SAMPLE_DIR` from `.config` (asked for by `source setup.sh`). Files that are
already complete are skipped, partial downloads are resumed, and each download is md5-checked
against Zenodo, so the script can be re-run at any time.

| files | content |
|---|---|
| `pseudo_data_200k_param_config_{muons_muongun,electrons_electrongun,chads_ksgun,dijets_dijet}.root` | per-block regression samples, 200k events each |
| `pseudo_data_200k_param_config_all_{muongun,ksgun,dijet,electrongun}.root` | sequential all-parameter fit samples, 200k events each |
| `pseudo_data_100k_param_config_all_HZZ4l.root` | independent closure sample, 100k events (VBF H → ZZ → 4l) |
| `qcd_dijet.cmnd`, `muon_gun.cmnd`, `electron_gun.cmnd`, `kshort_gun.cmnd`, `HZZ4l.cmnd` | the Pythia8 process cards the samples were generated from (identical to the copies in `src/parnassus/torch_delphes/processes/`) |

Each ROOT file holds one tree, `event_tree`, with jagged branches `truth_{pt,eta,phi,class,pdgid}`
(stable generator particles) and `pflow_{pt,eta,phi,class}` (reconstructed particle-flow objects);
`class` is 0 charged hadron, 1 electron, 2 muon, 3 neutral hadron, 4 photon. Every sample was
produced by running Pythia8 with its card, passing the truth particles through the frozen torch
detector card initialised from a "truth" parameter config, and storing both; the fits start the
same card at the CMS defaults and should recover the truth values.

## Single parameter regression

One parameter block at a time. Each per-block sample was generated with one block of detector
parameters moved away from the CMS defaults (its "truth" YAML, `param_config_<block>.yaml`).
The fit starts the card at the defaults with only that block trainable
(`optuna_config_<block>.yaml`) and should recover the truth values.

| block | sample | truth-moved parameters |
|---|---|---|
| charged hadrons (K_S → π⁺π⁻ gun) | `pseudo_data_200k_param_config_chads_ksgun.root` | tracking efficiency (4) + momentum resolution/scale (9) |
| muons (J/ψ, Z → μμ gun) | `pseudo_data_200k_param_config_muons_muongun.root` | tracking efficiency (4) + momentum resolution/scale (9) |
| electrons (J/ψ, Z → ee gun) | `pseudo_data_200k_param_config_electrons_electrongun.root` | tracking efficiency (6) + momentum resolution/scale (9) |
| calorimeters (QCD dijets) | `pseudo_data_200k_param_config_dijets_dijet.root` | ECal scale + resolution (11), HCal scale + resolution (6) |

All commands run from the repo root after `source setup.sh` (which also defines `$SAMPLE_DIR`).
Run the fits on a GPU node; without CUDA they silently fall back to the CPU.

### Charged hadrons (K_S gun)

```bash
OUT=doc/figure_pseudodata_k0sgun
CFG=src/parnassus/torch_delphes/param_configs
# fit
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
    --root-file "$SAMPLE_DIR/pseudo_data_200k_param_config_chads_ksgun.root" \
    --optuna-config "$CFG/optuna_config_chads.yaml" \
    --n-events 200000 --n-steps 100 --n-trials 1 --loss wasserstein_1d --mode delphes \
    --output-base "$OUT" --history-path "$OUT/all_optuna.json"
# plot
python -m parnassus.torch_delphes.plotting_scripts.plot_parameter_regression \
    --workspace "$OUT" --truth-config "$CFG/param_config_chads.yaml"
```

### Muons (J/ψ, Z → μμ gun)

```bash
OUT=doc/figure_pseudodata_muongun
CFG=src/parnassus/torch_delphes/param_configs
# fit
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
    --root-file "$SAMPLE_DIR/pseudo_data_200k_param_config_muons_muongun.root" \
    --optuna-config "$CFG/optuna_config_muons.yaml" \
    --n-events 200000 --n-steps 100 --n-trials 1 --loss wasserstein_1d --mode delphes \
    --output-base "$OUT" --history-path "$OUT/all_optuna.json"
# plot
python -m parnassus.torch_delphes.plotting_scripts.plot_parameter_regression \
    --workspace "$OUT" --truth-config "$CFG/param_config_muons.yaml"
```

### Electrons (J/ψ, Z → ee gun)

```bash
OUT=doc/figure_pseudodata_electrongun
CFG=src/parnassus/torch_delphes/param_configs
# fit
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
    --root-file "$SAMPLE_DIR/pseudo_data_200k_param_config_electrons_electrongun.root" \
    --optuna-config "$CFG/optuna_config_electrons.yaml" \
    --n-events 200000 --n-steps 100 --n-trials 1 --loss wasserstein_1d --mode delphes \
    --output-base "$OUT" --history-path "$OUT/all_optuna.json"
# plot
python -m parnassus.torch_delphes.plotting_scripts.plot_parameter_regression \
    --workspace "$OUT" --truth-config "$CFG/param_config_electrons.yaml"
```

### Calorimeters (QCD dijets)

```bash
OUT=doc/figure_pseudodata_dijet
CFG=src/parnassus/torch_delphes/param_configs
# fit
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
    --root-file "$SAMPLE_DIR/pseudo_data_200k_param_config_dijets_dijet.root" \
    --optuna-config "$CFG/optuna_config_dijets.yaml" \
    --n-events 200000 --n-steps 100 --n-trials 1 --loss wasserstein_1d --mode delphes \
    --output-base "$OUT" --history-path "$OUT/all_optuna.json"
# plot
python -m parnassus.torch_delphes.plotting_scripts.plot_parameter_regression \
    --workspace "$OUT" --truth-config "$CFG/param_config_dijets.yaml"
```

`--mode delphes` matches how the samples were made (no acceptance cuts, no photon merging).
`--n-trials 1` runs only the seed trial, i.e. the learning rates given as `init:` in the optuna
config; a larger number adds TPE trials that search the per-group learning rates. `--seed N`
changes the torch seed and the train/validation split. The fit writes
`$OUT/round_0/{materialized_config.yaml,history.json,intermediate_plots/}` and copies the best
trial's history to `$OUT/all_optuna.json`; the plot is `$OUT/plots/params_reg.pdf`, one page per
parameter block with the fitted value per epoch (solid) against the truth (dashed). The three gun
samples are small (≤ 18 MB) and fit in minutes; the dijet sample (1.7 GB, 200k multi-particle
events) takes hours on a single GPU and tens of GB of host memory.
`plotting_scripts.plot_example_regression` (same `--workspace` / `--truth-config`) draws the paper's
two-page version: the muon momentum-scale block and the train / validation loss per epoch.

## Sequential all-parameter regression

The four `pseudo_data_200k_param_config_all_*` samples were generated with *every* block moved
away from the CMS defaults at once (`param_config_all.yaml`). `run_sequential.sh` recovers them
one sample per stage: each stage starts from the previous stage's fitted values of all
parameters and trains only the block its sample constrains (see
`src/parnassus/torch_delphes/full_phasespace_tuning/README.md` for the reasoning).

| stage | sample | trainable block |
|---|---|---|
| `stage1_muons` | `pseudo_data_200k_param_config_all_muongun.root` | muon efficiency + momentum resolution/scale (13) |
| `stage2_chads` | `pseudo_data_200k_param_config_all_ksgun.root` | charged-hadron efficiency + momentum resolution/scale (13) |
| `stage3_calo` | `pseudo_data_200k_param_config_all_dijet.root` | ECal + HCal scale and resolution (17; run with `--no-pair-mass`) |
| `stage4_electrons` | `pseudo_data_200k_param_config_all_electrongun.root` | electron efficiency + momentum resolution/scale (15) |

```bash
NPROC=1 OUT_BASE=doc/figure_sequential \
    bash src/parnassus/torch_delphes/full_phasespace_tuning/run_sequential.sh
```

The script sources `setup.sh` itself, so it reads the samples from `SAMPLE_DIR` in `.config`
(file names from `SAMPLE_PATTERN`, default `pseudo_data_200k_param_config_all_%s.root`). Every
other knob is an environment variable (see the script header): `N_STEPS` (100 epochs per stage),
`N_EVENTS` (-1 = all), `EARLY_STOP` (10), `PICK` (`best` | `last` epoch carried to the next
stage), `PLOT` (1). `NPROC` > 1 launches each stage with `torchrun` on that many GPUs and splits
the 4096-event global batch across them

Each stage writes `doc/figure_sequential/<stage>/round_0/{materialized_config.yaml,history.json,intermediate_plots/}`
and `train.log`; with `PLOT=1` it also draws `<stage>/plots/params_reg.pdf` (fitted value per
epoch against the `param_config_all.yaml` truth, as in the single-block fits) and the
`plot_fit_results` observable figures. The final card, every parameter at its last-stage value,
is `doc/figure_sequential/fitted_config.yaml`. 

Closure on the independent HZZ4l sample: the fitted card against the sample's target and the
CMS-default card, one page per species and observable plus the lepton pair masses and m_4l
(CPU, all events by default, `--n-events` to cap):

```bash
python -m parnassus.torch_delphes.full_phasespace_tuning.compare_sample \
    --sample "$SAMPLE_DIR/pseudo_data_100k_param_config_all_HZZ4l.root" \
    --fitted-config doc/figure_sequential/fitted_config.yaml
# -> doc/figure_sequential/distributions_pseudo_data_100k_param_config_all_HZZ4l.pdf (or --output)
```

## Regenerating a sample instead of downloading it

```bash
CFG=src/parnassus/torch_delphes/param_configs
python -m parnassus.torch_delphes.generate_pseudodata \
    --process ksgun --param-config "$CFG/param_config_chads.yaml" \
    --n-events 200000 --seed 1 --n-workers 32 \
    --output "$SAMPLE_DIR/pseudo_data_200k_param_config_chads_ksgun.root"
```

`--process` is `ksgun`, `muongun`, `electrongun` or `dijet`, paired with the matching
`param_config_<block>.yaml` for the single-block samples (the config may be partial: unlisted
parameters keep the card defaults); with `param_config_all.yaml` the same four processes give
the `pseudo_data_200k_param_config_all_*` samples of the sequential fit, and `--process HZZ4l
--n-events 100000` its closure sample. Pythia runs on `--n-workers` CPU processes; the detector pass uses a GPU when one is
available. The Zenodo files were produced with
`src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh`, which fans the same command
out over a Slurm array with distinct seeds and merges the parts, so a single run with one seed
is statistically equivalent but not bit-identical. The gun samples generate in minutes; 200k
dijet events take several CPU-hours (roughly 0.1 s per event on 32 cores).
