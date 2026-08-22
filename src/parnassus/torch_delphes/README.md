# `torch_delphes` — fitting a differentiable Delphes card to CMS full sim

`torch_delphes` is the CMS energy-flow Delphes card re-implemented in PyTorch with learnable
detector parameters (tracking efficiencies, momentum resolutions and scales, calorimeter
resolutions and scales). Because the whole chain truth particles → detector response →
particle-flow objects → loss is differentiable, the parameters are fitted with Adam directly to
CMS full-simulation particle-flow candidates. The fitted card is then compared, on held-out
events, against full sim and against naive C++ Delphes.

Two commands do the work: **fit** (`tune_cms_fullsim.optuna_search`) and **validate**
(`plotting_scripts.plot_fullsim_comparison`). Everything below is run from the repo root after
`source setup.sh` (uv environment with torch, uproot, fastjet, mplhep).

---

## 1. Data

All samples live in `/global/cfs/cdirs/m3246/diff_delphes/cms_opendata_zenodo/`
(CMS open-data jets, Zenodo 11389651).

| file | content |
|---|---|
| `train_1000.root` (≈ 1 TeV jets, **the sample we fit**), `test_470.root`, `test_600.root`, … | `event_tree`: `truth_*` (generator particles: pt, eta, phi, class, pdgid) and `pflow_*` (full-sim PF candidates, pt ≥ 1 GeV); pt in GeV |
| `delphes_pu6p35_jet1000.root`, `delphes_pu6p35_jet470.root`, … | naive C++ Delphes (PU μ = 6.35) on the same process: `fastsim_tree` (`fs_pt` in **MeV**, `fs_class` 1 = charged / 0 = neutral) + `truth_tree` + `pflow_tree` |

Class codes: 0 charged hadron, 1 electron, 2 muon, 3 neutral hadron, 4 photon. Note that
`delphes_pu6p35_jet470.root` holds the same events as `test_470.root`, while
`delphes_pu6p35_jet1000.root` is an independent sample of the same process (not event-aligned
with `train_1000.root`).

## 2. Fit

The production setup (1 TeV jets, 5 GeV reco floor, population-fraction species weighting):

```bash
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
    --root-file /global/cfs/cdirs/m3246/diff_delphes/cms_opendata_zenodo/train_1000.root \
    --optuna-config src/parnassus/torch_delphes/param_configs_fullsim/optuna_config_ptbins.yaml \
    --n-events 100000 --n-steps 100 --n-trials 1 \
    --loss wasserstein_1d --mode fullsim --reco-pt-cut 5 --pid-weighting fraction \
    --output-base doc/figure_fullsim/1000_frac_pt5 \
    --history-path doc/figure_fullsim/1000_frac_pt5/all_optuna.json
```

What it does:

- **Model**: `CMSEnergyFlowDefault(learnable=True)` with the photon-cluster merger
  (R = 0.09, set in the YAML). 76 scalars, 64 trainable: charged-hadron tracking efficiency in
  12 (pT × |η|) bins, momentum-smearing `a`/`b`/`scale` for charged hadrons, electrons and muons
  in three |η| regions, ECal/HCal scales and resolution terms; the hadron-fraction logits and the
  bins below the reco floor are pinned (see the YAML header for the bin legend).
- **Data**: the first `--n-events` entries, split into train/validation inside that range. Truth
  particles enter the card with pt ≥ 0.25 GeV, |η| ≤ 2.7; reco objects on both sides are kept
  with pt ≥ `--reco-pt-cut` and |η| ≤ 2.7; the reco charged hadrons of the target are truncated
  per event to the truth charged-hadron count (`--no-chad-truncation` to disable).
- **Loss** (`--loss wasserstein_1d`): per-species 1D quantile-Wasserstein distances on log pT,
  log E and η, weighted by population fraction (`--pid-weighting fraction`), plus per-bin object
  count terms (the only gradient path of the efficiencies), a per-event log HT term
  (`--event-weight`) and the leading-pair mass term.
- **Optimizer**: Adam with per-group learning rates from the YAML, global batch 2048, early
  stopping on the validation loss (patience 10). With `--n-trials 1` Optuna just runs the YAML's
  `init` values; raise `--n-trials` to search learning rates and the merger radius.

Output (`--output-base`):

```
round_0/history.json              metadata (cuts, radius, loss settings), per-epoch train/val loss
                                  and parameter snapshots, best_result (lowest val loss)
round_0/materialized_config.yaml  the initial {value, trainable, lr_scale} of every scalar
round_0/intermediate_plots/       observable overlays every 10 epochs
all_optuna.json                   copy of the best round's history.json
```

Parameter values in `history.json` are physical (efficiencies in (0,1), scales around 1,
resolutions in GeV units); `plotting_scripts.plot_parameter_regression` and the plot scripts
read them back with `_set_trainee_from_snapshot`.

## 3. Validate — the full-sim comparison

```bash
python -m parnassus.torch_delphes.plotting_scripts.plot_fullsim_comparison \
    --workspace doc/figure_fullsim/1000_frac_pt5 \
    --sample /global/cfs/cdirs/m3246/diff_delphes/cms_opendata_zenodo/train_1000.root
```

Runs the fitted card (best epoch) on held-out entries [100 000, 120 000) and overlays up to four
legs — the same entry range and the same number of jets on every leg (a file that cannot cover
the range is an error) — under identical cuts (pt ≥ reco floor, |η| ≤ 2.7):

- **full sim** (plot label `CMS`) — the sample's `pflow_*` objects,
- **diff-Delphes** (plot label `Parnassus-P`) — the fitted card on the same truth (no pileup, untruncated),
- **C++ Delphes** (plot label `Delphes`) — `fastsim_tree` of the sibling `delphes_pu6p35_jet<bin>.root`
  (derived from the sample name; `--delphes` overrides),
- **Parnassus** (plot label `Parnassus-F`) — `fastsim_tree` of the generative fast sim, auto-picked by the sample's pT-hat
  bin from `PARNASSUS_FILES` in the script (`/global/cfs/cdirs/m3246/diff_delphes/parnassua_data/`,
  bins 800 and 1000; other bins skip the leg; `--parnassus` overrides, `--no-parnassus` drops it).
  Parnassus generates only (pt, η, φ) — no charged/neutral class — which is all the response
  pages need. Its jets are the first 200 000 entries of the Delphes file of
  the same bin, i.e. the same events as the C++ leg (and as full sim for `test_800`; an
  independent sample of the same process for `train_1000`); the metrics header states which.

Pages (4): the leading jet's (anti-kt R = 0.5, pt > 8 GeV, ≥ 2 constituents, |η| < 2.5) pT and
mass response (x − x_truth)/x_truth, η − η_truth and φ − φ_truth (wrapped to [−π, π)) relative to
the truth jet clustered from all truth particles of the event. Linear y, ratio panel to full sim
(band = full-sim √N), fixed binning per page (`BINS` at the top of the script).

Output in the fit folder: `plots/fullsim_comparison_pt<floor>.pdf` and `plots/fullsim_comparison_pt<floor>_metrics.txt`
(per page: quantile-W1 distance to full sim for each leg and the yield ratio N_leg/N_fullsim).
`--reco-pt-cut X` evaluates the same fit at another floor (`fullsim_comparison_ptX.*`), which is how fits
trained at different floors are ranked against each other. ~2 minutes on a CPU for 20 000 events
(`--n-events`); set `OMP_NUM_THREADS` on shared login nodes.

## 4. Other entry points

| command | purpose |
|---|---|
| `tune_cms_fullsim` (the plain CLI, `--param-config`) | a single fit without Optuna; same flags |
| `plotting_scripts.plot_distribution --workspace … --sample …` | per-species target / initial / tuned overlays of one fit |
| `plotting_scripts.plot_parameter_regression` | parameter recovery on pseudodata closure tests |
| `generate_pseudodata` | generate pseudodata with a known parameter set (closure tests) |

Configs for the delphes-mode closure work live in `param_configs/`; the full-sim configs in
`param_configs_fullsim/`.


