# Sequential full-phase-space tuning

Recover every card parameter from the four `param_config_all` pseudo-data samples
(muon gun, K_S gun, dijet, electron gun) **one sample per stage**: each stage starts from
the previous stage's final values of *all* parameters and trains only the block that
sample constrains. A trainable parameter with no lever drifts, and a later stage would
otherwise overwrite an earlier one, so the masks below are the whole method.

Why not one joint fit on the shuffled 800k mix: the pair-response terms are standardized
by one std pooled over all classes; dijet leading-2 charged-hadron pairs (a mis-pairing
continuum, std 1.9, half of all pairs) push it from ~0.15 to 1.42 and suppress the gun
pair terms 50-250x (2026-08-18). Fixing that (per-class / per-truth-group robust scale) is a
separate loss change.

| stage | sample | trainable | frozen at fitted values | why here |
|---|---|---|---|---|
| 1 `stage1_muons` | muongun | muon eff[0,1,3,4], a, b, scale (13) | -- | track-only, no cross-talk |
| 2 `stage2_chads` | ksgun | chad eff, a, b, scale (13) | muon | track-only; chad tracks are needed before the calo stage |
| 3 `stage3_calo` | dijet | ECal + HCal (17, incl. the four c_E) | muon, chad | calo needs the chad tracks; `--no-pair-mass` |
| 4 `stage4_electrons` | electrongun | electron eff, a, b, scale (15) | muon, chad, calo | electron energy = track (+) ECal, so after the calo stage |

Never trained: muon eff[2,5] and rate_raw (> 1 TeV), ECal barrel_a (anchor), HadronFractions.
Expected: scales, efficiencies, chad/muon/electron a, calo scales, c_N, forward c_S recover;
b partial (unrecoverable for electrons); the four c_E and common_c_S do not. (A possible later
stage: Z -> ee as an ECal lever at 45 GeV -- ECal common_c_S / barrel_b / endcap_a / endcap_b on
the electron gun with the electron block frozen -- is not part of the chain for now.)

## Files

- `stage*_*.yaml` -- one per stage: `process` (picks the sample), `trainable` (the mask),
  `extra_args` (appended to the tuning CLI).
- `make_stage_config.py` -- builds the stage's full `--param-config` (`{value, trainable,
  lr_scale}` for every card scalar): values = card defaults or the previous stage's
  `history.json` snapshot (`--pick last|best`), trainable = the mask, `lr_scale` = the
  absolute per-group lr (optuna seed values), used with `--lr 1`. Without `--stage` it
  writes the all-frozen final card.
- `compare_sample.py` -- closure on an independent sample (e.g. the HZZ4l pseudo-data), in the
  `plotting_scripts/plot_distribution.py` style (same `draw_page`): one PDF, one page per
  (species, log pT / log E / eta) for target vs trainee at the CMS card defaults ("initial") vs
  trainee at `fitted_config.yaml` ("tuned"), the leading-2 pair-mass response pages, plus
  m_ee / m_mumu (leading 2 same-flavour leptons) and m_4l (leading 4 e/mu). Delphes mode, all
  events by default (`--n-events`), CPU (~20 s per 5k gun events).
- `run_sequential.sh` -- runs the stages in order (`python -m
  parnassus.torch_delphes.tune_cms_fullsim`, `--loss wasserstein_1d --mode delphes
  --pid-weighting sqrt_fraction` (`PID_WEIGHTING`; the fitted species is the abundant one in
  every stage, so this mutes the stray-species shape terms without touching the count / pair /
  log HT levers), early stopping with patience `EARLY_STOP` (default 10; each stage's val loss sits on a floor from the
  frozen species, so late epochs only track noise), no lr decay, the early-stopping checkpoint
  carried on (`PICK=best`)), plots each stage (`plot_parameter_regression` -> `params_reg.pdf`,
  `plot_fit_results`, both against `param_config_all.yaml`), and writes
  `<OUT_BASE>/fitted_config.yaml`.

## Run

```bash
# compute node with 4 GPUs (the CLI's global batch is 4096 = 1024 per rank; the dijet
# stage needs >= 2 ranks to fit in memory)
bash src/parnassus/torch_delphes/full_phasespace_tuning/run_sequential.sh
# dry run
N_EVENTS=4000 N_STEPS=2 PLOT=0 OUT_BASE=/tmp/seq_dry bash .../run_sequential.sh
# resume from stage 3 with stage 2's result (D = this directory)
FROM_HISTORY=doc/figure_sequential/stage2_chads/round_0/history.json \
    bash $D/run_sequential.sh $D/stage3_calo.yaml $D/stage4_electrons.yaml
```

Closure on the HZZ4l sample (after `submit_pseudodata.sh ... --process HZZ4l` with
`param_config_all.yaml`):

```bash
python -m parnassus.torch_delphes.full_phasespace_tuning.compare_sample \
    --sample /global/cfs/cdirs/m3246/diff_delphes/allsamples/pseudo_data_100k_param_config_all_HZZ4l.root \
    --fitted-config doc/figure_sequential/fitted_config.yaml
# -> doc/figure_sequential/distributions_pseudo_data_100k_param_config_all_HZZ4l.pdf (or --output)
```

Outputs per stage under `doc/figure_sequential/<stage>/` in the optuna-workspace layout:
`round_0/materialized_config.yaml` (the stage's START config), `round_0/history.json`
(trajectory + per-epoch snapshots of all 68 values), `round_0/intermediate_plots/`,
`plots/params_reg.pdf` (fitted vs `param_config_all` truth per block) + the observable
figures, `train.log`, `plot.log`. Each stage's `history.json` is the input of the next.
