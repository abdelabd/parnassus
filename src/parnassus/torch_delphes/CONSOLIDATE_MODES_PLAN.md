# CONSOLIDATE_MODES_PLAN — one branch, three behaviors

Branch: `consolidate_modes` (created from the `BCE_eff` tip `4773249`).
Goal: a single branch where

| user toggle | reproduces |
|---|---|
| `--mode delphes --existence counts` | `diff_delphes` (count-term losses on pseudodata) |
| `--mode delphes --existence bce` | `BCE_eff` (survival/tower BCE on labeled pseudodata) |
| `--mode fullsim` | `diff_delphes_runze_cmssinglejet` (counts only, 12-bin chad eff, real data) |

`--existence` is a delphes-mode toggle; `--mode fullsim --existence bce` is a
hard error for now (BCE-on-fullsim is EFF_LOSS_PLAN.md Phase 3, out of scope).
All inherited optional parameters keep their current defaults.

## 0. Branch topology (verified 2026-09-15)

```
f56f619 diff_delphes_runze_restructure
  ├── (+2 commits) 4650b39 diff_delphes_runze_cmssinglejet
  └── (+9 commits) f777dd5 diff_delphes
        └── (+~20 commits) 4773249 BCE_eff == consolidate_modes (start)
```

- **`BCE_eff` strictly descends from `diff_delphes`** — every counts-mode code
  path already exists on `consolidate_modes`. Phase 1 is therefore a toggle
  definition + a parity PROOF (that ~20 BCE-era commits touching shared files —
  SimpleCalorimeter, CMSDefault, loss/training/data/dataloader — did not drift
  counts-mode behavior), not a merge.
- **`cmssinglejet` = `restructure` + 2 commits**, and the delta is confirmed to
  be exactly the delphes/fullsim discrepancy:
  1. chad tracking-efficiency binning 4 → 12 (pt,|eta|) regions
     (`learnable.py` `CMS_EFF_REGION_SPECS`: pt edges (0.1,1) →
     (0.1,1,10,25,50,100); electron/muon label offsets shift 5-10→13-18,
     11-16→19-24; `_DEFAULTS` replicate legacy values across sub-bins), with
     the config/test updates the new shapes force (`param_config_all/chads`,
     `cms_target_default`, `optuna_config*`, 4 test files);
  2. new fullsim configs `param_configs_fullsim/optuna_config_ptbins*.yaml`;
  3. plotting additions (leading-jet features in `plot_distribution.py`,
     tweaks in `plot_fit_results.py` / `plot_parameter_regression.py`).
  No training/loss/CLI logic is touched by the fullsim branch.
- **`cmssinglejet` predates `diff_delphes`' 9 commits** and therefore LACKS the
  dtype-DDP fix, the pair-mass DDP-gather fix, the comet stage naming, and the
  thread-count bump. Consolidation upgrades fullsim with these fixes — see the
  phase-2 parity caveat.

## 1. Phase 1 — consolidate counts and BCE (delphes mode)

### 1a. The `--existence` toggle

New flag on `tune_cms_fullsim` (cli.py) and `optuna_search.py`:
`--existence {counts,bce}`, **default `bce` in delphes mode** (today's
BCE_eff default), rejected with a clear error in fullsim mode (fullsim is
counts by construction until Phase 3). It is an umbrella that sets the
existing knobs' DEFAULTS; every underlying knob stays overridable:

| knob | `--existence counts` | `--existence bce` |
|---|---|---|
| `--eff-loss` | `counts` | `bce` |
| `--calo-bce` | off | ON (marginal conditioning — the only code path) |
| `--calo-count-weight` | default (1.0) | 0.0 |
| tracking count terms | on | off (structurally, as today) |

(The tower-BCE thresholds are self-consistent UNCONDITIONALLY — the
`--calo-bce-threshold` option was removed and the fixed-point evaluation
hardcoded before phase 1 started, per the user's 2026-09-15 ruling; own
commit. The bundle therefore only sets the three knobs above.)

**Decision (user, 2026-09-15): confirmed** — `--existence bce` bundles the
champion settings; explicit per-knob flags override the bundle. Also
confirmed: gate the tower-BCE export off in counts mode (§1b).

Precedence rule (simple + testable): argparse defaults < `--existence` bundle
< explicitly passed flags. Implemented by resolving the bundle only for args
the user did not explicitly pass (argparse `SUPPRESS` sentinel pattern).

`run_sequential.sh` needs no change (`EXTRA_ARGS="--existence counts"` etc.),
but its README/CLAUDE.md examples get the new spelling.

### 1b. Parity audit (code review before any run)

Known BCE-era edits that could have drifted counts-mode behavior — check each:
- `data.py`: label-free samples (`pseudo_data_200k_param_config_all_%s.root`)
  must still load under `--existence counts` (the survival-label hard error is
  bce-only). Verify the counts-era sample files still exist on cfs.
- `CMSDefault`: the tower-BCE `track_cond` export is now built UNCONDITIONALLY
  in learnable mode, and `SimpleCalorimeter` computes the marginal `bce_logq`
  whenever `compute_soft_count` — extra compute in counts mode, and it must
  consume NO RNG (it doesn't — coin enumeration + GH are deterministic) so
  draws stay identical. Decide: gate the bce export on a `tower_bce_enabled`
  switch (CMSDefault kwarg set from `--calo-bce`/`--existence`) to reclaim the
  compute — behavior-neutral either way, but gating restores counts-mode speed.
- `loss.py` `_count_terms(include_tracking=...)` and the count/event/pair term
  weights: confirm the counts path with today's defaults is term-for-term
  identical to `diff_delphes`' loss (same keys, same weights, same DDP gather).
- `training.py` `_inject_bce_logits`/`_inject_tower_bce`: must be no-ops (not
  even label lookups that can throw) under counts.
- `EVENT_WEIGHT` and other config.py constants: diff vs `diff_delphes`.

### 1c. Consistency tests (the gate)

Protocol for every parity pair: git worktree of the reference branch under
`/pscratch` + `PYTHONPATH=<worktree>/src` shadowing (assert `parnassus.__file__`),
same node type (1x4), same seed, `N_EVENTS=50000 N_STEPS=10`, EARLY_STOP=0,
PLOT=0. Compare per-epoch: total train/val loss, every loss-component
trajectory, and every trainable parameter trajectory from `history.json`.
Pass = trajectories identical to float noise (|rel dev| < 1e-4 per epoch;
bitwise equality expected on stage 1/2 where code paths should be untouched —
investigate ANY drift before waving it through).

1. **BCE parity:** `run_sequential.sh` (all 4 stages, short) with
   `--existence bce` on `consolidate_modes` vs the BCE_eff tip `4773249` with
   today's champion flags (`--calo-bce --calo-bce-grads detach
   --calo-bce-threshold self_consistent --calo-count-weight 0`), hungarian
   samples. This proves the toggle refactor is a pure re-plumbing.
2. **Counts parity:** same protocol with `--existence counts` on
   `consolidate_modes` vs `diff_delphes` (`f777dd5`) with ITS defaults, on the
   unlabeled sample set (`SAMPLE_PATTERN=pseudo_data_200k_param_config_all_%s.root`).
   This is the real test: ~20 commits of shared-file history must wash out.

**RESULTS (2026-09-15) — both pairs PASS with max |rel dev| = 0.0 EXACTLY**
(bitwise-identical train/val losses and every parameter at every epoch, all
four stages, both pairs; drivers + logs + outputs in
`doc/consolidate_tests/parity_{bce,counts}.{sh,log}` and
`{bce,counts}_{ref,new}/`). So: (a) the `--existence` resolver is a pure
re-plumbing of the champion/legacy flag bundles; (b) the entire BCE-era
history including the new tower-BCE export gating is byte-neutral for
counts-mode training.

## 2. Phase 2 — consolidate fullsim (counts only)

### 2a. Mode-dependent chad binning (the one semantic conflict)

The 12-region spec cannot simply replace the 4-region one — every delphes
pseudodata card (`param_config_all.yaml` etc.), the closure results, and the
sequential stage YAMLs assume `eff_logits[0..3]` and labels chad 1-4 /
electron 5-10 / muon 11-16. Plan:
- `learnable.py` carries BOTH spec sets: `CMS_EFF_REGION_SPECS` (legacy 4-bin,
  delphes default) and `CMS_EFF_REGION_SPECS_PTBINS` (12-bin, verbatim from
  `cmssinglejet` incl. shifted offsets). The learnable efficiency modules take
  the spec (and matching `_DEFAULTS`) as a constructor choice.
- `CMSDefault` gains `eff_binning: str = "cms4"` (`"ptbins12"` the
  alternative); `--mode fullsim` maps to `ptbins12` in cli.py/optuna_search.
  Label offsets always derive from the active spec.
- Configs: delphes YAMLs stay 4-bin; `param_configs_fullsim/` comes over
  verbatim (12-bin). `cms_target_default.yaml` / `param_config_all/chads.yaml`:
  KEEP the 4-bin delphes versions (the cmssinglejet 12-bin rewrites of these
  are only needed for fullsim-mode pseudodata sanity runs — port them as
  `*_ptbins.yaml` variants only if/when needed).
- Tests: keep the current (4-bin) assertions; add ptbins12 shape/label tests
  adapted from the cmssinglejet test diffs.

### 2b. The rest of the fullsim delta

- Port `param_configs_fullsim/optuna_config_ptbins{,_noscale}.yaml` verbatim.
- Port the plotting additions (`plot_distribution.py` leading-jet features,
  `plot_fit_results.py`, `plot_parameter_regression.py` tweaks) — resolve the
  textual overlap with BCE_eff's own `plot_distribution.py` edits by hand.
- Fullsim mode keeps counts existence-terms only: `--mode fullsim` +
  `--existence bce` errors out (Phase 3 later: truth_in_tracker labels,
  mode-dependent EVENT_WEIGHT — deliberately NOT ported now).

### 2c. Consistency test (the gate)

The user's reference command, shortened (`--n-events 50000 --n-steps 10`), on
`cmssinglejet` (worktree + PYTHONPATH shadow) vs `consolidate_modes`:

```
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
  --root-file /global/cfs/cdirs/m3246/diff_delphes/cms_opendata_zenodo/train_1000.root \
  --optuna-config src/parnassus/torch_delphes/param_configs_fullsim/optuna_config_ptbins.yaml \
  --n-events 50000 --n-steps 10 --n-trials 1 --loss wasserstein_1d \
  --output-base doc/consolidate_tests/fullsim_{ref,new} \
  --history-path doc/consolidate_tests/fullsim_{ref,new}/all_optuna.json \
  --mode fullsim --reco-pt-cut 5 --pid-weighting fraction
```

Compare `all_optuna.json` loss/parameter trajectories as in 1c.
**Caveat:** consolidate_modes carries the dtype-DDP + pair-mass-gather fixes
that `cmssinglejet` lacks. Run the parity at world_size=1 (single GPU) so the
DDP fixes cannot bite; if trajectories still differ, bisect whether the delta
is exactly one of the known fixes (acceptable, document it) or new drift (fix).

**RESULT (2026-09-15) — PASS: max |rel dev| 6e-15 (losses) / 2e-14 (all 76
parameters, 10 epochs), identical best val loss.** Driver + log + outputs in
`doc/consolidate_tests/parity_fullsim.{sh,log}`, `fullsim_{ref,new}/`.
Two consolidation bugs were caught and fixed by this gate (both "the card is
ptbins12 but a second code path still assumed the module-level cms4 layout"):
the optuna-config validation probe card, and the DATA-side per-species count
targets (`_build_pflow_event_data`) — `eff_binning` is now threaded through
`load_search_config`, `load_split_datasets`, and both pflow target loaders.

## 3. Execution order

1. Plan review (this file) → user sign-off on the `--existence bce` bundle
   question (§1a) and the `tower_bce_enabled` gating question (§1b).
2. Phase 1 implementation (one commit), gate tests green
   (34 gate + 47 fullsim-suite), then the two §1c parity runs (one commit
   recording results here).
3. Phase 2 implementation (one commit: binning/spec plumbing; one commit:
   configs+plots port), gates green + §2c parity run (results recorded here).
4. Update CLAUDE.md §4 (fullsim no longer needs a branch switch) and the
   README examples; final summary commit.

Artifacts: parity outputs under `doc/consolidate_tests/` (each with a
`reproduce.sh`); worktrees under `/pscratch/sd/a/aelabd/worktrees/` (removed
after the tests).

## 4. Risks / notes

- RNG parity: the toggle refactor must not add/remove any `torch.rand*` call
  or reorder module construction in ways that shift draw streams — parity
  tests are run same-seed to catch exactly this.
- `card(input)` mutates its input in place (see EFF_LOSS_PLAN.md gotcha):
  parity fingerprint scripts must pass `input.clone()`.
- The counts-era unlabeled samples must still exist on
  `/global/cfs/cdirs/m3246/diff_delphes/allsamples/` — verify before §1c.2.
- `optuna_search.py` and `cli.py` duplicate flag definitions — the
  `--existence` resolution helper lives in ONE place (config.py) and both
  import it, so the bundles cannot drift apart.
