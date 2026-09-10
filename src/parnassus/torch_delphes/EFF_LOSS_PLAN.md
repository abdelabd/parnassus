# BCE efficiency loss — implementation plan

Step-by-step plan for replacing the 3 tracking-efficiency count terms with the
per-particle BCE survival loss. Rationale, decisions, and background:
`EFF_LOSS_MOTIV.md`. Steps are sequential; each has its own verification gate so we
never build on an unvalidated layer.

## OVERNIGHT EXECUTION ORDER (user, 2026-09-09 — authoritative; details in the steps below)

0. ~~Archive the fresh count-term `run_sequential.sh` results as
   `doc/figure_sequential_dd`~~ DONE (this is the Step-0 baseline).
1. Update the pseudodata generation code (detailed Steps 1-2 below).
2. Generate the labeled samples — new filenames with a `_truth_matched_survival`
   tag before `.root`, same CFS area. Scope (user-confirmed): the four
   `param_config_all` sequential processes (200k each) + the test fixture + the
   single-stage samples (`param_config_muons` muongun, `param_config_chads` ksgun,
   200k) + `param_config_all` HZZ4l (100k). (Detailed Step 3.)
3. While generation runs on SLURM: training/loss code — BCE term behind
   `--eff-loss {counts,bce}`, **default `bce`** in delphes mode; tracking count
   terms kept in code, calo count terms always on. (Detailed Steps 4-7.)
4. `run_sequential.sh` on the `_truth_matched_survival` samples →
   `OUT_BASE=doc/figure_sequential_truth_matched_survival`. (Detailed Steps 8-9.)
5. Preprocessing script: Hungarian truth-reco matching (detailed Phase-2 item 3,
   run on pseudodata).
6. Apply it to produce a second labeled sample set, `_hungarian_matched_survival` —
   keep the TRUE labels under a separate branch name so the matcher confusion
   matrix comes for free.
7. `run_sequential.sh` on those → `doc/figure_sequential_hungarian_matched_survival`.

Operational decisions (user-confirmed 2026-09-09):
- Work on branch **`BCE_eff`** (created from `diff_delphes` @ f777dd5). Commit
  early and often with documentation; **do NOT push**, and do NOT commit to
  `diff_delphes`.
- I am authorized to submit ALL SLURM jobs myself tonight (CPU shared-QOS
  generation arrays; GPU closure runs via `salloc --no-shell` on the interactive
  QOS, falling back to sbatch on the regular QOS if interactive is capped).
- `submit_pseudodata.sh` defaults `REPO`/`ENV_PREFIX` to Runze's checkout/env —
  MUST override to this checkout and `parnassus_env`. Write a `reproduce.sh` next
  to every run/output (CLAUDE.md rule 5). Never overwrite existing samples.
- Set `COMET_NAME_PREFIX` per closure run (`tms`, `hms`) so Comet experiments are
  distinguishable.
- End-of-night deliverable: **`NEXT_DAY.md`** at the repo root — summary of what
  was done, results, failures, and what's queued; **under 1000 words**.

Standing decisions (from EFF_LOSS_MOTIV.md): calo count terms stay; two-phase rollout
(closure first, fullsim via matching preprocessing second); label schema
`truth_survived` / `truth_in_tracker` / `truth_eff_region` on every input particle;
closure BCE support = `truth_in_tracker`; neutrals never consumed by BCE;
`rate_raw` / muon `eff_logits[2,5]` stay frozen.
Open: the fullsim support question (EFF_LOSS_MOTIV.md §3b) — Phase 2 blocks on it.

---

## Phase 1 — closure (pseudodata)

### Step 0 — Pin the baseline

No code. Record what "as good as today" means before touching anything:

- Keep (or rerun once) a count-term sequential closure as the comparison anchor:
  `bash src/parnassus/torch_delphes/full_phasespace_tuning/run_sequential.sh` with
  `OUT_BASE=doc/figure_sequential_countbaseline`.
- Note the pytest baseline: `test_tune_cms_fullsim.py` = 23 failed / 24 passed
  (stale fixture); `test_torch_delphes_learnable.py`, `test_loss_ddp_gather.py` green.

**Gate:** baseline `params_reg.pdf` + `fitted_config.yaml` archived.

### Step 1 — UID column through the card forward

The propagator drops rows and splits species (`ParticlePropagator.py:105-185`), so
labels need a particle identity that survives the forward pass.

- `src/parnassus/data/particle_io.py`: add `UID = 21` to `ColumnMap` (after
  `EFF_REGION = 20`); bump the row width constant and audit every fixed-width
  construction site (grep for the current width literal and `EFF_REGION`).
- Confirm the UID column is *transported*, not interpreted: propagator masking
  (`ParticlePropagator.py:178`) and efficiency masking (`learnable.py:530-541`,
  momentum columns only) already preserve untouched columns; check the calo/eflow
  track path (`EFlowMerger._transform_tracks`, `EFlowMerger.py:150-224`) carries it
  to the output track rows. Neutral/tower outputs get no UID (expected).

**Gate:** full test suite at today's baseline (23/24 file unchanged, others green);
a new unit test builds a toy batch, runs the card forward, and asserts output track
rows carry the input UIDs.

### Step 2 — Generator writes the label branches

All in `generate_pseudodata.py` (+ a small hook file):

- `truth_arrays_to_pflow` (`generate_pseudodata.py:513-620`): before the card call at
  line 588, set `UID = flat truth row index` (1:1 with the `truth_*` arrays after
  padding-row drop at lines 579-580).
- Recording hook at the efficiency stage: capture, per species, the module *input*
  rows' `(UID, EFF_REGION)` — `EFF_REGION` is already tagged pre-mask by
  `_tag_eff_region` (`CMSDefault.py:319-337,452-470`). Implementation: a lightweight
  opt-in recorder on `_LearnableEfficiencyBase.forward` (`learnable.py:514-541`)
  storing `(uid, region, mask)`; do NOT reuse the `--debug` capture path — it filters
  the killed rows (`debug.py:190-200`). Recording the mask too is free and feeds the
  invariant test below.
- After the forward: derive the three per-truth-particle arrays by scattering over UID —
  `truth_in_tracker` (UID seen at an efficiency module input), `truth_eff_region`
  (from the hook; -1 elsewhere), `truth_survived` (UID present in the final output
  with pt > 0 — note `eflow_to_class_arrays` drops pt==0 ghosts at
  `generate_pseudodata.py:363`, so collect UIDs *before* that filter or from its kept
  rows). Neutrals: `truth_survived` via nothing for now — leave False/absent
  (BCE never reads them; revisit in Phase 2 if bookkeeping wants matched neutral labels).
- Write the three jagged branches next to the existing ones
  (`generate_pseudodata.py:557-562`).
- Embed provenance: write the resolved param-config YAML text + `git rev-parse HEAD`
  + CLI args into the output file (uproot supports string objects / a metadata tree).

**Gate (unit tests on a tiny in-process generation, no SLURM):**
- consistency: `truth_survived ⇒ truth_in_tracker`; `truth_eff_region >= 0` exactly
  on `truth_in_tracker`; charged species only.
- **the delphes-mode invariant**: recorded efficiency mask == `truth_survived` on the
  support, per charged species (this is what licenses the uniform output-presence label).
- closed form: per-region survival fraction ≈ `sigmoid(truth eff_logits[region])`
  within binomial error on a few-thousand-event gun sample.

### Step 3 — Regenerate fixture and samples

- Regenerate the `test_tune_cms_fullsim.py` pseudodata fixture with the new generator
  (this also restores the missing `truth_pdgid` and should take the file from
  23 failed / 24 passed to all green — update the recorded baseline in
  `torch_delphes/CLAUDE.md` and project memory afterwards).
- Production samples (USER launches; prepare the commands + a `reproduce.sh`):
  `submit_pseudodata.sh --config src/parnassus/torch_delphes/param_configs/param_config_all.yaml
  --process {muongun,electrongun,ksgun,dijet} --n_events 200000` from OUR branch,
  `OUTBASE` pointing at a NEW directory (e.g. `.../diff_delphes/allsamples_v2`) —
  never overwrite Runze's `allsamples/`. Also regenerate the single-stage samples
  (`param_config_muons.yaml` + muongun, `param_config_chads.yaml` + ksgun) used by
  the smaller runs in `torch_delphes/CLAUDE.md` §3.
- Run the Step-2 closed-form check as a standalone script over each produced file.

**Gate:** all four `_v2` samples exist, closed-form check passes per species/region,
fixture-driven test file green.

### Step 4 — Reader (`tune_cms_fullsim/data.py`)

- `load_cms_flow_root` (`data.py:69-103`): add the three branches to the read set
  (optional — absent in old files).
- `_build_truth_rows` (`data.py:109-186`): carry the labels through the SAME
  acceptance `sel` mask (`data.py:142-152`) so they stay index-aligned.
- Expose per-species BCE arrays on the dataset/target object: for each of
  {chad, electron, muon}, `(region_idx, x)` restricted to `truth_in_tracker`,
  split train/val alongside events. Return `None` when branches are absent.

**Gate:** unit test on the new fixture: shapes, alignment under acceptance cuts,
`None` on an old-schema file.

### Step 5 — Loss term (`tune_cms_fullsim/loss.py`)

- `_bce_eff_terms(...)`: one term per species —
  `F.binary_cross_entropy_with_logits(eff_logits[region_idx], x)` (index the raw
  logits, not `sigmoid` then `log` — exact and numerically stable). Species weight:
  mirror the count terms' `--pid-weighting` redistribution
  (`loss.py:830-841`, `_population_weights_from_counts` at loss.py:893) so `equal`
  stays a no-op. Overall scale: `bce_weight` (Step 7 calibrates the default).
- Assembly: in the `--eff-loss bce` path, skip the 3 tracking count terms in
  `_count_terms` (keep the 2 calo terms) and append the BCE terms in the term list at
  `loss.py:1481-1486` (and the `wasserstein` variant's at `loss.py:1097-1101`).
  Count machinery is NOT deleted yet (Step 10).
- DDP: global weighted mean via `diff_all_reduce` of per-rank `(sum, n)` per species
  — same pattern as the count reduce (`loss.py:1855-1868`). Empty-rank rule: a rank
  with no labeled muons must contribute a graph-connected zero (the
  `pred["pt"].reshape(-1)[:0]` idiom, `loss.py:1831-1842`).

**Gate:** unit tests — (a) toy fit recovers the closed-form per-region survival
fraction; (b) gradient flows to `eff_logits` and to nothing else; (c) extend
`test_loss_ddp_gather.py`: world_size 1 vs 2 give identical loss/grad, including a
rank with zero labeled particles of one species.

### Step 6 — Wiring (`cli.py`, `optuna_search.py`, `training.py`)

- Flags: `--eff-loss {counts,bce}` (default `counts`) and `--bce-weight` in `cli.py`
  (near the count flags, cli.py:171-213) and mirrored in `optuna_search.py:739-741`.
- Plumb through `training.py`'s loss-kwargs injection (`training.py:234-263`); pass
  the card's three efficiency-logit tensors (from the modules at
  `learnable.py:543/566/585`) and the per-species label arrays into the loss context.
- `--eff-loss bce` on a file without the branches: hard error naming the
  regeneration step, no silent fallback.
- Comet: log the BCE terms per epoch in `_log_comet_epoch` (`training.py:418-461`).

**Gate:** 1-GPU smoke run on the new muon-gun sample
(`... tune_cms_fullsim --eff-loss bce --n-events 4000 --n-steps 2 ...`) runs end to
end; old sample + `--eff-loss bce` produces the intended hard error;
`--eff-loss counts` is bit-identical to today (regression guard).

### Step 7 — Calibrate `--bce-weight`

On a reference muon-gun batch, compare `eff_logits` gradient norms under BCE vs the
count-term baseline; choose the default so the efficiency block moves at a comparable
rate (the shape/pair terms are untouched, so only this ratio matters). Record the
number and the measurement in this file.

**MEASURED (2026-09-09, 2048-event muon-gun batch, card at CMS defaults, login CPU):**
muon `eff_logits` |grad| per populated region — counts: [2.6e-3, 1.7e-2, 2.3e-3,
5.9e-3]; BCE (pooled, weight 1.0): [8.7e-3, 3.0e-1, 1.1e-2, 7.6e-2] — BCE is 3-17x
larger. Kept `BCE_WEIGHT = 1.0`: Adam normalizes the step by the gradient RMS, so
the magnitude difference barely changes step sizes; it raises the efficiency
block's SNR toward a near-convex optimum, and `eff_logits` receive no other
gradient to balance against. Loss scale is also comparable (BCE subtotal ~1.8 nats
at the untrained card vs count subtotal ~0.5; the BCE converges to the Bernoulli
entropy floor ~0.6-0.7, a constant offset that does not affect argmin/early
stopping).

**Gate:** default committed; smoke run shows stable early epochs (no efficiency-block
blow-up or freeze). DONE — 2-step CLI smokes in both modes ran clean.

### Step 8 — Single-stage closure validation

**RESULT (2026-09-10, muon stage, 1x4 interactive, `doc/figure_stage1_muons_tms/`
vs the count-term baseline `doc/figure_sequential_dd/stage1_muons/`):** BCE
recovers the four trained muon efficiency bins as well as or better than counts —
|fit - truth| (BCE vs counts): eff[0] 0.0021 vs 0.0061, eff[1] 0.0011 vs 0.0020,
eff[3] 0.0002 vs 0.0101, eff[4] 0.0010 vs 0.0008 (3-50x tighter on two bins,
comparable on the rest). The frozen > 1 TeV bins stay at card defaults in both, as
designed. The smearing block (a/b/scale_raw) is statistically identical between
the two runs — the BCE swap does not disturb the shape/pair-term fits. Val losses
are not comparable across modes (BCE adds a ~0.65-nat Bernoulli entropy floor).

Interactive 1x4 node (see `torch_delphes/CLAUDE.md`):

- Muon stage: `optuna_search` single trial on the `_v2` muon-gun sample with
  `--eff-loss bce` vs `--eff-loss counts`; compare `params_reg.pdf` (convergence
  speed, final bias/spread of `eff_logits[0,1,3,4]`; no regression in `a/b/scale_raw`).
- Repeat for chads (ksgun). Expect BCE to converge faster/tighter on the efficiency
  block (near-closed-form problem).

**RESULT (2026-09-10, chad stage, sequential stages 1-2 in
`doc/figure_sequential_truth_matched_survival/` vs baseline
`doc/figure_sequential_dd/stage2_chads/`):** BCE beats counts on every chad
efficiency bin — |fit - truth| (BCE vs counts): eff[0] 0.0033 vs 0.0237 (7x),
eff[1] 0.0021 vs 0.0068 (3x), eff[2] 0.0012 vs 0.0018, eff[3] 0.0008 vs 0.0058
(7x). Chad smearing block statistically identical between modes. Same convergence
speed (best epoch 20 vs 19). Both single-stage gates PASSED.

**Gate:** BCE recovers truth efficiencies at least as well as counts on both stages;
smearing parameters unaffected. If not, stop and diagnose before Step 9.

### Step 9 — Full sequential closure

- `SAMPLE_DIR=.../allsamples_v2 EXTRA_ARGS="--eff-loss bce" bash run_sequential.sh`
  (stage YAMLs unchanged; the calo stage's count terms still active).
- Compare `fitted_config.yaml` + per-stage `params_reg.pdf` against the Step-0
  baseline; `compare_sample.py` closure on HZZ4l (regenerate that sample too if used).

**Gate:** full-chain closure ≥ baseline on the efficiency blocks, == baseline
elsewhere. This closes Phase 1.

**RESULT (2026-09-10, full 4-stage tms closure in
`doc/figure_sequential_truth_matched_survival/` vs the count-term baseline
`doc/figure_sequential_dd/`): GATE PASSED.** Median |relative error| vs truth,
BCE vs counts — chad eff 0.0034 vs 0.0108 (3x better), muon eff 0.0019 vs 0.0049
(2.5x better), electron eff 0.0060 vs 0.0057 (equal); ECal/HCal recoverable
parameters (scales, c_N, central/forward c_S) identical to the 3rd decimal (the
calo loss is unchanged); all large calo residuals are the documented-unrecoverable
set (c_E's, common_c_S, barrel_b/endcap a/b), where mode differences are
single-seed scatter. Electron smearing identical (0.166 vs 0.164 — dominated by
the known-partial b_raw). Ran pipelined: stages 1-2 as soon as their samples
merged, stages 3-4 resumed from stage-2 history after the dijet merge.

### Step 10 — Flip the default, then excise (separate commits)

- Commit A: default `--eff-loss bce` for `--mode delphes`; `counts` stays the fullsim
  default. Update `full_phasespace_tuning/README.md`, stage YAML comments,
  `torch_delphes/CLAUDE.md`.
- Commit B (only after Phase 2 decides fullsim's fate): excise the tracking-count
  machinery — the 3 entries in `COUNT_TERM_KEYS` (`config.py:102-117`), their region
  observables in `data.py:376-469`, `_expected_reco_counts`
  (`CMSDefault.py:478-540`), and the count gathers at `loss.py:1855-1868` for the
  tracking species. Calo count terms and their plumbing stay.

---

## Phase 2 — fullsim (outline; blocks on the support decision, EFF_LOSS_MOTIV.md §3b)

1. **Decide the fullsim BCE support** (recommended: `truth_in_tracker`, uniform with
   closure — see the q-support/double-counting argument).
2. **Matcher survey** (short): confirm deltaR-gated Hungarian
   (`scipy.optimize.linear_sum_assignment`, class/charge-compatible pairs, cost
   deltaR^2 + optional relative-pt penalty, max-cost gate) is adequate vs current
   practice.
3. **Preprocessing script** (standalone, like `mix_gun_samples.py`): read a fullsim
   file → run OUR `ParticlePropagator` on gen particles (fills `truth_in_tracker`,
   `truth_eff_region`) → Hungarian-match survivors to pflow (fills `truth_survived`,
   + match index/cost branches) → write back. Output schema identical to Step 2's, so
   Steps 4-6 code runs unchanged.
4. **Label-noise measurement**: run the matcher on `_v2` pseudodata and compare
   matched labels to the true ones — confusion matrix per species/region = the
   systematic on fullsim efficiencies; quantify the double-counting residual
   (pt-migration near the reco-pt cut).
5. **Fullsim fit**: `--mode fullsim --eff-loss bce` on the preprocessed
   `train_1000.root` (on `diff_delphes_runze_cmssinglejet` or after merging that
   branch); compare against the count-term fullsim baseline.
6. Then Step-10 Commit B (excision) becomes decidable.
