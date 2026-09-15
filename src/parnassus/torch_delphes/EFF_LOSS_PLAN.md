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
Decided 2026-09-10: the fullsim BCE support is ALSO `truth_in_tracker` (uniform
with closure), and the chad binning + event weight become mode-dependent in the
`cmssinglejet` merge — see Phase 2 below.

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

## Phase 2 — neutral (tower) BCE for the calo block (REORDERED 2026-09-14)

**User decisions (2026-09-14):**

1. **This phase comes BEFORE any fullsim work** — the cmssinglejet merge and
   everything else fullsim-related is now Phase 3, gated behind this.
2. **The survival probability factorizes over the cascade stages:**

   p(tower survives) = PROD_i p(survive cascade stage i)

   — one analytic factor per stage (for the Gaussian tower smear each threshold
   stage is a Phi() in the significance/energy variable; in delphes mode the
   stages are the E_min floor and the significance cut — the photon merger is
   off). This is a deliberate modeling CHOICE: the stages share the same smear
   draw, so the product double-counts their correlation relative to the exact
   single-threshold marginal (a mis-specification in the sense of
   BCE_eff_neutral_question.md section 4). Accepted for simplicity; the closure
   gate measures whether the bias matters.

   **AMENDED 2026-09-14 (after the gate FAILED on the product form — result
   below): the four cuts are nested thresholds on one smear draw, so q is now
   the single tail probability at the element-wise MAX threshold — exact given
   the conditioning (BCE_eff_neutral_question.md section 6). The product form
   remains the right shape only for stages with genuinely independent
   randomness, of which this cascade has none.**
3. **The end-to-end test is one sequential closure**: `run_sequential.sh` with
   the tower BCE active, saved to `doc/figure_seq_hung_neutral_BCE` (on the
   hungarian-labeled sample set; the tower labels themselves come from tower
   occupancy of the pflow branches at load time — no sample regeneration).
   Anything beyond that run is decided after seeing it.

**Gate (unchanged):** the fitted calo block must beat the count-free floor
(scales stuck at init, 14-27% off) and match the with-counts baseline (scales to
~1-2%, c_N / central and forward c_S at baseline level).

**RESULT (2026-09-14, `doc/figure_seq_hung_neutral_BCE/`): GATE FAILED.** The
efficiency blocks stayed clean (chad 0.0028 / muon 0.0025 / electron 0.011
median rel err — the charged/neutral separation held), but the calo block came
out WORSE than the count-free floor on most recoverable parameters: ECal scales
0.07-0.54 rel err (floor 0.14-0.27, with-counts 0.001-0.011), HCal scales
0.35-0.45 (floor 0.14-0.20), common_c_N 0.51 (floor 0.13). A few c_S improved
vs the floor (forward_c_S 0.05, central_c_S 0.14) — the term is not
information-free, but its VALUE model is wrong enough that the fit actively
drives the scales away from truth rather than merely failing to move them.

**UPDATE (2026-09-14, de-duped q + stage-scoped tower BCE,
`doc/figure_seq_hung_neutral_BCE/`): GATE LARGELY PASSED, scales still short.**
Two iterations after the product-form failure below:
(1) q de-duped to the single max-threshold tail (exact given the conditioning —
BCE_eff_neutral_question.md sec 6; the product under-counted, q ~ q_true^2 on
track-free towers); (2) the tower BCE restricted to the stage that TRAINS calo
params — in stage 4 (calo frozen, electron gun) the term is untrainable noise
whose value RISES as the electron coins converge (single-track towers make the
option-(i) model-draw conditioning maximally wrong), and it poisoned the
PICK=best epoch selection (best=epoch 0) while every real term improved. Fixed
by rerunning stage 4 from the stage-3 history with --calo-bce off
(best epoch 31/42, e_eff back to 0.0066).
Final card vs anchors (median |rel err|): efficiencies chad 0.0014 / e 0.0066 /
mu 0.0025 (baseline quality); calo RESOLUTIONS mostly at-or-better than the
count chi^2 (e.g. HCal forward_c_E 0.055 vs 3.56, forward_c_S 0.004, c_N 0.042);
calo SCALES 0.13-0.17 — clearly better than or equal to the count-free floor
(0.17-0.22) but ~15x short of the chi^2 baseline (~0.01). Remaining suspect: the
residual conditioning bias in q's mean (the scale's home) — the (ii)/(iii)
track-conditioning options are the designated lever. The stage-4 pathology is
itself empirical evidence for option (iii) (coin marginalization).
Earlier failed iterations preserved: `..._product/` (factorized q, all stages),
`..._dedup_allstages/` (de-duped q, tower BCE in every stage).

Original product-form diagnosis (2026-09-14): the decreed per-stage product is
the prime suspect — for a tower with
no track energy, stages 1/3 and 2/4 are duplicate thresholds, so
q_factorized ~ q_true^2 (systematically low), and the fit compensates by
mis-moving the scale that controls the Phi arguments; the single-draw track
conditioning adds Jensen noise on top (BCE_eff_neutral_question.md sections 3-4:
a mis-specified likelihood buys a pseudo-truth, which is exactly the observed
signature). Escalation options, in increasing cost: (a) dedupe the correlated
stages — one Phi at the max effective threshold per tower, exact given the
conditioning, still loss-only; (b) option (ii)/(iii) track conditioning; (c) the
per-tower MC calibration of q from BCE_eff_neutral_question.md section 5 to
attribute the bias before more design. Decision pending.

**Implementation decisions (all blockers resolved, 2026-09-14):**

1. **Calo count terms OFF** in the Phase-2 run (`--calo-count-weight 0`): the
   tower BCE is tested as a REPLACEMENT, against the existing with-counts gate
   and count-free floor.
2. **Per-region-fair weighting**, the same schema as the calo count terms it
   replaces: the per-|eta|-region tower-BCE means combine with EQUAL region
   weight (not tower-population weight), protecting the forward-region
   c_E / c_S leverage exactly as CALO_COUNT_WEIGHT's per-region-fair chi^2 did.
3. **q(theta) conditions on the forward pass's own sampled quantities**
   (option (i)): the track energy/sigma under the subtraction are the tensors
   the calo already computed this draw (detached — no gradient into track
   params, matching the soft-count convention); only the tower's own Gaussian
   smear is marginalized analytically per stage. NOTHING about the TorchDelphes
   forward changes — outputs are byte-identical; the loss only reads new
   exports. Circle-back options, deliberately not chosen now:
   (ii) plug in the EXPECTED track energy instead of the sampled one;
   (iii) marginalize the track smears/efficiency coins (1-D numerical integral
   per tower). Revisit if the closure gate fails or the per-draw q proves too
   noisy.

   Note on (ii)/(iii) and the tracking efficiencies (discussed 2026-09-14):
   both make q_tower an explicit function of the eff_logits — (iii) necessarily
   (the coin marginal is a sum over survival subsets weighted by
   sigmoid(eff_logit) products), (ii) under its natural reading
   (E[E_trk] = sum_i eps_i E[E_i]) — and likewise of the track smearing
   parameters through the E_trk distribution. This gradient is physically real
   (a dead track's energy re-emerges as neutral excess: tower occupancy carries
   chad-efficiency information via the chad -> NH conversion channel), but
   WHETHER it flows is a routing choice separable from the estimator choice:
   - (ii)/(iii) with the track factors DETACHED = better-specified q, same
     clean separation (default if adopted): eff_logits keep their exact
     tracking-BCE likelihood uncontaminated by the tower model's
     approximations (factorization bias, calo mis-modeling, label subtleties)
     — the same reasoning as the count term's deliberate detachment.
   - (ii)/(iii) with the eps-dependence LIVE = a joint fit through the
     conversion channel: extra leverage on eff_logits from neutral occupancy,
     gated on the closure harness showing the efficiency recovery does not
     degrade (mixing an approximate secondary gradient into an exactly-fitted
     parameter must earn its keep empirically).
4. **Log-space evaluation everywhere, NO probability floor**: the tracking BCE
   already evaluates in logit space (binary_cross_entropy_with_logits); the
   tower BCE uses log Phi (log_ndtr) — same loss values, underflow-proof, tail
   towers give large-but-finite exact terms. The floor idea (a spurious-object
   rate epsilon) is shelved unless tail towers destabilize training. Balance
   between the charged and tower BCE terms: separate weight knobs (existing
   --bce-weight; new calo analog), disjoint parameter ownership (eff_logits vs
   calo scales/resolutions, track inputs detached) — calibrated like step 7.
5. **Support = towers the forward materializes** (i.e. cells with a truth
   deposit): a never-deposited cell has q identically 0 with no
   theta-dependence and x = 0 in closure, so its BCE term is exactly zero —
   skipping it changes no value and no gradient. (The HadronFractions corner
   case — model-empty-but-data-occupied cell in one calorimeter's grid — is
   handled gracefully by the log-space evaluation.)

### The tower-existence BCE (hashed out 2026-09-14; promoted to Phase 2 — see the
decisions above)

Motivated by the count-free interlude (the calo block NEEDS membership
information) and the observation that the neutral analog of the tracking BCE
exists once the coin is placed at the right level. Agreed so far:

- **The unit is the tower, not the particle.** Neutrals in one tower live or die
  together (one shared threshold event), so a per-particle BCE would count one
  coin multiple times; per-tower deduplicates it. Labels ARE constructible for
  neutrals (matching, or tower survival at generation) — the earlier "no labels
  for neutrals" claim was wrong; what neutrals lack is only an independent
  per-particle coin.
- **q(theta) exists**: it is the per-tower survival probability behind the
  expected-neutral-count export (the soft significance gate) — a function of the
  calo scales/resolutions rather than of a dedicated parameter, which is fine:
  BCE only needs a differentiable predicted probability. Caveat: the count
  machinery's gate is STRAIGHT-THROUGH (forward pinned to the hard 0/1 count) —
  ideal inside the rate chi^2, fatal inside log(q). A BCE consumer must use the
  soft sigmoid value (one-draw stochastic, Jensen-biased) or, recommended, the
  ANALYTIC marginal over the Gaussian smear (a Phi() in the significance
  variable): closed-form, differentiable, a true probability.
- **Closure labels need no regeneration**: the tower grid is a deterministic
  eta-phi binning and the trainee runs on the same truth events as the data, so
  x_tower = "data has a neutral object in this cell" is computable from the
  existing pflow branches at load time (ECal grid -> photons, HCal grid -> NH).
- **Validation gate**: with the calo rate-chi^2 off, the tower BCE must
  reproduce the with-counts calo recovery (scales to ~1-2%, c_N / central and
  forward c_S at the with-counts level). The count-free run is the failure
  baseline it must beat.

Open (NOT yet hashed out — discussion in progress): the exactness of the
analytic q vs the full hard decision cascade (track subtraction / neutral-excess
arbitration, E_min + significance); the tower support (which towers enter — tail
towers give log(q) blowups); region weighting (a pooled tower BCE
population-weights regions, which may re-drown the forward-region c_E/c_S
leverage that CALO_COUNT_WEIGHT's per-region-fair form was built to protect);
the fullsim tower<->object correspondence (off-grid reco, photon merger);
implementation plumbing (per-tower q/id export, DDP gathers, weight
calibration).


## Phase 3 — fullsim (plan agreed 2026-09-10; GATED behind Phase 2 above per the 2026-09-14 reorder)

**Decisions (user-confirmed):** the fullsim BCE support is **`truth_in_tracker`**
(uniform with closure; resolves the open question in EFF_LOSS_MOTIV.md §3b — the
all-input-particles alternative has the q=0 / double-counting problem). The chad
region binning AND the event weight become **mode-dependent** so none of the
Phase-1 pseudodata artifacts or results need regenerating.

### Interlude — the count-free variant (user-requested, 2026-09-10)

`doc/figure_sequential_hungarian_matched_survival_wo_neutral_counts/`: the hms
closure rerun with `--calo-count-weight 0`, i.e. ZERO count terms of any kind
(bce mode already drops the tracking ones structurally). Verdict:

- **Efficiency blocks: unaffected or better** — median |rel err| chad 0.0011
  (vs 0.0034 with calo counts), muon 0.0017 (vs 0.0019), electron 0.0066
  (vs 0.0060). The BCE terms alone fully determine the efficiencies.
- **Calo block: badly degraded — the calo count terms are load-bearing well
  beyond the wrong-signed-c_E story.** The RECOVERABLE calo parameters
  collapse: ECal/HCal energy scales stall near their init (~1.02) instead of
  fitting the perturbed truth (0.80-1.19), going from ~1-2% error to 14-27%;
  common_c_N 0.13 vs 0.01; central/forward c_S 0.25-0.32 vs ~0.01. Without the
  membership gradient, the calo stage effectively fails to fit.
- HZZ4l closure PDF rendered on-GPU on the same allocation
  (`distributions_HZZ4l.pdf` in the dir).

Conclusion: the "calo count terms stay" decision is now empirically forced, not
just argued. A count-free loss is viable for the tracking-efficiency block only.

### Phase 2b — conditioning experiments (user-directed 2026-09-14): the 2x2 matrix

Motivated by the remaining scales gap (0.13-0.17 vs the chi^2's ~0.01) and the
stage-4 pathology, both of which point at the option-(i) model-draw
conditioning. Try, IN ORDER: conditioning (ii) then (iii), EACH split into
(a) detached and (b) live gradient routing — four runs:

| run | conditioning | eff/smearing gradients |
|-----|--------------|------------------------|
| ii-a | expected track energy | `.detach()` — none |
| ii-b | expected track energy | LIVE |
| iii-a | full marginalization | `.detach()` — none |
| iii-b | full marginalization | LIVE |

**Definitions (per tower):**
- (ii) replaces the sampled track energy in q's thresholds with its
  coin-expectation over THIS draw's smears:
  `E[E_trk] = sum_i eps_i * E_i^(sampled smear)`, with
  `eps_i = compute_efficiency(pt_i, eta_outer_i)` of the track's species module
  (pre-mask smeared kinematics), and correspondingly
  `sigma_trk^2 -> sum_i eps_i * sigma_i^2`. Coins averaged, smears still
  conditioned.
- (iii) additionally marginalizes the track-side randomness. HYBRID evaluation
  (user dislikes loss approximations; 2026-09-14): for towers with few tracks
  (n <= 3, incl. the pathological single-track case) enumerate the coin subsets
  EXACTLY (2^n <= 8 Phi-terms); only for busier towers moment-match E_trk to
  N(m, v) with the exact coin+smear moments
  `m = sum_i eps_i E_i`, `v = sum_i [eps_i(1-eps_i) E_i^2 + eps_i sigma_i^2]`
  and integrate by 1-D Gauss-Hermite quadrature (~10 nodes; standard tool for
  Gaussian expectations of Phi; exact for polynomials of degree 2K-1, and CLT
  makes many-track E_trk near-Gaussian precisely where enumeration is
  expensive). NOT needed for (ii), which is quadrature-free by construction
  (a closed-form plug-in of the coin-expected track energy).
- (a)/(b): the eps_i and E_i factors above are detached (a) or left on the
  graph (b). Under (b), the tower BCE hands eff_logits (and the track smearing
  params through E_i) the physically-real conversion-channel gradient
  (dead track -> neutral excess); see the routing note above.

**Design constraint + config assignment (user ruling 2026-09-14):** sequential
stage 3 freezes ALL efficiency and smearing parameters, so live gradients need a
joint config to have anywhere to go. Ruling:
- (a) DETACHED arms use the standard `stage3_calo.yaml` (calo-only trainable) —
  i.e. today's champion pipeline with only the conditioning upgraded.
- (b) LIVE arms use `stage3_calo_joint.yaml` (calo block + chad efficiency +
  chad smearing trainable, starting from stage-2's fitted history).
Note the (a)/(b) pairs therefore differ in BOTH the detach and the trainable
set; if a (b) result ever needs disambiguating ("was it the gradients or just
the unfrozen chads refitting on dijet shape terms?"), the control is a
joint-config-with-detach run — not scheduled unless needed.
Stages 1-2 reused from the existing hungarian run; stage 4 refit from each
variant's stage-3 history (tower BCE auto-scoped off there).

**Clarification of the (b) gate (it is an ACCEPTANCE TEST, not a design
choice):** after a (b) run finishes, compare its fitted chad efficiency +
smearing against truth. If they are worse than stage-2 quality, the live
gradients did net harm (the imperfect tower term used the chad parameters to
absorb its own bias) and (b) is rejected; adoption of live gradients requires
"calo improves AND chads don't degrade".

**Plumbing:** `--calo-bce-conditioning {sampled,expected,marginal}` (default
sampled = today's option (i)) and `--calo-bce-grads {detach,live}` (default
detach); pre-mask smeared track tensors + per-track eps threaded from
CMSDefault into the calo export.

**Gates per variant:** (1) calo scales — the number to beat is 0.13-0.17
(current sampled-conditioning result), target the chi^2's ~0.01; (2) calo
resolutions stay at-or-better than current; (3) for the (b) arms ONLY: the chad
efficiency and smearing recovery must not degrade vs their stage-2 values
(the joint fit must not let the tower term contaminate the exactly-fitted
blocks) — if it does, (b) is falsified and (a) stands.

Output dirs: `doc/figure_seq_hung_nBCE_cond{ii,iii}_{detach,live}`.

### Phase 2b — RESULTS (all four arms + the joint-detach control + pass-2s; 2026-09-15)

Median |rel err| vs truth per block (fitted_config.yaml vs param_config_all):

| block | ii-a | iii-a | ii-b | iii-b | CONTROL (joint+detach) | chi^2 ref |
|---|---|---|---|---|---|---|
| chad eff | 0.0014 | ~ii-a | 0.0077 | (worse) | 0.0033 | — |
| chad smear | 0.0538 | ~ii-a | 0.0325 | 0.071 | 0.0568 | — |
| ECal scales | 0.0113 | ~ii-a | 0.0106 | ~ | 0.0174 | 0.0092 |
| HCal scales | 0.1374 | 0.13-0.17 | 0.1420 | 0.13-0.17 | 0.1174 | 0.014 |

Verdicts:
1. **(iii) ~ (ii)**: marginal conditioning bought nothing over expected under
   the (then-current) thresholds — the extra machinery is not the lever.
2. **(b) live gradients REJECTED by the acceptance test**: chad eff degrades
   (0.0014 -> 0.0077) and the control run pins it on the gradients, not the
   joint config (control keeps 0.0033 with the same trainable set but detached);
   chad smear results are inconsistent between ii-b (0.033) and iii-b (0.071).
3. **HCal scales fail in EVERY arm** (~0.75/0.75 fitted vs truth 0.849/0.893,
   near-identical across runs) — conditioning-independent, so the defect had to
   be in the tower-smear marginal itself. See the HCal debug below.
4. Pass-2 sweeps (stages 2->4 rerun from ii-a's fitted card): full-LR pass 2
   improves ECal scales 0.0113 -> 0.0071 (beats the chi^2 ref) and chad smear
   0.054 -> 0.031, but degrades chad eff to 0.0088 and leaves HCal untouched
   (0.144). The lr/10 variant (archived `..._pass2_lr01`) got ECal 0.0087 but
   chad smear 0.139. Pass 2 is not the HCal fix.

### HCal-scale debug (overnight 2026-09-14/15) — root cause found + fix

Question: why do HCal scales converge with counts info but not BCE info, and
why don't they move right when everything else is converged?

**Exonerated:** the optimizer and the plumbing. Gradient fingerprint at the
fitted point: distinct, finite-difference-consistent per-region gradients
([+0.0057, +0.0049] on scale_raw) — no broadcast/tying bug.

**Convicted (value-level pseudo-truth):** frozen-truth scans (everything at
truth, sweep one HCal scale): the count chi^2 minimizes exactly at truth
(0.849 / 0.893) while TowerBceHcal is MONOTONE toward low scale in region 0
(no interior minimum) and has a shallow biased-low minimum ~0.83 in region 1.
The fitted 0.75 is the equilibrium of this wrong pull against the shape terms.

**Root cause:** the tower BCE evaluated the two sigma-dependent cascade
thresholds at sigma_after(THIS DRAW's sampled smeared energy) — a fixed
threshold plugged into the tail probability. But the hard cut is
`E_sm > S * sigma(E_sm)`, SELF-CONSISTENT in E_sm: the true pass event is
`E_sm > E*` where E* is the unique fixed point of `E = S * sigma(E)`
(unique because E - S*sigma(E) rises from negative through one crossing when
S*c_E < 1). Same for the neutral-significance cut with
`E = E_trk + S*sqrt(sig_trk^2 + sigma(E)^2)`. The sampled-sigma plug-in is
biased near threshold AND adds draw-dependent variance.

**MC q* recalibration** (`doc/hcal_bce_debug/hcal_mc_qstar_v2.py`; 150
replicas x 256 dijet events at truth; a v1 of this MC was CORRUPTED — see the
gotcha below): bias (q_cand - q*) with the legacy sampled-sigma thresholds vs
the self-consistent fix, HCal support towers:

| trackfrac bin | n | biasA (sampled_sigma) | biasB (self_consistent) |
|---|---|---|---|
| trackless | 71730 | +0.0296 | **+0.0000** |
| (0, 0.3] | 9977 | +0.0756 | -0.0248 |
| (0.3, 0.6] | 4921 | +0.0465 | -0.0352 |
| near-threshold trackless | 16937 | +0.0713 | **-0.0013** |

Near-threshold RMS |q - q*| halves (0.131 -> 0.054). The trackless tail is
EXACT under the fix (as theory says: lognormal marginal + exact threshold);
the residual negative bias sits only on track-carrying towers = the (ii)
expected-conditioning residual, now unmasked.

**Frozen-truth scan under the fix** (`doc/hcal_bce_debug/hcal_scan_v2.py`):
region 1 (forward, trackless): minimum moves from ~0.83 to 0.90 — AT truth
(0.893). Region 0 (central, track-rich): monotone -> interior minimum at
~0.90-0.95, residual high-side shift consistent with the tracked-tower
conditioning bias. Both logs in `doc/hcal_bce_debug/`.

**Code:** `tower_bce_threshold {sampled_sigma,self_consistent}` on
SimpleCalorimeter/CMSDefault, CLI `--calo-bce-threshold`; self_consistent
solves the fixed points by 25 contraction iterations (energy argument detached
per iteration — the sigma_after_c gradient convention; resolution coefficients
stay live). Default stays sampled_sigma until the closure gate passes.

**Closure validation — GATE PASSED (2026-09-15, ~02:45):** stage-3-only refits
from ii-a's stage-2 history (same flags as ii-a + `--calo-bce-threshold
self_consistent`; exact commands in each dir's reproduce.sh; HZZ4l PDFs
rendered on-GPU in both dirs). Median |rel err| vs truth:

| block | chi^2 (dd) | ii-a (sampled_sigma) | ii+selfcon | iii+selfcon |
|---|---|---|---|---|
| ECal scales | 0.0092 | 0.0113 | 0.0018 | **0.0022** |
| ECal res | 0.1293 | 0.0812 | 0.0881 | **0.0823** |
| HCal scales | 0.0136 | 0.1374 | 0.0506 | **0.0168** |
| HCal res | 1.3940 | 0.3665 | 0.8099 | 0.9398 |

- **iii+selfcon fitted HCal scales 0.8650 / 0.9063 vs truth 0.8487 / 0.8934 —
  both ~1%, matching the chi^2 quality (0.014).** ii+selfcon gets region 1
  exactly (0.9049) but keeps a high-side region-0 residual (0.9237 vs 0.8487)
  — precisely the tracked-tower conditioning bias the MC table predicts, and
  marginal conditioning (iii) removes it. So (iii) DOES matter once the
  thresholds are right (reversing the earlier "(iii) ~ (ii)" verdict, which
  was measured under the broken thresholds).
- ECal scales improve ~6x over ii-a and ~4x over the chi^2 reference.
- HCal res looks worse than ii-a (0.94 vs 0.37) but BOTH beat the chi^2
  reference's 1.39 — that block is poorly determined in every pipeline (its
  apparent quality anticorrelates with how wrong the scales are; ii-a's "good"
  0.37 was the res absorbing the mis-fitted scales).
- Default (`sampled_sigma`) deliberately NOT flipped yet — recommend
  `--calo-bce-threshold self_consistent` + `--calo-bce-conditioning marginal`
  becomes the new champion setting; morning decision.

**Gotcha discovered on the way (pre-existing, affects analysis scripts only):**
`card(input)` MUTATES its input tensor in place (10 columns). Training is safe
(`tp[mask]` advanced indexing copies), but any script looping
`card(flat)` over replicas feeds each replica the previous one's mutated
input — this corrupted MC-v1 and explains its "tower set changed across
replicas" assert. Always pass `flat.clone()`. With that, the forward is
exactly deterministic per seed (tower set byte-stable across replicas).

### Step F1 — Merge `origin/diff_delphes_runze_cmssinglejet`, on a new branch

`git merge-tree` (2026-09-10) shows the merge into `BCE_eff` is textually CLEAN —
14 incoming commits (incl. `0dbccc1`/`4650b39`, deliberately excluded from
`diff_delphes` earlier but wanted here: the 12-bin chad structure is the point of
the fullsim fits), 19 files, zero conflicts; only `loss.py`,
`plot_distribution.py`, `test_torch_delphes_learnable.py` are touched by both
sides, in disjoint hunks. Do it on **`BCE_eff_fullsim`** so the validated closure
state on `BCE_eff` stays frozen.

### Step F2 — Mode-dependent chad binning (the semantic-conflict fix)

The incoming `4650b39` replaces `CMS_EFF_REGION_SPECS["charged_hadron"]` globally
(4 -> 12 regions, pt edges 0.1/1/10/25/50/100 x 2 |eta| bins), which changes the
card's eff_logits SHAPE (68 -> 76 scalars) and shifts the electron/muon label
offsets (4 -> 12, 10 -> 18) — silently invalidating every stored
`truth_eff_region`. Instead of regenerating, make the spec a toggle:

- Two chad variants, `cms4` (delphes/pseudodata: the Phase-1 binning, unchanged)
  and `ptbins12` (fullsim: Runze's binning). Everything downstream — e/mu label
  offsets, `_BCE_EXCLUDED_LABELS`, loader count targets, BCE species ranges,
  region tagging — already derives from the spec, so it adapts per variant.
- Card constructor knob (e.g. `chad_binning=`) wired from `--mode` in `cli.py`,
  `optuna_search.py` and `generate_pseudodata` (delphes -> cms4,
  fullsim -> ptbins12; overridable).
- **Provenance guard**: labeled files record which spec wrote their
  `truth_eff_region` (in the `.provenance.json` AND checked at load), so a file
  can never be read under the wrong offsets.
- Variant-aware odds and ends: the "exactly 68 parameters" test becomes 68/76 by
  variant; `param_configs/` (4-bin indices) pairs with delphes,
  `param_configs_fullsim/` (12-bin) with fullsim, as they already do in spirit.
- Consequence: NO regeneration of Phase-1 samples, fixture, or closures.

### Step F3 — Mode-dependent event weight

The incoming branch flips the default `EVENT_WEIGHT` 0.1 -> 1.0 ("increase the
event weight to 1" for cms fullsim). Keep BOTH: default 0.1 in `--mode delphes`
(preserves Phase-1 comparability), 1.0 in `--mode fullsim` (Runze's calibration),
resolved next to `--eff-loss` and overridable by the explicit `--event-weight`.

### Step F4 — Fullsim preprocessing (labels for train_1000.root)

Extend/run the matcher preprocessing on the CMS opendata file: run OUR
`ParticlePropagator` on the gen particles to fill `truth_in_tracker` and
`truth_eff_region` (eta at the outer tracker radius; the ptbins12 spec — this is
§5.3 option (a), unused in Phase 1 where generation stored regions for free),
then deltaR-gated Hungarian matching to pflow fills `truth_survived`; write the
branches back in the same schema so the training code runs unchanged. Support =
`truth_in_tracker` (decided): sub-threshold gen particles our propagator drops
get no label — they are unfixable by any efficiency value.

### Step F5 — Validation before any real-data fit

- One-off 12-bin closure smoke: generate a small labeled pseudodata sample under
  the ptbins12 spec, fit, check the closed-form survival fractions — validates
  the 12-bin code path that Phase-1 closure never exercised.
- Fullsim-mode pseudodata dry-run (cuts ON, matcher labels, `truth_in_tracker`
  support) against known truth: measures the REAL label noise (Phase-1's zero
  relied on delphes smearing preserving track direction; real data adds angular
  smearing/fakes) and the double-counting residual (pt-migration across the
  harmonized reco-pt cut).

### Step F6 — Fullsim fit and beyond

`--mode fullsim --eff-loss bce` on the preprocessed `train_1000.root`; compare
against the count-term fullsim baseline (`--eff-loss counts`, the fullsim
default until this validates). Interpretation caveat stands: the fitted
efficiency is the effective P(matched reco | model-propagated gen particle) —
real losses our propagator doesn't model land in it by construction. Then the
Step-10 Commit B excision of the tracking-count machinery becomes decidable.
