# Plan: replace the object-count loss terms with a per-particle BCE efficiency loss

Proposed loss (per efficiency species):

```
L_eff(theta) = - < x * log q(x; theta) + (1 - x) * log(1 - q(x; theta)) >_{x ~ p_data}
```

where `x` is the binary label "this truth particle survived reconstruction" in the
**target** data and `q` is the trainee's per-particle survival probability evaluated on
that same truth particle's kinematics. This is the exact Bernoulli negative
log-likelihood of the efficiency parameters — a direct, unbiased, per-particle gradient
path, replacing the indirect reco-space count matching we use today.

## 1. What the count terms are today (what "ditching" removes)

Five count terms are built in `_count_terms` (`tune_cms_fullsim/loss.py:718-844`,
assembled into the total at `loss.py:1481-1486` / `loss.py:1097-1101`; wiring table
`COUNT_TERM_KEYS` / `CALO_COUNT_TERM_KEYS` in `tune_cms_fullsim/config.py:102-117`):

- **3 tracking-efficiency terms** (ChargedHadron, Electron, Muon): population-weighted
  Pearson chi^2 between predicted and target per-region object *rates* (regions in
  |eta| x pt). The gradient reaches `eff_logits` through a value-exact reweighting of a
  gradient-free migration histogram (`CMSDefault._expected_reco_counts`,
  `defaults/CMSDefault.py:478-540`) — NOT through the survival mask, which is
  deliberately `.detach()`ed in the forward (`learnable.py:526`; comment at
  `learnable.py:31-47`: the straight-through gradient was a biased survivor-momentum
  scale).
- **2 calo-resolution terms** (EcalPhoton, HcalNeutralHadron): per-region squared
  relative error on soft-gated tower counts (`SimpleCalorimeter.compute_soft_count`,
  `SimpleCalorimeter.py:766-839`). These feed the calo resolution coefficients
  (`forward_c_E`, `forward_c_S`, `common_c_E`), for which they are the only
  correctly-signed d(membership)/d(theta) gradient source (`loss.py:63-74`).

Gradient dependency map:

| parameters | current gradient source | after BCE swap |
|---|---|---|
| `*TrackingEfficiency.eff_logits` (chad/e/mu) | ONLY the 3 tracking count terms | BCE (direct) |
| `MuonTrackingEfficiency.rate_raw` | none (frozen: the reweighting trick is blind to it, `learnable.py:594-604`) | BCE reaches it (see 5.6) |
| calo `c_E`/`c_S`/`c_N`, scales | shape terms + 2 calo count terms | unchanged — BCE does not cover these (see 2) |

## 2. Scope boundary: BCE can replace the tracking count terms, not the calo ones

The BCE label requires a 1:1 truth-particle <-> output-object correspondence. That
exists only for the **charged track streams** (charged hadrons, electrons, muons): the
efficiency modules zero momentum columns without dropping or reordering rows
(`learnable.py:530-541`). Neutral truth particles have **no** 1:1 output counterpart at
all — calo objects are *towers*: energies are summed per tower, thresholds gate the
tower (not the particle), `EFlowMerger` relabels streams, and `PhotonClusterMerger`
merges photons (`SimpleCalorimeter.py:553-566`, `EFlowMerger.py:177,212`,
`PhotonClusterMerger.py:50`). A per-particle Bernoulli likelihood is not defined there.

**Decided (2026-09-09):** the plan replaces the 3 tracking count terms with BCE and
*keeps* the 2 calo count terms. Dropping the calo terms too would resurrect the
wrong-signed-Wasserstein failure mode they were introduced to fix.

**Corollary — matching is a fullsim-only tool, and neutrals are never labeled for BCE:**
neutral reco objects (photons, neutral hadrons) are *created by the calorimeter* —
tower sums crossing significance thresholds — not tracked through per particle, so
there is no per-particle survival coin for a label to record, and no card parameter a
neutral BCE term would fit (their membership gradient is the calo count terms, kept).
Truth-reco matching is therefore NOT needed for pseudodata at all: for the charged
streams the generator preserves row identity through the efficiency stage
(`learnable.py:530-541` zeroes momentum in place, no drop/reorder), so labels are
recorded by identity, not matching. The Phase-2 matcher exists for exactly one case:
fullsim charged streams, where the survival outcomes are real but unrecorded. Rollout is in two
phases — **Phase 1: pseudodata/closure** (we generate the x labels ourselves, see 3a);
**Phase 2: fullsim** (a preprocessing step builds x labels by matching gen to reco
particles, e.g. Hungarian matching, see 3b). `--mode fullsim` keeps the count terms
until Phase 2 lands. `rate_raw` / muon `eff_logits[2,5]` **stay frozen** — they are
frozen because the training phase-space doesn't populate them, which a new gradient
path doesn't change.

## 3. The hard part: the labels `x` do not exist yet

No target file has any truth<->reco link — `TRUTH_BRANCHES`/`PFLOW_BRANCHES`
(`tune_cms_fullsim/config.py:30-37`) are unmatched sets, and the design doc says so
explicitly (`doc/differentiable_delphes.tex:249-250`).

### 3a. Pseudodata (closure): obtainable — regenerate with a survival branch

`generate_pseudodata.py` runs the same card and knows the survival internally, but
discards it (`eflow_to_class_arrays`, `generate_pseudodata.py:341-383`, drops the
pt==0 ghost rows at line 363). Change:

- Tag each truth row with its index (a UID column in `ColumnMap`,
  `src/parnassus/data/particle_io.py:27-57` — columns currently end at
  `EFF_REGION = 20`) inside `truth_arrays_to_pflow` before the card call
  (`generate_pseudodata.py:588`).
- Capture the efficiency-stage mask per species and scatter it back to truth order;
  write it as a jagged bool branch `truth_survived` next to the existing truth branches
  (`generate_pseudodata.py:557-562`). Also write `truth_eff_region` (the target card's
  `EFF_REGION` tag, `CMSDefault.py:452-470`) — see 5.3.
- Label schema (decided 2026-09-09, uniform across modes): **every truth/gen input
  particle gets a label** `x` = "this particle survives reconstruction" (appears in
  the reco output), plus a support flag and a region index:
  - `truth_survived` (bool, every particle),
  - `truth_in_tracker` (bool: reached the efficiency stage, i.e. survived
    `ParticlePropagator`; false ⇒ excluded from the BCE support),
  - `truth_eff_region` (int, valid where `truth_in_tracker`; -1 otherwise) — the
    efficiency region is binned by eta at the *outer tracker radius*, not truth eta,
    so it is stored rather than recomputed on the fit side.
- Support of the BCE expectation in closure (decided): **post-propagation,
  pre-efficiency** particles, i.e. `truth_in_tracker == True`. On that support, in
  delphes mode, "survives reconstruction" is *equal* to the efficiency module's
  Bernoulli draw for the charged streams: downstream never deletes a surviving track
  (arbitration rescales momenta, `SimpleCalorimeter.py:718`; no acceptance cuts, no
  photon merger in delphes mode), and a killed track never re-appears as a track. So
  the uniform output-presence label carries no bias *given the support restriction* —
  restricting the support is what removes the propagation-loss bias; the
  delphes-mode invariant removes the downstream bias. **This invariant must be
  asserted by a generator-side unit test** (coin flip == output presence on the
  support, per charged species), since any future downstream change that deletes
  tracks would silently bias the labels.
- Excluded populations: propagator-dropped particles (x=0 by construction, but the
  geometry loss is not the efficiency's fault — including them re-introduces the
  bias) and reco/post-efficiency objects (conditioned on x=1: a survival probability
  cannot be estimated from survivors alone). Neutrals carry `truth_survived` too for
  bookkeeping, but the BCE never consumes them (see the corollary in Section 2).
- Regenerate all pseudodata samples (`slurm_scripts/submit_pseudodata.sh`) and the
  test fixture — which is already stale (missing `truth_pdgid`; cause of the known
  23-failed baseline of `test_tune_cms_fullsim.py`). One regeneration fixes both.

  **Provenance of the current `run_sequential.sh` samples** (so regeneration
  reproduces them): the four
  `/global/cfs/cdirs/m3246/diff_delphes/allsamples/pseudo_data_200k_param_config_all_<process>.root`
  were produced 2026-08-18 by Runze (NERSC user `mukyu`) from their checkout
  (`/global/u2/m/mukyu/MCGen/torch_delphes/parnassus`, i.e. pre-`4650b39` card code)
  via `submit_pseudodata.sh --config param_configs/param_config_all.yaml
  --process <muongun|electrongun|ksgun|dijet> --n_events 200000` — SLURM CPU arrays
  (shared QOS, 32 cores/task), guns as 2 x 100k tasks (seeds 1-2), dijet as
  20 x 10k tasks (seeds 1-20); the seed drives both Pythia and the target card's
  smearing draws. Logs live in `allsamples/logs/` (job ids 572477xx).
  `allsamples/mix_gun_samples.py` then concatenated + shuffled the four into
  `pseudo_data_all.root` (not used by `run_sequential.sh`). Regeneration cost is
  small: measured ~0.11 s/event for dijet, guns are ~free — well under one
  shared-QOS array per process. Regenerate from OUR branch so the target card
  exactly matches the trainee code.

  **Config identity:** the logged `--param-config` path is Runze's checkout's
  `param_configs/param_config_all.yaml` (checkout unreadable to us), but the file's
  last commit anywhere is `2590faa` (Runze, 2026-08-18 15:07) and the samples were
  generated 16:08-17:49 the same day, after that commit and before any later one —
  so our repo's `param_config_all.yaml` (unchanged since `2590faa`) is the
  generation config, barring uncommitted local edits on Runze's side. Note the
  generator does NOT embed the config in the ROOT output — worth adding a metadata
  branch/key during the same regeneration (supports rule 5 of
  `torch_delphes/CLAUDE.md`).
- Old files stay usable: the reader skips missing branches (`data.py:95-96`), so BCE
  mode just errors cleanly (or falls back) when `truth_survived` is absent.

### 3b. Fullsim (CMS opendata): Phase 2 — a matching preprocessing step

`train_1000.root` has only unmatched truth/pflow sets; `--mode fullsim` changes
acceptance cuts, not the reader (`runner.py:48-68`). The labels must therefore be
*constructed* by a preprocessing step that matches gen (truth) to reco (pflow)
particles per event and defines x = "this truth particle has a reco match":

- **Matcher**: per-event bipartite assignment (Hungarian / `scipy.optimize.linear_sum_assignment`)
  between truth and pflow objects of compatible class/charge, cost =
  deltaR^2 (optionally + a relative-pt penalty), with a max-cost gate so unmatched
  rows stay unmatched (an all-pairs-above-threshold row = x=0; an unmatched pflow
  object = fake, ignored by this loss). Survey what CMS/current literature actually
  uses before committing (deltaR-gated Hungarian is the common baseline; MLPF-style
  learned matching is overkill for a label-construction step).
- **Where**: a standalone preprocessing script (like `mix_gun_samples.py`) that reads
  a fullsim ROOT file and writes it back with `truth_survived` (+ optionally the
  match index and cost) appended — so `data.py` sees the identical branch in both
  modes and the training-side code from 3a/5.x is unchanged.
- **Support and regions in fullsim**: real data has no "pre-efficiency" population,
  but it doesn't need one — the propagator is deterministic and theta-free, so the
  preprocessing step runs OUR `ParticlePropagator` on the gen particles (Section 5.3
  option (a)) to fill `truth_in_tracker` and `truth_eff_region`; every gen particle
  still gets `truth_survived` from the matcher. The fit-side BCE term is then
  byte-identical in both modes.

  **UNRESOLVED — expectation over ALL input particles in fullsim (proposed
  2026-09-09) has a q-support problem.** The trainee's propagator hard-drops charged
  particles whose helix never reaches the tracker radius
  (`ParticlePropagator.py:143-146,178`) — a deterministic pt threshold well above
  the fullsim `truth_pt_cut=0.25`. For such a particle the model's survival
  probability is structurally 0 *regardless of theta* (the efficiency module never
  sees it, and it has no eta-outer region to index q with). But the real detector
  does reconstruct some of them, so the matcher will produce x=1 labels there:
  `BCE(q=0, x=1)` is unbounded, and any finite workaround (clamping, truth-eta
  binning) makes `eff_logits` absorb a loss the forward model still applies on top —
  the double-counting failure: the model's output rate becomes
  P_prop(model) x eff(fitted), doubly suppressed. Recommendation: use the
  `truth_in_tracker` support in fullsim too (uniform with closure). Sub-threshold
  particles are unfixable by ANY value of the efficiency parameters — the model
  cannot produce them — so excluding them from the BCE is honest; their absence
  shows up where it belongs, in the shape/count terms. If coverage of that region
  matters, the correct fix is a propagator/acceptance model change, not an
  efficiency label.
- **Semantic shift (accepted)**: in fullsim the fitted efficiency is the effective
  P(matched reco object | gen particle the model propagates into the tracker) — real
  losses our propagator doesn't model land in it by construction, which is what a
  Delphes efficiency card means. Watch for **double counting**: losses that the
  labels fold in AND the model still applies downstream of the efficiency stage
  (track arbitration, reco-pt-cut migration) would suppress twice. For this card the
  downstream track deletion is ~nil and the reco-pt cut is harmonized, so the
  residual is pt-migration near the cut — measure it in the pseudodata dry-run
  below rather than assuming it.
- **Scope**: charged streams only (chad/electron/muon), same as pseudodata; neutral
  truth has no 1:1 reco counterpart (Section 2).
- **Known label-noise caveats** (accept, and quantify on pseudodata first): fakes,
  merged/overlapping objects, pt migration across the reco-pt cut, and arbitration
  all leak into x, so the BCE estimates *matched-efficiency*, not a pure tracking
  Bernoulli; and unlike 3a the label folds in propagation loss + acceptance, so the
  fullsim q must be interpreted (or corrected) accordingly. Dry-run the matcher on
  pseudodata, where the true labels exist: the confusion matrix of
  matched-vs-true-survival directly measures the label noise and sets the
  systematic on the fitted efficiencies.

Phase order: pseudodata first (3a), fullsim matching second, gated on Phase-1 closure
results.

## 4. Why this is worth it

- Exact likelihood instead of a chi^2 on binned rates: per-particle resolution,
  no region-rate floor (`COUNT_RATE_FLOOR`), no batch-aggregation subtleties.
- With piecewise-constant `q` per efficiency region, the BCE MLE has a closed form:
  `sigmoid(eff_logit_r) -> empirical survival fraction in region r`. This gives (a) an
  analytic cross-check of every fit, and (b) near-convex, fast convergence for the
  efficiency block.
- Decouples efficiency fitting from the reco-space forward: `q` only needs
  `compute_efficiency` on the data truth particles — no dependence on the smearing /
  calo state of the rest of the card, so stage coupling shrinks.
- `rate_raw` (muon > 1 TeV roll-off) becomes trainable for free: BCE differentiates
  through `compute_efficiency`, which the reweighting trick could not (see 5.6).

## 5. Implementation sketch (no code yet)

### 5.1 Data (`tune_cms_fullsim/data.py`)
Read `truth_survived` (+ `truth_eff_region`) in `load_cms_flow_root`; carry them
through `_build_truth_rows` **filtered by the same acceptance `sel` mask**
(`data.py:142-152`) so labels stay index-aligned with truth rows. Expose per-species
`(pt, eta, region, x)` arrays on the target side.

### 5.2 New loss term (`tune_cms_fullsim/loss.py`)
`_bce_eff_terms(...)`: for each of the 3 species, `q = module.compute_efficiency(...)`
(`learnable.py:446-455`; muon override `learnable.py:624`) on the **data truth**
kinematics, `binary_cross_entropy` against x, clamped/eps-guarded. One term per
species so `--pid-weighting` can redistribute across them exactly as it does across
count terms today (`loss.py:830-841`).

### 5.3 The eta_outer question
`compute_efficiency` bins by eta at the **outer tracker radius**, computed by the
propagator inside the card forward — the data-side truth particles haven't been
propagated. Two options: (a) run `ParticlePropagator` (deterministic, theta-free) on
the data truth once at load time; (b) trust the stored `truth_eff_region` from
generation and index `sigmoid(eff_logits)[region]` directly — exact for closure since
target card and trainee share `EfficiencyRegionSpec` (`learnable.py:384-394`).
Prefer (b) for pseudodata (exact, cheap); (a) becomes necessary only for a future
fullsim matcher.

### 5.4 Wiring (`cli.py`, `training.py`, `optuna_search.py`)
- New flag `--eff-loss {counts,bce}` (default `counts` until validated) + `--bce-weight`.
- `bce` mode: skip building the 3 tracking count terms (and their target region-count
  observables) and add the BCE terms in the assembly at `loss.py:1481-1486`; keep the
  calo count terms untouched. The tracking-count machinery (`_expected_reco_counts`,
  `EFF_REGION` tagging) stays in the code until BCE is validated, then can be excised.
- Term weighting: BCE magnitudes (~nats/particle) are not commensurate with the
  Wasserstein terms; start with a dedicated weight calibrated so the efficiency-block
  gradient norms match the current count terms' on a reference batch.

### 5.5 DDP
The BCE reduction must be a global weighted mean: `diff_all_reduce` of
(sum of per-particle BCE, count) per species — same pattern as the count-term reduce at
`loss.py:1855-1868`. Mind the empty-rank rule from `_all_gather_varlen`
(`loss.py:297-315`): a rank whose shard has no muons must still contribute a
graph-connected zero. Extend `test_loss_ddp_gather.py`.

### 5.6 Trainee forward: unchanged
The Gumbel straight-through mask and its `.detach()` (`learnable.py:526`) stay —
sampling is still needed so the *shape/pair* terms see realistic survivor populations;
BCE supplies the efficiency gradient out-of-band. Optionally unfreeze
`MuonTrackingEfficiency.rate_raw` (and revisit muon `eff_logits[2,5]`) in
`full_phasespace_tuning/stage1_muons.yaml` once BCE demonstrably constrains them.

### 5.7 Tests & docs
- New unit test: BCE fit on a toy sample recovers the closed-form per-region survival
  fraction; region-alignment test between stored `truth_eff_region` and the trainee's
  region spec; the delphes-mode invariant test from 3a (coin flip == output presence
  on the `truth_in_tracker` support, per charged species).
- Update `test_tune_cms_fullsim.py` fixture (regenerated pseudodata, see 3a),
  `full_phasespace_tuning/README.md`, and the stage YAML comments.

### 5.8 Size of the Phase-1 (closure) change

Modest and contained — no architectural surgery; roughly 300-500 lines across ~6-8
files, in four independent pieces:

1. **Generator label capture** (the biggest single piece): tag truth rows with a UID
   so identity survives the propagator's row-drop and species split, record
   per-particle (in_tracker, eff_region, survived) during the target-card forward,
   scatter back to truth order, write the three branches. Touches
   `parnassus/data/particle_io.py` (new column), `generate_pseudodata.py`
   (`truth_arrays_to_pflow` + writer), and a small recording hook in the card path
   (`learnable.py` / `CMSDefault.py`). Main risk: any code assuming a fixed row
   width when the UID column is added — the existing test suite gates this.
2. **Reader**: `data.py` — read the three branches, apply the same acceptance `sel`
   mask, group per species. ~50-80 lines.
3. **Loss term**: `_bce_eff_terms` + assembly switch + differentiable weighted-mean
   all-reduce. ~80-120 lines plus tests.
4. **Wiring**: `--eff-loss` / `--bce-weight` flags in `cli.py`/`optuna_search.py`,
   pass the efficiency modules (or their logits) into the loss context. ~40 lines.

Plus the operational cost: regenerate the pseudodata samples and the test fixture
(Section 3a provenance — cheap SLURM CPU arrays). For comparison, this is far less
machinery than the count-term apparatus it replaces (`_expected_reco_counts`
migration-histogram reweighting + region-count observables + their DDP plumbing),
which can be excised once BCE is validated.

## 6. Validation plan

1. Closed-form check: per-region empirical survival fraction from the new branch vs
   the target card's known efficiencies (pure data validation, no training).
2. Single-stage closure: muon gun with `--eff-loss bce`, compare `params_reg.pdf`
   convergence/covariance of `eff_logits` vs the count-term baseline; same for chads.
3. Full `run_sequential.sh` closure in both modes; check no regression in the
   *non*-efficiency parameters (the shape terms lose the count terms' side pressure).
4. DDP invariance: NPROC=1 vs 4 give the same loss trajectory (extend the existing
   gather test).

## 7. Decisions and remaining open items

Decided (2026-09-09):

1. **Calo count terms stay** — BCE cannot cover neutrals/towers.
2. **Two-phase rollout**: Phase 1 pseudodata (labels generated by us, Section 3a);
   Phase 2 fullsim via a gen-reco matching preprocessing step (Section 3b). Count
   terms remain the fullsim efficiency loss until Phase 2.
3. **Label schema (both modes)**: every input particle carries
   `truth_survived` (output presence), `truth_in_tracker`, `truth_eff_region`;
   closure BCE support = `truth_in_tracker` particles, where output-presence equals
   the efficiency coin by the delphes-mode invariant (rationale + required
   invariant test in 3a). Neutrals are labeled for bookkeeping but never consumed
   by BCE (Section 2 corollary).
4. **`rate_raw` / muon `eff_logits[2,5]` stay frozen** — the constraint is
   training-phase-space coverage, not gradient availability.

Still open:

- **Fullsim BCE support**: proposal was "all input particles"; Section 3b flags the
  q-support / double-counting problem with that and recommends the
  `truth_in_tracker` support in both modes. Needs a decision.
- **Matcher choice for Phase 2**: survey the state of the art before committing;
  baseline is deltaR-gated Hungarian assignment (Section 3b). Quantify its label
  noise on pseudodata, where true labels exist.
- **Regeneration**: all `pseudo_data_*` samples and fixtures must be regenerated with
  the new branches (~same cost as the original jobs, Section 3a); embed the
  generation config in the output while we're at it.
- **BCE term weight** relative to the Wasserstein shape terms (Section 5.4) — to be
  calibrated on a reference batch in Phase 1.
