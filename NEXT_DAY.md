# NEXT_DAY — BCE efficiency loss, overnight 2026-09-09/10 (branch `BCE_eff`)

**TL;DR: all 7 steps of your plan are done, and the BCE loss wins.** The full
sequential closure on survival-labeled pseudodata recovers the tracking
efficiencies **2.5–3x closer to truth** than the count-term baseline, with every
other parameter block statistically unchanged. The Hungarian-matching rehearsal
introduced **zero label noise**, so the matcher-labeled closure reproduces the
truth-labeled one exactly.

## What ran (your step numbering)

0. Baseline archived: `doc/figure_sequential_dd/` (your count-term run).
1. **Generator**: new `ColumnMap.UID` rides through the card forward; forward
   hooks on the efficiency modules record region + Bernoulli outcome; three new
   branches (`truth_survived`, `truth_in_tracker`, `truth_eff_region`) + a
   `.provenance.json` sidecar (command/seed/config text/git commit) per file. A
   hard invariant check (mask == presence in output) runs on every generation.
2. **Samples**: all regenerated with labels as `*_truth_matched_survival.root`
   next to the originals — 4 sequential (200k each), HZZ4l (100k), both
   single-stage samples. Labels validated closed-form: per-region survival
   fractions match the truth card within 1.5 sigma everywhere (11.7M chads in
   dijet alone).
3. **Training code**: `--eff-loss {counts,bce}` (bce = default in delphes mode),
   `--bce-weight`, `--bce-weighting {pooled,per_species}` (pooled = exact joint
   likelihood, per your toggle request). Tracking count terms dropped in bce
   mode; calo count terms kept. DDP value/gradient equivalence proven by a new
   gloo test incl. an empty rank. `bce_weight=1.0` kept after gradient-norm
   calibration (BCE 3–17x stronger on eff_logits; Adam normalizes).
4. **tms closure** → `doc/figure_sequential_truth_matched_survival/`:
   median |rel err| vs truth (BCE vs counts): chad eff **0.0034 vs 0.0108**,
   muon eff **0.0019 vs 0.0049**, electron eff 0.0060 vs 0.0057; calo +
   smearing blocks identical (recoverable params to the 3rd decimal). Ran
   pipelined: stages 1–2 as soon as their samples merged, 3–4 resumed from
   stage-2 history after the dijet merge.
5. **Hungarian matcher** (`hungarian_survival_matching.py`): per-event,
   per-class deltaR-gated assignment; generator labels kept as
   `truth_survived_generator`; confusion-matrix sidecars written.
6. **hms samples**: `*_hungarian_matched_survival.root` (4 sequential samples).
   Confusion matrices: **agreement 1.0, fp=0, fn=0 on all species/samples** —
   at delphes fidelity the smearing preserves direction, so matches sit at
   deltaR≈0. Real fullsim will add angular smearing/fakes; re-measure there.
7. **hms closure** → `doc/figure_sequential_hungarian_matched_survival/`:
   identical to tms to 4 decimals on every efficiency block, as the perfect
   labels predict.

## Bonus / housekeeping

- **`test_tune_cms_fullsim.py` is 47/47 green**: the stale fixture was
  regenerated (5k dijet + debug + labels; old file kept as
  `cms_pseudodata_no_labels.root`). The 23-failure baseline is retired —
  CLAUDE.md + memory updated. One real bug it caught was fixed (dense loader
  needed the bce keys too).
- New tests: `test_survival_labels.py`, `test_bce_eff_loss.py` (closed-form MLE
  recovery, pooled==flat identity, gradient routing, assembly switch), BCE DDP
  case in `test_loss_ddp_gather.py`. Pre-existing failures elsewhere (12,
  verified on the unmodified tree) are listed in CLAUDE.md.
- Comet: runs logged as `tms_stage1_*`, `tms_*`, `hms_*` with per-epoch `bce`
  breakdown components. Note: BCE-mode val losses carry a ~0.65-nat Bernoulli
  entropy floor — don't compare values across modes.
- CLAUDE.md: added "pipeline with sbatch --dependency" doctrine (your ask) and
  the --eff-loss section.
- HZZ4l closure plot (`compare_sample.py` on the tms fitted card) was launched
  last: `doc/figure_sequential_truth_matched_survival/distributions_HZZ4l.pdf`
  (check it exists; it was still rendering at write-up time).

## Hiccups (all resolved)

- My ad-hoc fixture sbatch omitted `--n-workers`: the generator auto-spawned 256
  Pythia workers on a 32-core shared slice → OOM. Resubmitted with
  `--n-workers 32`; the production arrays always passed it.
- Two session restarts killed background monitors/sruns; the SLURM chains were
  unaffected and the interrupted hms run had already completed.

## Suggested next

1. Eyeball `params_reg.pdf` in the two new closure dirs vs `figure_sequential_dd`.
2. Decide on flipping any remaining docs/defaults, then the Step-10 excision of
   the tracking-count machinery (kept in code, off in bce mode).
3. Phase 2 (fullsim): the support decision (EFF_LOSS_MOTIV 3b) + run the matcher
   preprocessing on `train_1000.root` with the propagator-derived support.
4. The Comet API key sits in the committed CLAUDE.md — swap for
   `$(cat ~/.comet_api_key)` before this branch is ever pushed.

12 commits on `BCE_eff` (f07929d..), nothing pushed, `diff_delphes` untouched.
