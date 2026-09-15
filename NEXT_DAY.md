# NEXT_DAY — HCal-scale debug, overnight 2026-09-14/15 (branch `BCE_eff`)

**TL;DR: root cause found, fixed, and closure-validated. The tower BCE was
evaluating its significance thresholds at the *sampled* smeared energy; the
hard cut is self-consistent (`E_sm > S·σ(E_sm)`), so the correct threshold is
the fixed point `E* = S·σ(E*)`. With `--calo-bce-threshold self_consistent`
plus marginal conditioning, the HCal scales land at ~1% — matching the chi²
count terms — and the ECal scales get 4–6× better than either baseline.**

## Why counts worked and BCE didn't

Your question, answered: the count chi² compares *realized* survival counts,
so any consistent q enters symmetrically and its minimum sits at truth (method
of moments). The BCE is a *likelihood*, so a mis-specified q shifts its
minimizer (KL argmin ≠ truth). The mis-specification: we plugged
σ_after(this draw's sampled E_sm) into the tail probability as a fixed
threshold. The true pass event for `E_sm > S·σ(E_sm)` (σ increasing in E) is
`E_sm > E*` at the unique fixed-point crossing. Frozen-truth scans confirmed a
value-level pseudo-truth: TowerBceHcal was monotone toward *low* scale in
region 0 (no interior minimum) while the count chi² minimized exactly at
truth. The fitted 0.75/0.75 was the equilibrium against the shape terms. The
optimizer, DDP, and per-region gradients were all exonerated first
(finite-difference-consistent, distinct per region).

## Evidence chain (all in `doc/hcal_bce_debug/` + EFF_LOSS_PLAN.md)

1. **MC q\* calibration** (150 replicas × 256 dijet events at truth): legacy
   thresholds over-predict q by **+0.030 on trackless towers** (+0.071 near
   threshold); the self-consistent fix takes trackless bias to **+0.000
   exactly** (the lognormal marginal + exact threshold is exact there) and
   halves near-threshold RMS. Residual −0.02..−0.06 bias remains only on
   track-carrying towers = the expected-conditioning (ii) residual.
2. **Frozen-truth scans**: region 1 (forward, trackless) minimum returns to
   truth (0.90 vs 0.893); region 0 (central, track-rich) gains an interior
   minimum with a high-side shift — the conditioning residual again.
3. **Stage-3 closures** (from ii-a's stage-2 history; median |rel err|):

   | block | chi² (dd) | ii-a legacy | ii+selfcon | **iii+selfcon** |
   |---|---|---|---|---|
   | ECal scales | 0.0092 | 0.0113 | 0.0018 | **0.0022** |
   | HCal scales | 0.0136 | 0.1374 | 0.0506 | **0.0168** |

   iii+selfcon fits HCal **0.8650/0.9063 vs truth 0.8487/0.8934**. ii+selfcon
   nails region 1 but keeps the region-0 tracked-tower bias (0.924 vs 0.849) —
   so **marginal conditioning (iii) matters after all** once the thresholds
   are right; the earlier "(iii)≈(ii)" verdict was an artifact of the broken
   thresholds. HCal *resolutions* look worse than ii-a (0.94 vs 0.37) but both
   beat the chi² reference (1.39): that block is ill-determined everywhere,
   and ii-a's "good" res was absorbing its mis-fitted scales.
   HZZ4l PDFs are in both closure dirs.

## Other results from the night

- **Live-gradient arms (b) formally rejected**: the joint+detach control keeps
  chad eff at 0.0033 vs ii-b's 0.0077 with the same trainable set — the
  degradation is the live gradients, not the joint config.
- **Full-LR pass-2** (`..._detach_pass2`): ECal scales 0.0113→0.0071, chad
  smear 0.054→0.031, but chad eff 0.0014→0.0088 and HCal untouched. Not the
  fix; superseded by selfcon anyway.
- **Gotcha (pre-existing, important):** `card(input)` **mutates its input
  in place** (10 columns). Training is safe (`tp[mask]` copies), but analysis
  scripts looping `card(flat)` feed each forward the previous one's mutated
  input — this corrupted the first MC calibration. Always pass
  `flat.clone()`; with that the forward is exactly deterministic per seed.

## Code (committed on `BCE_eff`, not pushed)

- `2edd917`: `tower_bce_threshold {sampled_sigma,self_consistent}` on
  SimpleCalorimeter/CMSDefault + `--calo-bce-threshold` CLI; 25 detached-energy
  contraction iterations (σ_after_c gradient convention: resolution
  coefficients live); bce diagnostic exports; MC/scan scripts + logs; plan
  update. Gate tests green (26/26).
- Second commit (this morning): closure results in EFF_LOSS_PLAN.md, the two
  closure dirs' reproduce.sh + HZZ4l PDFs, this file.
- **Default is still `sampled_sigma`** — flipping it (and making
  `marginal` the default conditioning) is your call.

## Suggested morning decisions

1. Adopt `--calo-bce-threshold self_consistent --calo-bce-conditioning
   marginal` as the champion setting (flip defaults?) and rerun the full
   4-stage sequential + pass-2 with it.
2. HCal res identifiability: accept as ill-posed, or add a dedicated probe?
3. Fix or fence the `card(input)` in-place mutation (a `.clone()` at the
   forward entry would cost one copy and remove the landmine).
4. Phase 3 (fullsim) is now unblocked if you call the neutral BCE done.

All allocations released. Runs: `doc/figure_seq_hung_nBCE_cond{ii,iii}_selfcon`
(each with exact-command `reproduce.sh`), debug artifacts in
`doc/hcal_bce_debug/` (`mc_v2.log`, `scan_v2.log`, scripts, scorer).
