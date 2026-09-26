# eff_fix_plan — efficiency-then-smearing + truth-bin BCE

Branch: `eff_before_smearing` (HEAD `5ff33c2`). Status: APPROVED 2026-09-25. Local commits OK (no push). Benchmark only A (points 1+2 together) and B. Generation via sbatch, charged to m3246.

## 0. Why (the diagnosis, 2026-09-25)

The 2b8fcb7 regression is region migration, not matching (matching is identical, see
`compare_matching.py`). The learnable card samples each track's survival from
`eff(bin(smeared pT))`. Since 2b8fcb7, the BCE's eps is evaluated on the *trainee's own,
independent* smearing draw. Near a pT edge (1 GeV; 100 GeV for electrons) that draw often
lands in a different bin than the one that drew the label, so every efficiency is pulled
toward its pT-neighbour. A per-region fixed-point calculation (scratchpad
`region_migration.py`, run at truth params) reproduces the 2b8fcb7 fits to within ~0.01.
Example: electron eff[0] is predicted 0.456 and fitted 0.468, against a truth of 0.377.

The bin that actually drew a killed track's survival can't be recovered from the plain
sample, and fullsim will never have it. So the fix is to make the model's survival decision
depend only on truth-level kinematics, as C++ Delphes does (`Efficiency` before
`MomentumSmearing`) and as `diff_delphes_luigi` models it. Then the bin is computable in
training from the truth row alone.

Smear-then-mask was introduced in `bf028c5` for a mechanical reason only. The Gumbel mask
zeroes momentum columns but keeps the row, and `MomentumSmearing` recomputes
`E = sqrt(p^2 + m^2)`. So a mask applied *before* smearing would bring killed tracks back at
their rest mass. Evaluating the efficiency on smeared pT was an unintended side effect of
that reorder.

## 1. Code changes (approach A: decide on truth, apply after smearing)

Each step is one local commit on `eff_before_smearing` (no push). Each commit's hash names
its benchmark directory.

### Commit A — points 1+2: survival decided on unsmeared kinematics
- `learnable.py`, `_LearnableEfficiencyBase.forward(particles, kin=None)`: the efficiency
  (and hence the mask) is computed from `kin` (default `particles`, so existing callers are
  unchanged). The mask is still multiplied onto `particles`.
- `CMSDefault.py`, learnable branch: pass `kin=<species>_propagated` to the three
  efficiency modules, which still receive the smeared rows. `MomentumSmearing` keeps rows
  1:1 (clone + index_put), so the rows stay aligned. `_tag_eff_region` also takes the
  propagated rows, so the `EFF_REGION` tag names the bin the mask actually used. Update
  the order comment. This preserves the `bf028c5` fix, since the mask is still applied
  last.
- Deliberately **not** changed in A: `TrackSurvivalExport`, which still holds eps on the
  smeared kinematics. It feeds both the track BCE and the tower-BCE conditioning. This
  isolates point 3 into commit B.
- Tests: update/extend `test_torch_delphes_learnable.py`. The mask probability must be a
  function of the propagated (pT, eta_outer), and a killed track must have
  E = pT = p = 0 after the chain. Then run the full `src/parnassus/tests` suite.
  Check 1x1 then 1x4 on a short run, per the CLAUDE.md loss-change rule.
- Docs: update the order/semantics comments in `CMSDefault.py`, `learnable.py`, and the
  torch_delphes CLAUDE.md "One efficiency function" bullet.

Commits A and B also revise the docstrings and comments that describe the changed behaviour.

### Commit B — point 3: BCE eps on the truth bin
- `CMSDefault.py`: `TrackSurvivalExport` evaluates `compute_efficiency` on the propagated
  rows, i.e. exactly the function the mask used. No loss or training code changes: pairing
  by UID with the Hungarian labels stays as it is.
- Consequence: the tower-BCE marginal conditioning (`track_cond` eps) moves to the same,
  now-correct survival probability, because it shares the export. The track energy
  `e_pre` stays the smeared pre-mask E. This is intended: one survival probability per
  track.
- Tests: in the forward, the exported eps equals the efficiency on propagated (pT,
  eta_outer). Update `test_bce_eff_loss.py` / `test_loss_ddp_gather.py` expectations if
  they pin smeared kinematics.

## 2. New pseudo-data (needed once, for A and B)

The existing samples drew survival from the smeared-pT bin, so they are the wrong target
for the new model. Regenerate with **commit A's** `generate_pseudodata.py` (the target is
the learnable card, which carries the change). Use the recipe recorded in the provenance
sidecars (`param_config_all.yaml`, `--n-workers 32`, `--device cpu`) with the same
seed/task layout, so the Pythia truth events are identical to the current plain samples:

| sample | layout (tasks x events, seeds) |
|---|---|
| muongun, electrongun, ksgun | 2 x 100k, seeds 1-2 |
| dijet | 20 x 10k, seeds 1-20 |
| HZZ4l (eval) | 10 x 10k, seeds 1-10 |

- Output: **new files only, nothing is overwritten.** They go in the new sister directory
  `/global/cfs/cdirs/m3246/diff_delphes/allsamples_09_25_2026/`, with the same filenames as
  the originals (`pseudo_data_200k_param_config_all_<proc>.root`,
  `pseudo_data_100k_param_config_all_HZZ4l.root`). Before submitting, the
  submitter refuses to run if a target filename already exists.
- How: `slurm_scripts/submit_pseudodata.sh` (CPU `shared` QOS array plus an `afterok`
  merge job), one submission per process, with these environment overrides:
  - `REPO=<worktree of commit A>`, with `PYTHONPATH=<worktree A>/src` exported, so the
    array tasks import commit A's card and not the editable install's working tree;
  - `ENV_PREFIX=/pscratch/sd/a/aelabd/parnassus/parnassus_env`;
  - `OUTBASE=.../allsamples_09_25_2026`, `PARTS_DIR=.../allsamples_09_25_2026/parts`;
  - `MERGED_NAME=<name above>`;
  - `TIME_LIMIT`: 01:00:00 for dijet/HZZ4l tasks, 02:00:00 for 100k-event gun tasks.

  Example (dijet):
  ```bash
  REPO=$WT_A PYTHONPATH=$WT_A/src ENV_PREFIX=/pscratch/sd/a/aelabd/parnassus/parnassus_env \
  OUTBASE=/global/cfs/cdirs/m3246/diff_delphes/allsamples_09_25_2026 \
  PARTS_DIR=/global/cfs/cdirs/m3246/diff_delphes/allsamples_09_25_2026/parts \
  MERGED_NAME=pseudo_data_200k_param_config_all_dijet.root TIME_LIMIT=01:00:00 \
  bash $WT_A/src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh \
    --config $WT_A/src/parnassus/torch_delphes/param_configs/param_config_all.yaml \
    --process dijet --n_events 200000 --n_tasks 20
  ```
  Fallback, if the shared queue is slow: run the same `generate_pseudodata` commands on
  the interactive nodes' CPUs, with the same seeds and names.
- Validation:
  - (a) truth branches are bitwise identical to the current plain samples (the
    `compare_matching._assert_same_events` check). If they are not, report it and continue,
    since the benchmarks remain valid on their own.
  - (b) the fixed-point check on the new samples, using `region_migration.py` adapted to
    the truth bin. Under the new model, the truth-bin fixed point must equal truth, and the
    smeared-bin one shows the bias expected for commit A. This sanity check runs before any
    2 h training.

## 3. Benchmarks

Hyperparameters are identical to the existing git_forensics benchmarks: `run_sequential.sh`
defaults (4 stages, `NPROC=4`, 100 epochs, early stop 10, pick best, `--lr 1`,
`wasserstein_1d`, `--pid-weighting sqrt_fraction`) and `EXTRA_ARGS="--existence bce"`. This
gives in-training Hungarian matching, deltaR <= 0.05, and tower-BCE detach.

```bash
COMMIT=<hash>   # A, then B
D=<worktree_of_COMMIT>/src/parnassus/torch_delphes/full_phasespace_tuning
REPO=/pscratch/sd/a/aelabd/parnassus PYTHONPATH=<worktree_of_COMMIT>/src \
OUT_BASE=runs/git_forensics/pseudo_seq_bce_det_inspecies_hungarian/$COMMIT \
SAMPLE_DIR=/global/cfs/cdirs/m3246/diff_delphes/allsamples_09_25_2026 \
SAMPLE_PATTERN=pseudo_data_200k_param_config_all_%s.root \
COMET_NAME_PREFIX=$COMMIT EXTRA_ARGS="--existence bce" \
bash $D/run_sequential.sh

python -m parnassus.torch_delphes.full_phasespace_tuning.compare_sample \
  --sample /global/cfs/cdirs/m3246/diff_delphes/allsamples_09_25_2026/pseudo_data_100k_param_config_all_HZZ4l.root \
  --fitted-config runs/git_forensics/pseudo_seq_bce_det_inspecies_hungarian/$COMMIT/fitted_config.yaml \
  --output runs/git_forensics/pseudo_seq_bce_det_inspecies_hungarian/$COMMIT/distributions_HZZ4l.pdf --device cuda
```

- Each benchmark runs from a **git worktree of its own commit** on `/pscratch`, so editing
  B cannot leak into A's run. `PYTHONPATH` shadows the editable install; verify with
  `python -c 'import parnassus; print(parnassus.__file__)'` before launching.
  `REPO` points at the main checkout, only for the untracked `setup.sh` and the truth
  config.
- A `reproduce.sh` goes into each output dir, per CLAUDE.md §6.
- The eval uses the regenerated HZZ4l sample, because the fitted card is now a
  different model from the old HZZ4l target.

## 4. Schedule / pipelining (two 1x4 interactive nodes, 4 h wall each, re-allocated as they expire)

1. Node 1: implement commit A plus its tests (1x1, then 1x4), then commit A and create its
   worktree.
2. Submit all five generation chains via `submit_pseudodata.sh` right after commit A.
   They run in the CPU `shared` queue and don't occupy the GPU nodes.
3. As soon as the guns are merged and pass the check, node 1 (GPU) starts benchmark A
   stages 1–2 (muons, chads, about 40 min), while dijet is still generating. Stage 3 starts when dijet is merged, via `FROM_HISTORY`.
4. Meanwhile implement and test commit B, then commit it and create its worktree. Node 2
   runs benchmark B end-to-end once the samples exist.
5. HZZ4l evals run at the tail of each run.
6. Summary: `runs/git_forensics/EFF_FIX_RESULTS.md`, a fitted-vs-truth table for all
   tracking-efficiency parameters (plus all other blocks) across 28f4aee, 2b8fcb7, A and
   B, with the fixed-point predictions, what changed in the non-efficiency parameters
   (the training-dynamics question), and pointers to the PDFs.

Expected outcome:
- A: the survival model is now right, but eps still comes from the trainee's smeared pT,
  so a migration bias remains (a different size and sign than 2b8fcb7).
- B: efficiencies recover truth like 28f4aee. It needs no generator information at all,
  so it applies to fullsim too.

## 5. Provenance

Every command that produces a result is written down verbatim in
`/pscratch/sd/a/aelabd/parnassus/NEXT_DAY_2026-09-25.md`, as it is run. That covers
sample generation (each `submit_pseudodata.sh` invocation plus its job ids), merges,
validation checks, each training run, each HZZ4l eval, the commit hashes, the worktree
paths, and the allocation commands. Each benchmark dir also gets a `reproduce.sh`. The
diagnostic scripts from the investigation (`region_migration*.py`) are copied from the
node-local scratchpad to `runs/git_forensics/diagnostics/` so they survive.

## 6. Not in scope
- Pushing; overwriting or deleting any existing sample or run directory.
- Fullsim-specific label semantics (post-tracking losses such as reco cuts, misID, or ΔR
  gate misses count as "killed"). Noted for later.
- Changing any hyperparameter.
