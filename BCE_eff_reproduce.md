# Reproducing the `BCE_eff` results

Exact commands to recreate everything from the 2026-09-09/10 overnight run:
labeled pseudodata, both sequential closures, and the HZZ4l comparisons. Design
docs: `src/parnassus/torch_delphes/EFF_LOSS_MOTIV.md` (why) and
`EFF_LOSS_PLAN.md` (step-by-step, with the measured results inlined).

Everything below runs from the repo root on a Perlmutter login node, on branch
`BCE_eff` (the generation-side truth card is
`src/parnassus/torch_delphes/param_configs/param_config_all.yaml`, unchanged
since commit `2590faa`; each generated file carries a `.provenance.json` sidecar
recording the exact command, seed, config text and git commit that made it).

```bash
cd /pscratch/sd/a/aelabd/parnassus && git checkout BCE_eff
source parnassus_env/bin/activate     # only needed for the login-node steps
```

## 1a. Pseudodata with truth-matched (generator) survival labels

One script submits all seven sample sets as SLURM arrays + dependent merges
(guns 2x100k tasks, dijet 20x10k, HZZ4l 10x10k; outputs land as
`*_truth_matched_survival.root` next to the originals, which are never touched):

```bash
bash src/parnassus/torch_delphes/slurm_scripts/submit_truth_matched_survival_samples.sh
```

That is: the four sequential-closure samples (muongun / electrongun / ksgun /
dijet, 200k each, `param_config_all`), HZZ4l (100k, `param_config_all`), and the
two single-stage samples (`param_config_muons`+muongun,
`param_config_chads`+ksgun, 200k each). A single sample by hand, e.g.:

```bash
REPO=$PWD ENV_PREFIX=$PWD/parnassus_env \
OUTBASE=/global/cfs/cdirs/m3246/diff_delphes/allsamples \
MERGED_NAME=pseudo_data_200k_param_config_all_muongun_truth_matched_survival.root \
    bash src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh \
    --config src/parnassus/torch_delphes/param_configs/param_config_all.yaml \
    --process muongun --n_events 200000 --n_tasks 2
```

(`REPO`/`ENV_PREFIX` overrides are REQUIRED: the script's defaults point at
Runze's checkout. The labels come from `generate_pseudodata.py` itself — UID
column + forward hooks on the efficiency modules; a hard invariant check
[efficiency mask == presence in the EFlowObject output] runs on every
generation, so a completed job certifies its own labels.)

Closed-form validation of any labeled file (per-region survival fraction must
match the truth card): see the snippet in `EFF_LOSS_PLAN.md` step 3, or trust
`pytest src/parnassus/tests/test_survival_labels.py`.

## 1b. Pseudodata with Hungarian-matched survival labels

After the four sequential `_truth_matched_survival` merges finish (chain it with
`--dependency=afterok:<the 4 merge jobids>`):

```bash
sbatch src/parnassus/torch_delphes/slurm_scripts/hungarian_match_samples.sbatch
```

This writes `*_hungarian_matched_survival.root` for the four sequential samples:
`truth_survived` is REPLACED by per-event, per-class deltaR-gated (0.05)
Hungarian truth<->reco matching; the generator labels are preserved as
`truth_survived_generator`; a `.confusion.json` sidecar per file records the
matcher-vs-generator confusion matrix (2026-09-10 result: agreement 1.0,
fp = fn = 0 on every sample and species). One file by hand:

```bash
python -m parnassus.torch_delphes.hungarian_survival_matching \
    --input  /global/cfs/cdirs/m3246/diff_delphes/allsamples/pseudo_data_200k_param_config_all_dijet_truth_matched_survival.root \
    --output /global/cfs/cdirs/m3246/diff_delphes/allsamples/pseudo_data_200k_param_config_all_dijet_hungarian_matched_survival.root
```

## 2. Sequential tunings

All three closures are `run_sequential.sh` under different sample patterns; the
BCE efficiency loss is on automatically (`--mode delphes` defaults to
`--eff-loss bce`). Via the sbatch wrapper (regular GPU queue; chain behind the
sample merges with `--dependency`):

```bash
# (a) BCE on generator labels -> doc/figure_sequential_truth_matched_survival
TAG=truth_matched_survival sbatch --export=ALL,TAG \
    src/parnassus/torch_delphes/slurm_scripts/run_sequential_survival.sbatch

# (b) BCE on Hungarian labels -> doc/figure_sequential_hungarian_matched_survival
TAG=hungarian_matched_survival sbatch --export=ALL,TAG \
    src/parnassus/torch_delphes/slurm_scripts/run_sequential_survival.sbatch
```

Or interactively (what the overnight run actually did — faster queue):

```bash
salloc --no-shell -C gpu -q interactive -t 180 --nodes 1 --ntasks-per-node=1 \
    --gpus-per-node=4 -c 128 -A m3246_g          # note the printed JOBID
D=src/parnassus/torch_delphes/full_phasespace_tuning
srun --jobid=<JOBID> -N1 -n1 --gpus-per-node=4 -c 128 --export=ALL,\
SAMPLE_DIR=/global/cfs/cdirs/m3246/diff_delphes/allsamples,\
SAMPLE_PATTERN=pseudo_data_200k_param_config_all_%s_truth_matched_survival.root,\
OUT_BASE=$PWD/doc/figure_sequential_truth_matched_survival,\
COMET_NAME_PREFIX=tms,COMET_API_KEY=<key>,COMET_WORKSPACE=abdelabd \
    bash $D/run_sequential.sh
```

Pipelined variant (stages 1-2 before the dijet sample exists, 3-4 after — the
sbatch wrapper takes the same `STAGES` / `FROM_HISTORY` env):

```bash
# stages 1-2 now:
... bash $D/run_sequential.sh $D/stage1_muons.yaml $D/stage2_chads.yaml
# stages 3-4 once the dijet merge lands:
FROM_HISTORY=$PWD/doc/figure_sequential_truth_matched_survival/stage2_chads/round_0/history.json \
    ... bash $D/run_sequential.sh $D/stage3_calo.yaml $D/stage4_electrons.yaml
```

(c) The count-term BASELINE (`doc/figure_sequential_dd/`) is plain
`run_sequential.sh` on the ORIGINAL unlabeled samples — on this branch you must
now say so explicitly, since delphes mode defaults to bce and hard-errors on
unlabeled files:

```bash
EXTRA_ARGS="--eff-loss counts" OUT_BASE=$PWD/doc/figure_sequential_dd_repro \
    bash $D/run_sequential.sh
```

## 3. HZZ4l comparisons (independent-sample closure of each fitted card)

CPU-only, login node, ~40 min each for 100k HZZ4l events; one PDF each (target
vs CMS-default "initial" vs "tuned": per-species kinematics, pair-mass
responses, m_ee / m_mumu / m_4l):

```bash
S=/global/cfs/cdirs/m3246/diff_delphes/allsamples/pseudo_data_100k_param_config_all_HZZ4l_truth_matched_survival.root
for tag in truth_matched_survival hungarian_matched_survival dd; do
    python -m parnassus.torch_delphes.full_phasespace_tuning.compare_sample \
        --sample $S \
        --fitted-config doc/figure_sequential_$tag/fitted_config.yaml \
        --output doc/figure_sequential_$tag/distributions_HZZ4l.pdf
done
```

(The same labeled HZZ4l sample serves all three — `compare_sample` ignores the
label branches, and the generating truth card is identical.)

## 4. Everything else from the overnight run

- **Test-fixture regeneration** (made `test_tune_cms_fullsim.py` 47/47 green;
  ALWAYS pass `--n-workers` on shared nodes — auto-detect sees all 256 logical
  CPUs and OOMs a 32-core slice):

  ```bash
  sbatch -A m3246 -C cpu -q shared -N1 -n1 -c 32 -t 00:55:00 --wrap \
    "source $PWD/parnassus_env/bin/activate && cd $PWD && \
     python -m parnassus.torch_delphes.generate_pseudodata \
       --output src/parnassus/tests/benchmark_data/cms_pseudodata.root \
       --n-events 5000 --process dijet --seed 1 --debug --n-workers 32 --device cpu"
  ```

- **Single-stage validation runs** (BCE-vs-counts on one stage; how the stage-8
  gates were measured): run `run_sequential.sh` with a single stage YAML into a
  scratch `OUT_BASE`, once as-is (bce) and once with
  `EXTRA_ARGS="--eff-loss counts"`, then compare
  `<OUT_BASE>/<stage>/round_0/history.json` `best_result.parameters` against
  `param_config_all.yaml`.

- **Tests** (all green; the gates for this work):

  ```bash
  pytest src/parnassus/tests/test_survival_labels.py \
         src/parnassus/tests/test_bce_eff_loss.py \
         src/parnassus/tests/test_loss_ddp_gather.py \
         src/parnassus/tests/test_torch_delphes_learnable.py \
         src/parnassus/tests/test_tune_cms_fullsim.py
  ```

- **Loss knobs** (see `tune_cms_fullsim --help`): `--eff-loss {counts,bce}`,
  `--bce-weight` (default 1.0, calibration in `EFF_LOSS_PLAN.md` step 7),
  `--bce-weighting {pooled,per_species}` (pooled = the exact joint likelihood).

- **Results summary**: `NEXT_DAY.md`. Per-run reproduce files also sit in each
  output directory (`reproduce.sh`) and next to the samples
  (`allsamples/reproduce_truth_matched_survival.sh`).
