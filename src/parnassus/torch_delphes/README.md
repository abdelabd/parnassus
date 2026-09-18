# `torch_delphes` — Differentiable Detector Tuning

A CMS-like detector simulation in PyTorch, tuned with Adam: generate pseudodata
with known truth parameters, fit a learnable card back to it (closure), or fit
real CMS full-sim data. Modes: `--mode delphes` (pseudodata) and `--mode fullsim` (real data). 
In `delphes` mode, can toggle `--existence {counts,bce}` to choose between object-counts or BCE
for object-existence loss term. 

All commands run from the repo root on a compute node
(`salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node=4 --gpus-per-node=4 -A m3246_g`)
after `source parnassus_env/bin/activate`. For Comet logging, additionally set your
`COMET_API_KEY`  and `COMET_WORKSPACE` environment variables. 

 Deeper docs: `CLAUDE.md` (ops),
`EFF_LOSS_PLAN.md` (BCE design + results), `CONSOLIDATE_MODES_PLAN.md` (modes).

## 1. Generate pseudodata

Plain samples (truth + pflow branches only; SLURM array + dependent merge, from a
login node), e.g. the four sequential-closure samples of the `param_config_all`
truth card (guns 2x100k, dijet 20x10k):

```bash
CFG=src/parnassus/torch_delphes/param_configs/param_config_all.yaml
for p in muongun electrongun ksgun; do
  OUTBASE=$DD/allsamples bash src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh \
      --config $CFG --process $p --n_events 200000 --n_tasks 2
done
OUTBASE=$DD/allsamples bash src/parnassus/torch_delphes/slurm_scripts/submit_pseudodata.sh \
    --config $CFG --process dijet --n_events 200000 --n_tasks 20
```

The BCE efficiency loss needs no labeled sample set: the truth<->reco survival
labels are built in memory by Hungarian matching right before training.

## 2. Sequential closure — BCE, chad parameters only receive chad gradients in stage 2
```bash
D=src/parnassus/torch_delphes/full_phasespace_tuning
OUT_BASE=doc/pseudo_seq_bce_det \
COMET_NAME_PREFIX=pseudo_bce_det \
EXTRA_ARGS="--existence bce" \
bash $D/run_sequential.sh
```

## 3. Sequential closure — BCE, chad parameters additionally receive calo gradients in stage 3

```bash
D=src/parnassus/torch_delphes/full_phasespace_tuning
OUT_BASE=doc/pseudo_seq_bce_live \
COMET_NAME_PREFIX=pseudo_bce_live \
EXTRA_ARGS="--existence bce --calo-bce-grads live" \
bash $D/run_sequential.sh $D/stage1_muons.yaml $D/stage2_chads.yaml $D/stage3_calo_joint.yaml $D/stage4_electrons.yaml
```



## 4. Sequential closure — counts

```bash
D=src/parnassus/torch_delphes/full_phasespace_tuning
OUT_BASE=doc/pseudo_seq_counts \
COMET_NAME_PREFIX=pseudo_counts \
EXTRA_ARGS="--existence counts" \
bash $D/run_sequential.sh
```

(Counts needs no survival labels — the default unlabeled sample pattern.)

## 5. Fullsim fit

```bash
python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
  --root-file /global/cfs/cdirs/m3246/diff_delphes/cms_opendata_zenodo/train_1000.root \
  --optuna-config src/parnassus/torch_delphes/param_configs_fullsim/optuna_config_ptbins.yaml \
  --n-events 100000 --n-steps 100 --n-trials 1 --loss wasserstein_1d \
  --output-base doc/fullsim_counts --history-path doc/fullsim_counts/all_optuna.json \
  --mode fullsim --reco-pt-cut 5 --pid-weighting fraction
```

## 6. HZZ4l eval (closure PDF for any fitted card)

```bash
python -m parnassus.torch_delphes.full_phasespace_tuning.compare_sample \
  --sample /global/cfs/cdirs/m3246/diff_delphes/allsamples/pseudo_data_100k_param_config_all_HZZ4l_truth_matched_survival.root \
  --fitted-config <OUT_BASE>/fitted_config.yaml \
  --output <OUT_BASE>/distributions_HZZ4l.pdf --device cuda
```

Outputs per run: per-stage `history.json`, `params_reg.pdf` + observable plots,
and the final `fitted_config.yaml` under `<OUT_BASE>/`.
