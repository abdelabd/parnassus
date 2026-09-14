# Parnassus-P

## Installation

Requirements: bash, git and curl. `uv` and a matching Python (3.12) are fetched
automatically if missing.

```bash
git clone <this repo> && cd parnassus
source setup.sh
```

`source setup.sh` (not `bash setup.sh`) installs the pinned environment from `uv.lock`
and activates it. On first use it asks for four directories and saves them in the
git-ignored `.config`:

| key | meaning |
|---|---|
| `ENV_PREFIX` | this clone's virtualenv (unique per clone; several GB — on NERSC use `$SCRATCH` or CFS, not `$HOME`) |
| `UV_CACHE` | uv download cache (can be shared by all your clones) |
| `UV_PYTHON_DIR` | where uv puts the Python it downloads if none matches on `PATH` |
| `SAMPLE_DIR` | where `prepare_zenodo_samples.sh` puts the paper's samples (4.2 GB, see Data) |

Press Enter to accept a default. Later `source setup.sh` calls reuse `.config` and ask only
for keys that are missing; edit a value, or delete its line to be asked again. The Slurm scripts under
`src/parnassus/torch_delphes/` source `setup.sh` themselves and so use the same `.config`.

## Data

The samples of the paper are on Zenodo: [record 22071385](https://zenodo.org/records/22071385)
(DOI 10.5281/zenodo.22071385, CC-BY-4.0; 14 files, 4.2 GB). Download them all with

```bash
bash prepare_zenodo_samples.sh
```

The files go to `SAMPLE_DIR` from `.config` (asked for by `source setup.sh`). Files that are
already complete are skipped, partial downloads are resumed, and each download is md5-checked
against Zenodo, so the script can be re-run at any time.

| files | content |
|---|---|
| `pseudo_data_200k_param_config_{muons_muongun,electrons_electrongun,chads_ksgun,dijets_dijet}.root` | per-block regression samples, 200k events each |
| `pseudo_data_200k_param_config_all_{muongun,ksgun,dijet,electrongun}.root` | sequential all-parameter fit samples, 200k events each |
| `pseudo_data_100k_param_config_all_HZZ4l.root` | independent closure sample, 100k events (VBF H → ZZ → 4l) |
| `qcd_dijet.cmnd`, `muon_gun.cmnd`, `electron_gun.cmnd`, `kshort_gun.cmnd`, `HZZ4l.cmnd` | the Pythia8 process cards the samples were generated from (the copies in `src/parnassus/torch_delphes/processes/` are identical apart from explanatory comments) |

Each ROOT file holds one tree, `event_tree`, with jagged branches `truth_{pt,eta,phi,class,pdgid}`
(stable generator particles) and `pflow_{pt,eta,phi,class}` (reconstructed particle-flow objects);
`class` is 0 charged hadron, 1 electron, 2 muon, 3 neutral hadron, 4 photon. Every sample was
produced by running Pythia8 with its card, passing the truth particles through the frozen torch
detector card initialised from a "truth" parameter config, and storing both; the fits start the
same card at the CMS defaults and should recover the truth values.
