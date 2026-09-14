# Parnassus-P

## Installation

Requirements: bash, git and curl. `uv` and a matching Python (3.12) are fetched
automatically if missing.

```bash
git clone <this repo> && cd parnassus
source setup.sh
```

`source setup.sh` (not `bash setup.sh`) installs the pinned environment from `uv.lock`
and activates it. On first use it asks for three directories and saves them in the
git-ignored `.config`:

| key | meaning |
|---|---|
| `ENV_PREFIX` | this clone's virtualenv (unique per clone; several GB — on NERSC use `$SCRATCH` or CFS, not `$HOME`) |
| `UV_CACHE` | uv download cache (can be shared by all your clones) |
| `UV_PYTHON_DIR` | where uv puts the Python it downloads if none matches on `PATH` |

Press Enter to accept a default. Later `source setup.sh` calls reuse `.config` without
asking; edit it, or delete it to be asked again. The Slurm scripts under
`src/parnassus/torch_delphes/` source `setup.sh` themselves and so use the same `.config`.
