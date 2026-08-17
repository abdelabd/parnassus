#!/bin/bash
#SBATCH -A m3246
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -J mergepseudo
#
# Merge the per-seed ${PART_PREFIX}_seed*.root files produced by
# submit_pseudodata.sh into $OUTBASE/$MERGED_NAME, validate the total event
# count, then delete the parts. Normally submitted by submit_pseudodata.sh with
# --dependency=afterok, but can also be run standalone once the parts exist:
#   OUTBASE=... MERGED_NAME=pseudo_data_100k_<cfg>_<process>.root \
#       sbatch src/parnassus/torch_delphes/slurm_scripts/merge_pseudodata.sbatch
# Config comes from the environment (same vars as submit_pseudodata.sh);
# PART_PREFIX defaults to the MERGED_NAME stem.
#
# Concatenation is safe: the ROOT files carry no persisted event-number branch
# (event identity is the TTree entry index), so appending entries yields globally
# unique indices with no duplication.

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/global/cfs/cdirs/m3246/Runze/MCGen/envs/parnassus}"
# shellcheck disable=SC1091
source "$ENV_PREFIX/bin/activate"

export OUTBASE="${OUTBASE:-/global/cfs/cdirs/m3246/diff_delphes}"
export PARTS_DIR="${PARTS_DIR:-$OUTBASE/parts}"
export MERGED_NAME="${MERGED_NAME:?MERGED_NAME must be set (e.g. pseudo_data_100k_<cfg>_<process>.root)}"
export PART_PREFIX="${PART_PREFIX:-${MERGED_NAME%.root}}"
export NJOBS="${NJOBS:-10}"

python - <<'PY'
import glob
import os
import sys

import uproot

OUTBASE = os.environ["OUTBASE"]
PARTS_DIR = os.environ["PARTS_DIR"]
MERGED = os.path.join(OUTBASE, os.environ["MERGED_NAME"])
PART_PREFIX = os.environ["PART_PREFIX"]
NJOBS = int(os.environ["NJOBS"])
TREE = "event_tree"

parts = sorted(glob.glob(os.path.join(PARTS_DIR, f"{PART_PREFIX}_seed*.root")))
if not parts:
    sys.exit(f"[merge] ERROR: no part files found in {PARTS_DIR}")
if len(parts) != NJOBS:
    print(f"[merge] WARNING: found {len(parts)} parts, expected NJOBS={NJOBS}")
print(f"[merge] merging {len(parts)} files -> {MERGED}")

# Count source entries up front for a post-merge sanity check.
src_total = 0
for p in parts:
    with uproot.open(p) as f:
        src_total += f[TREE].num_entries
print(f"[merge] source events total: {src_total}")

# Streaming concat: read one file (~95 MB) at a time and append, so peak memory
# stays at a single part rather than the whole sample. Reading every field
# (rec.fields) keeps it correct for --debug files (~150 extra branches) too.
os.makedirs(OUTBASE, exist_ok=True)
written = 0
with uproot.recreate(MERGED) as fout:
    for i, p in enumerate(parts):
        with uproot.open(p) as f:
            rec = f[TREE].arrays(library="ak")
        payload = {field: rec[field] for field in rec.fields}
        if i == 0:
            fout[TREE] = payload
        else:
            fout[TREE].extend(payload)
        written += len(rec)
        print(f"[merge]   {os.path.basename(p)}: +{len(rec)} (running total {written})")

with uproot.open(MERGED) as f:
    merged_n = f[TREE].num_entries
if merged_n != src_total:
    sys.exit(
        f"[merge] ERROR: merged {merged_n} != source total {src_total}; "
        "keeping parts for inspection"
    )
print(f"[merge] OK: merged {merged_n} events == sum of parts")

for p in parts:
    os.remove(p)
try:
    os.rmdir(PARTS_DIR)
except OSError:
    pass
print(f"[merge] deleted {len(parts)} part files")
print(f"[merge] DONE: {MERGED} ({merged_n} events)")
PY
