#!/bin/bash
# Download the paper's samples from Zenodo (record 22071385: 9 pseudo-data ROOT files +
# 5 Pythia8 cards, 4.2 GB) into SAMPLE_DIR. Files that are already complete are skipped,
# partial downloads are resumed, and every downloaded file is md5-checked.
# USAGE (from anywhere; needs curl, python3, md5sum):
#   bash <repo>/prepare_zenodo_samples.sh
# SAMPLE_DIR is read from <repo>/.config, which `source setup.sh` writes (it asks for the
# path once). Edit that line to move the samples elsewhere.
set -euo pipefail

RECORD="${ZENODO_RECORD:-22071385}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$REPO_DIR/.config"

# shellcheck disable=SC1090
if [ -f "$CONFIG" ]; then source "$CONFIG"; fi
if [ -z "${SAMPLE_DIR:-}" ]; then
    echo "[zenodo] SAMPLE_DIR is not set in $CONFIG — run  source $REPO_DIR/setup.sh  once (it asks for it), then re-run." >&2
    exit 1
fi
mkdir -p "$SAMPLE_DIR"
echo "[zenodo] record $RECORD -> $SAMPLE_DIR"

# The Zenodo REST API lists the files; one line per file: name md5 size url.
curl -sSfL "https://zenodo.org/api/records/$RECORD" | python3 -c '
import json, sys
for f in sorted(json.load(sys.stdin)["files"], key=lambda f: f["key"]):
    print(f["key"], f["checksum"].split(":")[1], f["size"], f["links"]["self"])
' | while read -r name md5 size url; do
    out="$SAMPLE_DIR/$name"
    if [ -f "$out" ] && [ "$(wc -c < "$out")" -eq "$size" ]; then
        echo "[zenodo] have  $name"
        continue
    fi
    echo "[zenodo] fetch $name ($((size / 1000000)) MB)"
    curl -fL -C - --retry 5 --progress-bar -o "$out" "$url" </dev/null   # -C -: resume a partial file
    echo "$md5  $out" | md5sum -c --quiet
done
echo "[zenodo] done: $SAMPLE_DIR"
