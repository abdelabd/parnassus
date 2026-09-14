#!/bin/bash
# Idempotent install + activation of a per-repo uv env.
# USAGE: source this script, do not execute it (from any cwd):
#   source <repo>/setup.sh
# The env name is derived from the repo folder, so this script can be copy-pasted
# into any repo and each repo gets its own isolated env.
#
# All user-specific paths live in <repo>/.config (git-ignored, one per clone):
#   ENV_PREFIX="..."     this clone's virtualenv (unique per clone; several GB —
#                        on NERSC use $SCRATCH or CFS, not $HOME)
#   UV_CACHE="..."       uv download cache (can be shared by all your clones)
#   UV_PYTHON_DIR="..."  where uv puts the Python it downloads if none matches on PATH
#   SAMPLE_DIR="..."     where prepare_zenodo_samples.sh puts the paper's samples (4.2 GB)
# Any key that .config does not define yet is asked for once (Enter = default) and
# appended. To change a value, edit .config, or delete its line to be asked again.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="$(basename "$REPO_DIR")"
CONFIG="$REPO_DIR/.config"

# ask KEY DEFAULT DESCRIPTION — prompt for KEY unless .config already defines it.
ask() {
    local reply
    [ -n "${!1:-}" ] && return 0
    read -r -p "  $1 ($3) [$2]: " reply || true
    reply="${reply:-$2}"
    [ -f "$CONFIG" ] || echo "# Paths for this clone (git-ignored), written by setup.sh. Edit, or delete a line to be asked again." > "$CONFIG"
    printf '%s="%s"\n' "$1" "${reply/#\~/$HOME}" >> "$CONFIG"   # spell out ~: it does not expand inside the quotes
}
# shellcheck disable=SC1090
if [ -f "$CONFIG" ]; then source "$CONFIG"; fi
BASE="${SCRATCH:-$HOME/.local/share}"
ask ENV_PREFIX    "$BASE/envs/$ENV_NAME"               "this clone's uv env, several GB"
ask UV_CACHE      "$BASE/uv-cache"                     "uv cache, shareable"
ask UV_PYTHON_DIR "$BASE/uv-python"                    "uv Pythons, shareable"
ask SAMPLE_DIR    "${SCRATCH:-$HOME}/parnassus_samples" "Zenodo samples, 4.2 GB"
unset -f ask
# shellcheck disable=SC1090
source "$CONFIG"

export UV_PROJECT_ENVIRONMENT="$ENV_PREFIX"
export UV_CACHE_DIR="$UV_CACHE"
export UV_PYTHON_INSTALL_DIR="$UV_PYTHON_DIR"
mkdir -p "$UV_CACHE" "$UV_PYTHON_DIR"

# Ensure uv is on PATH (install to ~/.local/bin if missing).
# The installer is ~30MB, no sudo needed, no system pkgs touched.
if ! command -v uv >/dev/null 2>&1; then
    if [ ! -x "$HOME/.local/bin/uv" ]; then
        echo "[setup] Installing uv to ~/.local/bin ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync from the committed lock; fail loudly if pyproject.toml and the lock disagree.
# Idempotent: no-op if the env already matches the lock.
# The cache is on CFS/GPFS, whose flock() support is intermittent and can return
# errno 524 (ENOTSUPP). Retry so a transient lock failure self-heals; if it never
# succeeds, warn loudly so a genuinely-needed sync is never silently skipped.
sync_ok=0
for attempt in 1 2 3; do
    if ( cd "$REPO_DIR" && uv sync --locked --all-extras ); then
        sync_ok=1
        break
    fi
    echo "[setup] uv sync failed (attempt $attempt/3) — retrying in 2s ..." >&2
    sleep 2
done
if [ "$sync_ok" -ne 1 ]; then
    echo "[setup] WARNING: 'uv sync' did not complete after 3 attempts — see uv's output above." >&2
    echo "[setup]          If it is a CFS/GPFS lock error (os error 524), just re-run: source \"$REPO_DIR/setup.sh\"" >&2
    echo "[setup]          Otherwise uv.lock and pyproject.toml may disagree; the active env may be STALE." >&2
fi

# Activate
source "$ENV_PREFIX/bin/activate"
echo "[setup] $ENV_NAME env active: $ENV_PREFIX"
echo "[setup] python: $(which python)   ($(python --version))"
