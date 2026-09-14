#!/bin/bash
# Idempotent install + activation of a per-repo uv env.
# USAGE: source this script, do not execute it (from any cwd):
#   source <repo>/setup.sh
# The env name is derived from the repo folder, so this script can be copy-pasted
# into any repo and each repo gets its own isolated env.
#
# Where things go is read from <repo>/.config (git-ignored, one per clone):
#   ENV_PREFIX="..."     this clone's virtualenv (unique per clone; several GB —
#                        on NERSC use $SCRATCH or CFS, not $HOME)
#   UV_CACHE="..."       uv download cache (can be shared by all your clones)
#   UV_PYTHON_DIR="..."  where uv puts the Python it downloads if none matches on PATH
# On first use you are prompted for the three paths and the answers are saved.
# To change them, edit .config, or delete it to be prompted again.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="$(basename "$REPO_DIR")"
CONFIG="$REPO_DIR/.config"

if [ ! -f "$CONFIG" ]; then
    BASE="${SCRATCH:-$HOME/.local/share}"
    echo "[setup] No $CONFIG yet — where should uv put things? (Enter = default)"
    read -r -p "  ENV_PREFIX    (this clone's env)      [$BASE/envs/$ENV_NAME]: " ENV_PREFIX
    read -r -p "  UV_CACHE      (uv cache, shareable)   [$BASE/uv-cache]: " UV_CACHE
    read -r -p "  UV_PYTHON_DIR (uv Pythons, shareable) [$BASE/uv-python]: " UV_PYTHON_DIR
    ENV_PREFIX="${ENV_PREFIX:-$BASE/envs/$ENV_NAME}"
    UV_CACHE="${UV_CACHE:-$BASE/uv-cache}"
    UV_PYTHON_DIR="${UV_PYTHON_DIR:-$BASE/uv-python}"
    # A leading ~ would not expand inside the quotes written below, so spell it out.
    {
        echo "# Written by setup.sh. Edit, or delete to be prompted again."
        printf '%s="%s"\n' ENV_PREFIX "${ENV_PREFIX/#\~/$HOME}" \
                           UV_CACHE "${UV_CACHE/#\~/$HOME}" \
                           UV_PYTHON_DIR "${UV_PYTHON_DIR/#\~/$HOME}"
    } > "$CONFIG"
    echo "[setup] wrote $CONFIG"
fi
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
