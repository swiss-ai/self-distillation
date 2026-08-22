#!/bin/bash

# One-time builder for the shared venv used by run_sdpo.sh.
#
# Why a prebuilt venv:
#   The launch scripts otherwise run `pip install -e .` on every node of every
#   job. With many nodes hitting the shared NFS pip cache concurrently this
#   intermittently fails ("Stale file handle" / ESTALE) and leaves nodes on
#   mismatched dependency versions (e.g. numpy 1.26 vs 2.1), which later crashes
#   the cross-node reward/benchmark deserialization. Building the environment
#   ONCE into a shared venv and just activating it gives every node an identical,
#   pre-resolved environment.
#
# Why it must run inside the container:
#   The venv pins itself to the interpreter that created it and, with
#   --system-site-packages, inherits whatever site-packages are visible at build
#   time (torch/vllm/etc live INSIDE the container image). Building on the host
#   would point at the wrong interpreter and miss the GPU stack. This script
#   re-execs itself through `srun --environment=$ENVIRONMENT` so you can just run
#   it from a login node.
#
# Usage:
#   ./experiments/constitutional_ai/build_env.sh            # build (idempotent)
#   REBUILD=1 ./experiments/constitutional_ai/build_env.sh  # wipe and rebuild
#   VENV_DIR=/some/other/path ./.../build_env.sh            # custom location

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

VENV_DIR="${VENV_DIR:-/users/$USER/venvs/sd-cai}"
ENVIRONMENT="${ENVIRONMENT:-sd}"
ACCOUNT="${ACCOUNT:-infra01}"
PARTITION="${PARTITION:-normal}"
TIME="${TIME:-00:30:00}"
REBUILD="${REBUILD:-0}"

build() {
    cd "$REPO_DIR"

    if [[ "$REBUILD" == "1" && -d "$VENV_DIR" ]]; then
        echo "REBUILD=1: removing existing venv at $VENV_DIR"
        rm -rf "$VENV_DIR"
    fi

    mkdir -p "$(dirname "$VENV_DIR")"

    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
        echo "Creating venv (with system site-packages) at $VENV_DIR"
        python -m venv --system-site-packages "$VENV_DIR"
    else
        echo "Reusing existing venv at $VENV_DIR (pass REBUILD=1 to recreate)"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    # --no-cache-dir keeps this off the shared NFS pip cache. This runs once and
    # single-threaded, so it is far more reliable than the per-job concurrent path.
    pip install --no-cache-dir -e .
    pip install --no-cache-dir --upgrade wandb

    echo "----------------------------------------------------------------"
    echo "venv ready at $VENV_DIR"
    python -c "import numpy, sys; print('python', sys.version.split()[0]); print('numpy', numpy.__version__)"
    echo "run_sdpo.sh will now activate it automatically."
}

if [[ "${IN_CONTAINER:-0}" == "1" ]]; then
    # Already inside the container (re-exec'd by the branch below).
    build
else
    echo "Building venv inside the '$ENVIRONMENT' container via srun..."
    export VENV_DIR ENVIRONMENT REBUILD IN_CONTAINER=1
    srun --environment="$ENVIRONMENT" --nodes=1 --ntasks=1 \
        --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME" \
        bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
fi
