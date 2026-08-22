#!/bin/bash

# Host the constitutional-AI judge as an OpenAI-compatible vLLM endpoint.
#
# The judge critiques each student response against the constitution in
# verl/trainer/config/sdpo_cai.yaml; run_sdpo.sh injects that feedback into the
# teacher reprompt. The judge runs as its OWN Slurm job so that every node of the
# training job is used for training.
#
# This script is a convenience wrapper around `sml` (swiss-ai/model-launch), which
# is CSCS-specific. NOTHING in the training path depends on it: run_sdpo.sh only
# needs an OpenAI-compatible /v1/chat/completions endpoint, so any vLLM / SGLang /
# hosted deployment works. If you serve the judge yourself, just point
# LLM_JUDGE_BASE_URL / LLM_JUDGE_MODEL / LLM_JUDGE_API_KEY at it and skip this script.
#
# The released Apertus-1.5-8B constitutional-AI run used the internal
# Apertus-1.5-70B-SFT-RL-DPO-FINAL checkpoint as the judge (16 replicas, TP=4).
#
# Usage: ./launch_judge.sh [--dry-run] [extra sml args...]

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi

ML_DIR="${ML_DIR:-/users/$USER/projects/model-launch}"
SML="${SML:-${ML_DIR}/.venv/bin/sml}"

FIRECREST_SYSTEM="${FIRECREST_SYSTEM:-clariden}"
ACCOUNT="${ACCOUNT:-infra01}"
PARTITION="${PARTITION:-normal}"
# Optional Slurm reservation; the released runs used the internal CSCS
# reservation SD-69241-apertus-1-5-0. Leave empty elsewhere.
RESERVATION="${RESERVATION:-}"
JUDGE_TIME="${JUDGE_TIME:-12:00:00}"
JUDGE_REPLICAS="${JUDGE_REPLICAS:-16}"

# Judge checkpoint and the name it is served under. SERVED_MODEL_NAME must match
# JUDGE_MODEL in run_sdpo.sh (both default to apertus-cai-judge-$USER).
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/iopsstor/scratch/cscs/smarian/models/Apertus-1.5-70B-SFT-RL-DPO-FINAL-fixed}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-apertus-cai-judge-$USER}"

if [[ ! -x "$SML" ]]; then
    echo "ERROR: sml not found at $SML. Set SML or ML_DIR, or build the venv:" >&2
    echo "  cd $ML_DIR && uv venv --python 3.12 && source .venv/bin/activate && uv pip install ." >&2
    exit 1
fi

# sml advanced reads the .toml env by a path relative to the model-launch repo root.
cd "$ML_DIR"

sml_cmd=(
    "$SML" advanced
    --system "$FIRECREST_SYSTEM"
    --partition "$PARTITION"
    --account "$ACCOUNT"
    --time "$JUDGE_TIME"
    --framework vllm
    --environment src/swiss_ai_model_launch/assets/envs/vllm.toml
    --replicas "$JUDGE_REPLICAS"
    --nodes-per-replica 1
    --no-tui
    --framework-args "--model ${JUDGE_MODEL_PATH} --host 0.0.0.0 --served-model-name ${SERVED_MODEL_NAME} --tensor-parallel-size 4 --max-model-len 16384"
)

if [[ -n "${RESERVATION:-}" ]]; then
    sml_cmd+=(--reservation "$RESERVATION")
fi

sml_cmd+=("$@")

echo "Hosting judge '${SERVED_MODEL_NAME}' via sml (${JUDGE_REPLICAS} replica(s), TP=4, reservation ${RESERVATION:-none})"
if [ "$DRY_RUN" = true ]; then
    echo "----------------------------------------------------------------"
    printf '%q ' "${sml_cmd[@]}"; echo
    exit 0
fi

"${sml_cmd[@]}"

echo
echo "Judge submitted. Once healthy it is reachable at:"
echo "  LLM_JUDGE_BASE_URL=https://api.swissai.svc.cscs.ch/v1"
echo "  LLM_JUDGE_MODEL=${SERVED_MODEL_NAME}"
echo "Then submit training: ./experiments/constitutional_ai/run_sdpo.sh"
