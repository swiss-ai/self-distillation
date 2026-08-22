#!/bin/bash
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
# export RAY_DEBUG=1
ulimit -c 0

# Some cluster/container environments export ROCm device visibility vars even on
# NVIDIA jobs. Ray/verl treats ROCR/HIP and CUDA visibility settings as mutually
# exclusive, so drop the ROCm variants when CUDA is in use.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  unset ROCR_VISIBLE_DEVICES
  unset HIP_VISIBLE_DEVICES
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export EXPERIMENT="${1:-experiment}"
CONFIG_NAME="${2:-ppo_trainer}"
export TASK="${3:-datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000}"

if [[ "$#" -ge 3 ]]; then
  shift 3
else
  echo "Usage: $0 <experiment_name> <config_name> <data_path>"
  echo "Example: $0 test ppo_trainer datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"
  exit 1
fi

echo "Experiment: ${EXPERIMENT}"
echo "Config: ${CONFIG_NAME}"
echo "Task: ${TASK}"
echo "Arguments: $*"

python -m verl.trainer.main_ppo --config-name "${CONFIG_NAME}" "$@"
