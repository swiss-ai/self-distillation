#!/bin/bash

# Constitutional-AI SDPO for Apertus-1.5-8B.
#
# This is the exact training that produces the constitutional-AI stage of the
# released Apertus-1.5-8B model. Instead of a scalar reward, a hosted 70B judge
# critiques each student response against the constitution, and that textual
# feedback is injected into the teacher reprompt; the student is then distilled
# towards the teacher's response distribution.
# See verl/trainer/config/sdpo_cai.yaml for the constitution and the prompts.
#
# The judge is hosted OUT-OF-BAND (see launch_judge.sh) and reached over an
# OpenAI-compatible endpoint. This script only points the training job at it; it
# does not allocate a judge node, so all reserved nodes are used for training.
#
# Workflow:
#   1. ./experiments/constitutional_ai/build_env.sh     # one-time: build the shared venv
#   2. ./experiments/constitutional_ai/launch_judge.sh  # host the judge (separate Slurm job)
#   3. ./experiments/constitutional_ai/run_sdpo.sh      # submit training pointing at it
#
# Judge connection (consumed by verl/utils/reward_score/feedback/llm.py):
#   LLM_JUDGE_BASE_URL  - OpenAI-compatible base URL of the judge endpoint
#   LLM_JUDGE_MODEL     - served model name (must match launch_judge.sh)
#   LLM_JUDGE_API_KEY   - API key for that endpoint; put it in .env (see .env.example)
#
# Usage: ./run_sdpo.sh [--dry-run] [hydra overrides...]

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
    shift
fi

EXTRA_OVERRIDES=("$@")

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_DIR="$(pwd)"
VENV_DIR="${VENV_DIR:-/users/$USER/venvs/sd-cai}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    ENV_SETUP="echo Using prebuilt venv ${VENV_DIR}; source ${VENV_DIR}/bin/activate"
else
    ENV_SETUP="echo venv ${VENV_DIR} not found, installing per-node; pip install --no-cache-dir -e .; pip install --no-cache-dir --upgrade wandb"
fi

CONFIG_NAME="${CONFIG_NAME:-sdpo_cai}"
BASE_JOB_NAME="${BASE_JOB_NAME:-sdpo-cai-8b}"
PROJECT_NAME="${PROJECT_NAME:-apertus-1.5-sdpo}"
WANDB_ENTITY="${WANDB_ENTITY:-apertus}"

# ---- Hosted judge connection ------------------------------------------------
# Any OpenAI-compatible endpoint works. The released model used the internal
# Apertus-1.5-70B-SFT-RL-DPO-FINAL checkpoint as the judge, served with vLLM
# (see launch_judge.sh); SERVED_MODEL_NAME there must match LLM_JUDGE_MODEL here.
# The training runs used the CSCS inference gateway. NOTE: the inference API host
# is api.swissai.svc.cscs.ch (serving.swissai.svc.cscs.ch is only the web portal
# for keys). The gateway is probed via POST /v1/chat/completions; it does not
# expose GET /v1/models. LLM_JUDGE_API_KEY must be set, with a value, in .env.
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.swissai.svc.cscs.ch/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-apertus-cai-judge-$USER}"
# Short judge tag for the experiment name so runs with different judges don't share a
# wandb run / checkpoint / rollout dir. Drops the vendor prefix and the -$USER suffix.
JUDGE_TAG="${JUDGE_MODEL##*/}"
JUDGE_TAG="${JUDGE_TAG%-$USER}"
# Max seconds to wait for the judge before aborting (instead of hanging the whole job).
JUDGE_WAIT_SECONDS="${JUDGE_WAIT_SECONDS:-2400}"

# ---- Data -------------------------------------------------------------------
# Regenerate with:
#   python data/preprocess/gretel_safety_dataset.py --output-dir datasets/gretel-safety-alignment
DATA_PATH="${DATA_PATH:-datasets/gretel-safety-alignment}"

# ---- Cluster ----------------------------------------------------------------
ACCOUNT="${ACCOUNT:-infra01}"
TRAIN_NODES="${TRAIN_NODES:-32}"
NODES="${NODES:-$TRAIN_NODES}"
PARTITION="${PARTITION:-normal}"
TIME="${TIME:-12:00:00}"
ENVIRONMENT="${ENVIRONMENT:-sd}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TRAIN_GPUS_TOTAL=$((TRAIN_NODES * GPUS_PER_NODE))

ROLLOUT_TP="${ROLLOUT_TP:-4}"
FSDP_OFFLOAD="${FSDP_OFFLOAD:-true}"

MEM="${MEM:-460000}"
CPUS_PER_TASK="${CPUS_PER_TASK:-288}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/logs/constitutional-ai/sdpo}"
# Optional Slurm reservation. The released runs used the internal CSCS
# reservation SD-69241-apertus-1-5-0; leave empty elsewhere.
RESERVATION="${RESERVATION:-}"

# ---- Hyperparameters (Apertus-1.5-8B, as released) --------------------------
# One epoch of the ~7.4k-prompt gretel-safety-alignment set is the real limiter:
# at train_batch_size=64 that is ~116 steps, so TOTAL_STEPS=150 is never reached.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
LR="${LR:-1e-7}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.0}"
ALPHA="${ALPHA:-1.0}"
TOTAL_STEPS="${TOTAL_STEPS:-150}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
FULL_LOGIT_DISTILLATION="${FULL_LOGIT_DISTILLATION:-True}"
DISTILLATION_TOPK="${DISTILLATION_TOPK:-64}"

# Student checkpoint. The released run started from the internal Apertus-1.5-8B
# SFT+RL+DPO checkpoint (with the tf4-legacy RoPE config fix, see README);
# swiss-ai/Apertus-v1.5-8B is the public post-CAI model.
MODEL_PATH="${MODEL_PATH:-/iopsstor/scratch/cscs/smarian/models/Apertus-1.5-8B-SFT-RL-DPO-fixed}"
MODEL_NAME="${MODEL_NAME:-Apertus-1.5-8B-SFT-RL-DPO-ropefix}"

# =============================================================================
# JOB SUBMISSION
# =============================================================================

submit_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"

    # Source .env so LLM_JUDGE_API_KEY is available at runtime. ENV_SETUP (computed
    # once above) either activates the prebuilt venv or does the per-node fallback install.
    local setup_cmds="${SETUP_CMDS:-cd ${REPO_DIR}; set -a; source .env; set +a; ${ENV_SETUP}}"
    local train_cmd="cd ${REPO_DIR}; bash training/verl_training.sh ${exp_name} ${CONFIG_NAME} ${data_path} ${script_args}"

    # Node 0 is Ray head + trainer; the rest are Ray workers. The judge is hosted
    # out-of-band (launch_judge.sh) and reached over the network via LLM_JUDGE_BASE_URL,
    # so no judge node is allocated here. LLM_JUDGE_* are exported on every node because
    # the reward (judge) call can run from any Ray worker.
    local node_script="\
NODES_ARR=(\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")); \
HEAD_NODE=\${NODES_ARR[0]}; \
TRAIN_NNODES=\$SLURM_NNODES; \
if [[ \"\$SLURM_NODEID\" == \"0\" ]]; then \
  ${setup_cmds}; \
  export LLM_JUDGE_BASE_URL=\"${JUDGE_BASE_URL}\"; \
  export LLM_JUDGE_MODEL=\"${JUDGE_MODEL}\"; \
  export EXPERIMENT=\"${exp_name}\"; \
  export TASK=\"${data_path}\"; \
  ray start --head --node-ip-address=\"\$HEAD_NODE\" --port=6379 --num-gpus=${GPUS_PER_NODE}; \
  echo \"Waiting (max ${JUDGE_WAIT_SECONDS}s) for judge \$LLM_JUDGE_MODEL at \$LLM_JUDGE_BASE_URL ...\"; \
  judge_wait=0; \
  until curl -sf -o /dev/null -X POST \"\$LLM_JUDGE_BASE_URL/chat/completions\" -H \"Authorization: Bearer \${LLM_JUDGE_API_KEY:-EMPTY}\" -H \"Content-Type: application/json\" -d \"{\\\"model\\\":\\\"\$LLM_JUDGE_MODEL\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":\\\"ping\\\"}],\\\"max_tokens\\\":1}\"; do \
    judge_wait=\$((judge_wait + 15)); \
    if (( judge_wait > ${JUDGE_WAIT_SECONDS} )); then echo \"ERROR: judge \$LLM_JUDGE_MODEL not reachable at \$LLM_JUDGE_BASE_URL after ${JUDGE_WAIT_SECONDS}s (check LLM_JUDGE_API_KEY in .env and that launch_judge.sh is healthy); aborting.\"; scancel \"\$SLURM_JOB_ID\"; exit 1; fi; \
    sleep 15; \
  done; \
  echo \"Judge ready.\"; \
  echo \"Waiting for \$TRAIN_NNODES Ray nodes to join...\"; \
  until [[ \$(ray status 2>/dev/null | grep -c node_) -ge \$TRAIN_NNODES ]]; do sleep 5; done; \
  echo \"Ray cluster ready.\"; \
  ${train_cmd}; \
  CKPT_DIR=\"\${SCRATCH}/checkpoints/${PROJECT_NAME}/${exp_name}\"; \
  if [[ -d \"\$CKPT_DIR\" ]]; then \
    echo \"Merging FSDP checkpoints to HF format under \$CKPT_DIR/hf ...\"; \
    cd ${REPO_DIR}; \
    bash ./scripts/merge_fsdp_checkpoint.sh \"\$CKPT_DIR\" \"\$CKPT_DIR/hf\" || echo \"FSDP→HF merge failed (continuing)\"; \
  else \
    echo \"Skipping HF merge: \$CKPT_DIR does not exist.\"; \
  fi; \
  scancel \"\$SLURM_JOB_ID\"; \
else \
  ${setup_cmds}; \
  export LLM_JUDGE_BASE_URL=\"${JUDGE_BASE_URL}\"; \
  export LLM_JUDGE_MODEL=\"${JUDGE_MODEL}\"; \
  until ray health-check --address=\"\${HEAD_NODE}:6379\" >/dev/null 2>&1; do sleep 5; done; \
  ray start --address=\"\${HEAD_NODE}:6379\" --num-gpus=${GPUS_PER_NODE} --block; \
fi"

    local wrapped_cmd="srun --environment=$ENVIRONMENT bash -c '$node_script'"

    local sbatch_cmd=(
        sbatch
        --job-name="$BASE_JOB_NAME"
        --account="$ACCOUNT"
        --nodes="$NODES"
        --partition="$PARTITION"
        --time="$TIME"
        --ntasks-per-node="$NTASKS_PER_NODE"
        --gpus-per-node="$GPUS_PER_NODE"
        --mem="$MEM"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="${OUTPUT_DIR}/%j.log"
        --error="${OUTPUT_DIR}/%j.err"
    )

    if [[ -n "${RESERVATION:-}" ]]; then
        sbatch_cmd+=(--reservation="$RESERVATION")
    fi

    # Exclude known-bad nodes (e.g. one where pyxis container mount OOMs → "DUE TO TASK
    # FAILURE"). SBATCH_EXCLUDE env var is ignored on this system, so pass --exclude explicitly.
    if [[ -n "${EXCLUDE_NODES:-}" ]]; then
        sbatch_cmd+=(--exclude="$EXCLUDE_NODES")
    fi

    sbatch_cmd+=(--wrap="$wrapped_cmd")

    if [ "$DRY_RUN" = true ]; then
        echo "----------------------------------------------------------------"
        echo "Would submit job for: $exp_name"
        echo "${sbatch_cmd[@]}"
    else
        echo "Submitting job for: $exp_name"
        mkdir -p "$OUTPUT_DIR"
        "${sbatch_cmd[@]}"
    fi
}

# =============================================================================
# LAUNCH
# =============================================================================

# Batch-size divisibility guard: verl aborts on these, so fail early and loudly.
if (( (TRAIN_BATCH_SIZE * ROLLOUT_BATCH_SIZE) % TRAIN_GPUS_TOTAL != 0 )); then
    echo "ERROR: train_batch $TRAIN_BATCH_SIZE * n $ROLLOUT_BATCH_SIZE must be divisible by $TRAIN_GPUS_TOTAL GPUs" >&2; exit 1
fi
if (( (PPO_MINI_BATCH_SIZE * ROLLOUT_BATCH_SIZE) % TRAIN_GPUS_TOTAL != 0 || \
      (PPO_MINI_BATCH_SIZE * ROLLOUT_BATCH_SIZE) < TRAIN_GPUS_TOTAL )); then
    echo "ERROR: ppo_mini $PPO_MINI_BATCH_SIZE * n $ROLLOUT_BATCH_SIZE must be a multiple of, and >= $TRAIN_GPUS_TOTAL GPUs" >&2; exit 1
fi
if (( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
    echo "ERROR: train_batch $TRAIN_BATCH_SIZE not divisible by ppo_mini $PPO_MINI_BATCH_SIZE" >&2; exit 1
fi

DISTILL_TAG="fld${FULL_LOGIT_DISTILLATION}"
if [[ "$FULL_LOGIT_DISTILLATION" == "True" ]]; then
    DISTILL_TAG="${DISTILL_TAG}-topk${DISTILLATION_TOPK}"
fi
DATA_TAG="$(basename "$DATA_PATH")"
EXP_NAME="${EXP_NAME:-${MODEL_NAME}-SDPO-judge-${JUDGE_TAG}-${DATA_TAG}-bs${TRAIN_BATCH_SIZE}-mb${PPO_MINI_BATCH_SIZE}-n${ROLLOUT_BATCH_SIZE}-lr${LR}-alpha${ALPHA}-tur${TEACHER_UPDATE_RATE}-${DISTILL_TAG}-steps${TOTAL_STEPS}-epochs${TOTAL_EPOCHS}-ropefix-${TRAIN_NODES}nodes}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
trainer.nnodes=$TRAIN_NODES \
trainer.n_gpus_per_node=$GPUS_PER_NODE \
trainer.project_name=${PROJECT_NAME} \
trainer.wandb_entity=${WANDB_ENTITY} \
trainer.total_training_steps=$TOTAL_STEPS \
trainer.total_epochs=$TOTAL_EPOCHS \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.full_logit_distillation=$FULL_LOGIT_DISTILLATION \
actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=4 \
actor_rollout_ref.actor.fsdp_config.fsdp_size=$TRAIN_GPUS_TOTAL \
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
actor_rollout_ref.actor.fsdp_config.param_offload=$FSDP_OFFLOAD \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=$FSDP_OFFLOAD \
actor_rollout_ref.ref.fsdp_config.fsdp_size=$TRAIN_GPUS_TOTAL \
actor_rollout_ref.ref.fsdp_config.reshard_after_forward=true \
actor_rollout_ref.ref.fsdp_config.param_offload=$FSDP_OFFLOAD \
actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT:-7200} \
actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.5} \
trainer.save_freq=${SAVE_FREQ:-25} \
trainer.rollout_data_dir=${OUTPUT_DIR}/generations/${EXP_NAME}/rollouts \
trainer.validation_data_dir=${OUTPUT_DIR}/generations/${EXP_NAME}/val \
trainer.log_val_generations=10 \
trainer.skip_dataset_validation=True \
++trainer.benchmarks.enabled=[gsm8k,mmlu] \
++trainer.benchmarks.math.num_generations=2 \
++trainer.benchmarks.math.gsm8k.max_samples=128 \
++trainer.benchmarks.math.gsm8k.prompt_length=4096 \
++trainer.benchmarks.mcq.num_generations=2 \
++trainer.benchmarks.mcq.mmlu.max_samples=128 \
++trainer.benchmarks.mcq.mmlu.prompt_length=4096"

if ((${#EXTRA_OVERRIDES[@]} > 0)); then
    printf -v EXTRA_ARGS_STR ' %q' "${EXTRA_OVERRIDES[@]}"
    ARGS+="${EXTRA_ARGS_STR}"
fi

submit_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
