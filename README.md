# Self-Distillation for Apertus

Post-training with **self-distillation**: instead of optimizing a scalar reward, the model is
distilled towards a *better-informed version of itself*. A rollout is critiqued in natural language,
the critique is fed back to the same model as extra context, and the student is trained towards the
distribution that this re-prompted **teacher** places over the student's own tokens. The training
signal is text end to end — no reward model, no preference pairs.

The recipe shipped here is the **constitutional-AI** stage of
[Apertus-1.5](https://huggingface.co/collections/swiss-ai/apertus): a judge model critiques each
response against a written constitution, and that critique is what the teacher conditions on.

The repository is a fork of [verl](https://github.com/volcengine/verl) that adds the SDPO objective
from [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)
([lasgroup/SDPO](https://github.com/lasgroup/SDPO)), a feedback-conditioned teacher, and the Apertus
training scripts. Everything else — FSDP/Megatron backends, vLLM rollout, Ray orchestration, the
[upstream docs](https://verl.readthedocs.io) — behaves as in verl.

## Resources

- [Apertus model collection](https://huggingface.co/collections/swiss-ai/apertus) — the released checkpoints
- [Apertus tech report](https://github.com/swiss-ai/apertus-tech-report) — model family, data, and evaluation
- [SDPO paper](https://arxiv.org/abs/2601.20802) — the self-distillation objective
- [verl](https://github.com/volcengine/verl) — the RL framework this builds on

---

## How it works

One training step, as orchestrated by `RayPPOTrainer.fit()`:

**1. Rollout.** The policy generates `rollout.n` responses per prompt via vLLM. Standard verl.

**2. Critique.** Each response goes through verl's custom-reward hook, which dispatches to a
*feedback scorer* selected per row by `reward_model.scoring_module`. Scorers return **text plus a
scalar**, and constitutional-AI runs use the `llm` scorer: it calls an OpenAI-compatible endpoint
with the constitution and the response, and returns the judge's critique. Its scalar is deliberately
fixed below `success_reward_threshold` so no rollout is ever promoted to a demonstration — text is
the only channel.

**3. Teacher reprompt.** A second prompt is rendered per sample — the original request with the
critique substituted into `reprompt_template` / `feedback_template`. The tokenized teacher batch is
unioned into the training batch alongside a `self_distillation_mask` marking which samples were
reprompted.

**4. Distillation.** The actor runs a second, no-grad forward over the teacher batch and computes a
masked divergence between student and teacher over the student's own tokens. Over full logits —
optionally truncated to the top `distillation_topk` — `alpha` interpolates from forward KL (`0.0`)
through a generalized JSD to reverse KL (`1.0`); the cheaper sampled-token mode
(`full_logit_distillation: False`, which the Apertus run uses) supports reverse KL only. An
importance-ratio clip (`is_clip`) applies in both.

The teacher shares weights with the policy. `teacher_update_rate: 0.0` freezes it at the initial
policy; a non-zero value makes it an EMA of the student.

| Stage | Implementation |
|---|---|
| orchestration, teacher reprompt | `_maybe_build_self_distillation_batch()` in `verl/trainer/ppo/ray_trainer.py` |
| the objective | `compute_self_distillation_loss()` in `verl/trainer/ppo/core_algos.py` |
| teacher forward, EMA update | `DataParallelPPOActor.update_policy()` in `verl/workers/actor/dp_actor.py` |
| feedback scorers | `verl/utils/reward_score/feedback/` (`llm`, `math`, `mcq`, `zero`) |

---

## Installation

Training runs inside a container image with a shared virtual environment built once, ahead of the
job. Without that step every node of every job runs its own `pip install`, which intermittently
fails on shared NFS and leaves nodes on mismatched dependency versions.

On CSCS Alps, declare the container environment as `~/.edf/sd.toml`:

```toml
image = "/capstor/store/cscs/swissai/infra01/env/sd-v0.4.sqsh"

mounts = ["/capstor", "/iopsstor", "/users", "/tmp"]
```

then build the venv inside it:

```bash
./experiments/constitutional_ai/build_env.sh
```

Elsewhere, install the package as usual (`pip install -e .`) into an environment with a CUDA PyTorch
build; see verl's [installation guide](https://verl.readthedocs.io/en/latest/start/install.html) for
backend-specific requirements.

## Quickstart

The constitutional-AI recipe, end to end:

```bash
# 1. one-time environment build (see above)
./experiments/constitutional_ai/build_env.sh

# 2. prepare training prompts
python data/preprocess/gretel_safety_dataset.py \
    --output-dir datasets/gretel-safety-alignment

# 3. host the judge as a separate job
./experiments/constitutional_ai/launch_judge.sh

# 4. submit training
./experiments/constitutional_ai/run_sdpo.sh
```

`run_sdpo.sh` submits a single Slurm job and, on completion, merges the FSDP shards to Hugging Face
format under `$SCRATCH/checkpoints/$PROJECT_NAME/$EXP_NAME/hf/`. Pass `--dry-run` to print the
`sbatch` command without submitting it. Trailing arguments are forwarded as Hydra overrides:

```bash
./experiments/constitutional_ai/run_sdpo.sh \
    actor_rollout_ref.actor.self_distillation.alpha=0.5
```

Every cluster and hyperparameter setting in the script is an environment variable with a default
(`MODEL_PATH`, `LR`, `TRAIN_BATCH_SIZE`, `TRAIN_NODES`, `ACCOUNT`, `RESERVATION`, …). The committed
defaults are those of the released Apertus-1.5-8B constitutional-AI run, so the commands above
reproduce it — set at least `MODEL_PATH` and `ACCOUNT` for your own environment.

### Judge endpoint

Training needs an OpenAI-compatible `/v1/chat/completions` endpoint and nothing more.
`launch_judge.sh` is a CSCS-specific convenience wrapper around
[`sml`](https://github.com/swiss-ai/model-launch); to serve the judge some other way, skip it and set
`LLM_JUDGE_BASE_URL`, `LLM_JUDGE_MODEL` and `LLM_JUDGE_API_KEY` (the last in `.env`, see
`.env.example`). `run_sdpo.sh` blocks until the endpoint answers a ping and aborts rather than
holding the allocation open.

### Data

Preprocessing scripts write `train.parquet` / `test.parquet` in the canonical schema: `prompt` (chat
messages), `extra_info`, `reward_model` (`ground_truth` and `scoring_module`), and an optional
`answer`. `user.yaml` resolves them from the `TASK` environment variable. `datasets/` is gitignored —
regenerate rather than commit.

---

## Configuration

| To change | Edit |
|---|---|
| the constitution | `judge_prompt_template` in `verl/trainer/config/sdpo_cai.yaml` |
| how feedback reaches the teacher | `reprompt_template` / `feedback_template`, same file |
| the judge model or endpoint | `JUDGE_*` in `run_sdpo.sh`; `LLM_JUDGE_*` environment variables |
| base model, learning rate, batch sizes, nodes | environment variables read by `run_sdpo.sh` |
| the objective | `self_distillation.*` in `verl/trainer/config/actor/actor.yaml` |
| the training prompts | a script in `data/preprocess/`; point `DATA_PATH` at its output |

Config layering is Hydra's: `sdpo_cai.yaml` extends `sdpo.yaml`, which extends `user.yaml` and verl's
`ppo_trainer.yaml`. Every self-distillation knob is documented inline in `actor/actor.yaml`.

### Adding a feedback source

The judge is just a scorer. Add a module under `verl/utils/reward_score/feedback/` exposing
`compute_score(...) -> dict` with a `feedback` key, register it in that package's `__init__.py`, and
emit its name as `reward_model.scoring_module` in your parquet rows. Nothing in the trainer changes.

### Changing the training signal

`sdpo_cai.yaml` routes judge text into the teacher prompt, but that is one of several sources the
teacher can condition on. The same machinery supports a privileged dataset answer
(`use_dataset_expert_answer`), a peer rollout that scored above `success_reward_threshold`, or a
fixed context paragraph for pure context distillation (`always_apply_reprompt`) — see the documented
knobs in `actor/actor.yaml`.

---

## Repository layout

```
experiments/constitutional_ai/
    run_sdpo.sh              submits the training job (Slurm)
    launch_judge.sh          hosts the judge as a separate job
    build_env.sh             one-time venv build inside the container
training/verl_training.sh    thin entrypoint into verl.trainer.main_ppo

verl/trainer/config/
    sdpo_cai.yaml            constitution, judge prompt, teacher templates
    sdpo.yaml                the SDPO objective defaults
    user.yaml                dataset wiring, rollout/logging defaults
    actor/actor.yaml         every self_distillation knob, documented inline

verl/trainer/ppo/
    ray_trainer.py           rollout -> judge -> teacher reprompt orchestration
    core_algos.py            compute_self_distillation_loss (the objective)
verl/workers/actor/dp_actor.py     teacher forward + loss dispatch
verl/utils/reward_score/feedback/  scoring modules; llm.py is the judge client

data/preprocess/             dataset -> canonical parquet
docs/apertus/operations.md   cluster tuning and base-checkpoint preparation
```

---

## Outputs

| Artifact | Location |
|---|---|
| Slurm logs | `$OUTPUT_DIR/<jobid>.{log,err}` |
| rollouts, teacher prompts, judge feedback | `$OUTPUT_DIR/generations/$EXP_NAME/rollouts` |
| checkpoints | `$SCRATCH/checkpoints/$PROJECT_NAME/$EXP_NAME/`, merged under `hf/` |
| metrics | Weights & Biases, `$WANDB_ENTITY/$PROJECT_NAME` |

The rollout dumps contain the fully rendered teacher prompt and the judge's critique for each sample.
Inspecting them is the fastest way to confirm that the constitution is reaching the judge and the
feedback is reaching the teacher — worth doing on the first run after changing any template.

Validation runs GSM8K and MMLU as a capability-regression check (the held-out split itself is skipped
via `trainer.skip_dataset_validation`). `trainer.benchmarks.*` also covers GPQA-Diamond, MMLU-Pro,
MedQA and MedMCQA.

Cluster tuning and base-checkpoint preparation — including two metadata repairs that silently degrade
results when skipped — are documented in
[docs/apertus/operations.md](docs/apertus/operations.md).

---

## License and attribution

Apache 2.0 — see [LICENSE](LICENSE) and [Notice.txt](Notice.txt).

Derived from [verl](https://github.com/volcengine/verl) (Copyright 2023–2024 Bytedance Ltd.) and from
the SDPO implementation in [lasgroup/SDPO](https://github.com/lasgroup/SDPO). The constitutional-AI
training path, the judge-feedback reward modules, and the Apertus experiment scripts are the
additions here.
