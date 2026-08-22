# Operational notes

Cluster and checkpoint details for running the constitutional-AI training. Defaults in
`experiments/constitutional_ai/run_sdpo.sh` already encode all of this; this document explains *why*,
so the settings survive being changed.

## Preparing a base checkpoint

Apertus checkpoints exported by recent tooling need two metadata repairs before this pipeline can use
them. **Both fail silently** — training runs to completion and produces a worse model, or dies on
every node at startup.

**RoPE.** Checkpoints written by transformers 5 store RoPE under `rope_parameters`. This pipeline runs
transformers 4, which cannot read that key, finds no legacy `rope_theta` / `rope_scaling`, and falls
back to Apertus-1 defaults (`rope_theta=12e6, factor=8` instead of `4e6, 32`). Training proceeds
normally and the results are wrong.

Fix by rewriting `config.json` in tf4-legacy style — top-level `rope_theta: 4000000` plus

```json
"rope_scaling": {
  "rope_type": "llama3", "type": "llama3", "factor": 32.0,
  "high_freq_factor": 4.0, "low_freq_factor": 1.0,
  "original_max_position_embeddings": 8192
}
```

and no `rope_parameters` key. Weights are untouched; this is pure metadata, so reshards and hardlinks
of the same safetensors stay valid.

**Tokenizer class.** Apertus exports ship `tokenizer_class: "TokenizersBackend"`, a transformers-5
class. vLLM loads it fine — so a judge started on such a checkpoint looks healthy — but verl calls
`AutoTokenizer` and dies with `ValueError: Tokenizer class TokenizersBackend does not exist`, taking
down every node at startup. Fix:

```bash
sed -i 's/"TokenizersBackend"/"PreTrainedTokenizerFast"/' tokenizer_config.json
```

`PreTrainedTokenizerFast` loads the same `tokenizer.json` and works in both verl and vLLM.

**Verify before submitting a multi-node job.** Two minutes here saves a whole allocation:

```bash
srun --environment=sd --nodes=1 --time=00:05:00 python -c "
from transformers import AutoTokenizer, AutoConfig
p = '$MODEL_PATH'
print(type(AutoTokenizer.from_pretrained(p)).__name__)
c = AutoConfig.from_pretrained(p); print(c.rope_theta, c.rope_scaling)"
# want: PreTrainedTokenizerFast, 4000000, factor 32.0
```

## Cluster settings

Two overrides in `run_sdpo.sh` are load-bearing:

- **`actor_rollout_ref.nccl_timeout=7200`.** The 600s default is shorter than a large checkpoint load,
  which then fails as a gloo init-barrier timeout rather than as a load error.
- **`actor_rollout_ref.rollout.gpu_memory_utilization=0.5`.** This is a floor, not a tuning knob.
  Below it, vLLM passes init but dies roughly 25 minutes in: once optimizer state materializes after
  the first actor update, vLLM's KV re-init fails on the next rollout wake. Above it, the actor update
  itself OOMs. Runs with `use_kl_loss=True` are tighter still and need ~0.48.

**Checkpoint load speed.** Slow loads usually mean the safetensors are striped across a single OST
(`lfs getstripe -c` returns 1), so every rank contends on one server. Re-stripe wide and point
`MODEL_PATH` at the copy:

```bash
mkdir DST && lfs setstripe -c 16 DST && cp -a SRC/. DST/
```

Shard *size* matters too: ~5GB shards are what the rest of the pipeline produces. Very large shards
(~49GB) leave a much bigger caching-allocator residual per GPU, which starves vLLM — re-shard to ~5GB
rather than lowering `gpu_memory_utilization`.

## Reading job outcomes

**One epoch is the real limiter, not `total_training_steps`.** A single pass over the dataset ends
training regardless of the nominal step budget, so a run that stops early may have completed
normally — compute `dataset_size / train_batch_size` before concluding anything was cut short.

A successful job logs `All done. Outputs in .../hf` and then cancels its own allocation, so `sacct`
shows **`CANCELLED+` with exit code `0:0` — that is success**. Real failures show no `All done` line,
plus a maximum step below the epoch length.

**Container mount failures are usually one bad node.** `pyxis: Failed to mount ... exit code 137 ...
filesystem seems too slow/overloaded` is misleading: exit 137 means the mount helper was OOM-killed on
one specific node. The head node reaches "Waiting for N Ray nodes to join" and no workers appear. Find
the culprit with `grep -oE 'nid[0-9]+: task [0-9]+: Exited with exit code 1' <job>.err | head -1`; if
it is the same node each time, exclude it. Note that `SBATCH_EXCLUDE` is ignored on this system —
`run_sdpo.sh` therefore takes `EXCLUDE_NODES` and passes an explicit `--exclude`.
