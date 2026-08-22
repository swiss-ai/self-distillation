#!/usr/bin/env bash
# Merge all FSDP-sharded verl checkpoints in a directory into HF-loadable dirs.
#
# Usage:
#   ./merge_fsdp_checkpoint.sh <checkpoints_root> <output_dir>
#
# <checkpoints_root> should contain one or more `global_step_*` subdirectories
# (each with an `actor/` folder). Each is merged into
# <output_dir>/<step_dirname>/.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <checkpoints_root> <output_dir>" >&2
    exit 1
fi

root="${1%/}"
output_dir="${2%/}"

if [[ ! -d "$root" ]]; then
    echo "Error: $root is not a directory" >&2
    exit 1
fi

mkdir -p "$output_dir"

shopt -s nullglob
steps=("$root"/global_step_*)
shopt -u nullglob

if [[ ${#steps[@]} -eq 0 ]]; then
    echo "Error: no global_step_* subdirectories found in $root" >&2
    exit 1
fi

echo "Found ${#steps[@]} checkpoint(s) in $root"

for ckpt in "${steps[@]}"; do
    name="$(basename "$ckpt")"
    actor_dir="$ckpt/actor"
    target="$output_dir/$name"

    if [[ ! -d "$actor_dir/huggingface" ]]; then
        echo "Skipping $name: missing $actor_dir/huggingface" >&2
        continue
    fi

    if [[ -f "$target/config.json" ]]; then
        echo "Skipping $name: $target already merged"
        continue
    fi

    echo "=== Merging $name ==="
    mkdir -p "$target"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$target"
done

echo "All done. Outputs in $output_dir"
