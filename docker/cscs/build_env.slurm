#!/bin/bash
#SBATCH --job-name=build-env
#SBATCH -A a156
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Directory that contains the Dockerfile
RECIPE_DIR="/users/antonbaumann/self-distillation/docker/cscs"

# Tag for the podman image (local to this job)
IMAGE_TAG="sd"

# Where to store the SquashFS image used by the Container Engine
SQSH_DIR="/capstor/store/cscs/swissai/a156/container-images/env"
SQSH_IMAGE="$SQSH_DIR/sd-v0.4.sqsh"
echo "Using RECIPE_DIR=${RECIPE_DIR}"
echo "Will store SquashFS image at ${SQSH_IMAGE}"

mkdir -p "$SQSH_DIR"

# Build image with podman
srun bash -lc "
  cd \"$RECIPE_DIR\" && \
  echo 'Building podman image...' && \
  podman build -t \"$IMAGE_TAG\" . && \
  echo 'Importing image to SquashFS with enroot...' && \
  enroot import -x mount -o \"$SQSH_IMAGE\" podman://\"$IMAGE_TAG\"
"
