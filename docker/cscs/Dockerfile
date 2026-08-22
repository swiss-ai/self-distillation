# Base: NGC vLLM container for aarch64 with CUDA 13.1, Python 3.12
# FROM nvcr.io/nvidia/vllm:25.12.post1-py3
FROM nvcr.io/nvidia/vllm:26.02-py3

ARG DEBIAN_FRONTEND=noninteractive

# Unset ROCR_VISIBLE_DEVICES to prevent conflict with CUDA_VISIBLE_DEVICES
# (cluster sets both, but verl requires only one)
ENV ROCR_VISIBLE_DEVICES=""

WORKDIR /app

# Copy requirements file first (for layer caching)
COPY requirements-gh200.txt /app/requirements-gh200.txt

# Install Python dependencies (excluding NGC pre-installed packages)
RUN pip install --no-cache-dir -r requirements-gh200.txt

# Default command
CMD ["/bin/bash"]