# Use existing ROCm-PyTorch image
FROM rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/venv/bin:$PATH" \
    VIRTUAL_ENV="/venv"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Create virtual environment with system site packages
RUN python3 -m venv --system-site-packages /venv

# Set working directory
WORKDIR /app

# Install project dependencies in chunks to facilitate caching and progress
RUN uv pip install transformers accelerate huggingface-hub timm
RUN uv pip install peft
RUN uv pip install anomalib
RUN uv pip install albumentations scikit-learn onnx
# Copy the rest of the project
COPY . .

# Install project
RUN uv pip install -e .

# Default command
CMD ["bash"]
