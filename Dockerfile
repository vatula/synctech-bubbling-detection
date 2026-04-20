# Use existing ROCm-PyTorch image
FROM rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy dependency manifests
COPY pyproject.toml uv.lock ./

# Install dependencies (use cached layer)
# Ensure we install the project itself.
RUN uv sync --frozen

# Copy the source code
COPY . .

# Ensure scripts are executable at build time
RUN chmod +x *.sh

# Default command
CMD ["bash"]
