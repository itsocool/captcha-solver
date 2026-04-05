# Use the Astral UV image for Python 3.12 + uv package manager
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sudo \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user to match Dev Container expectations
RUN groupadd -g 1000 vscode || true \
    && useradd -m -u 1000 -g vscode -s /bin/bash vscode || true \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

# Python dependencies are installed from pyproject.toml in postCreate via `uv sync`

USER vscode
