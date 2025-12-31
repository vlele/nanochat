#!/bin/bash

# Tokenizer Training Script for WSL
# Run from the nanochat project root directory

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python environment setup with uv

# Install uv if not already installed
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true

# Create virtual environment if it doesn't exist
[ -d ".venv" ] || uv venv

# Install dependencies (gpu extras for CUDA support)
uv sync --extra gpu

# Activate the virtual environment
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Install Rust and build the rustbpe tokenizer

# Install Rust if not already installed
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

# Build the Rust BPE tokenizer as a Python module
echo "Building rustbpe tokenizer..."
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# -----------------------------------------------------------------------------
# Download dataset (just enough for tokenizer training)

echo "Downloading 8 data shards (~800MB) for tokenizer training..."
python -m nanochat.dataset -n 8

# -----------------------------------------------------------------------------
# Train the tokenizer

echo "Training tokenizer on ~2B characters..."
python -m scripts.tok_train --max_chars=2000000000

# -----------------------------------------------------------------------------
# Evaluate the tokenizer

echo "Evaluating tokenizer..."
python -m scripts.tok_eval

echo ""
echo "✅ Tokenizer training complete!"
echo "   Saved to: $NANOCHAT_BASE_DIR/tokenizer/"