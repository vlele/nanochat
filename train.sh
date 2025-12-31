#!/bin/bash

# Base Model Training Script for WSL2 with GPU
# Run AFTER train_tokenizer.sh has completed
# 
# Usage:
#   ./train_base.sh              # Use defaults (auto-detect GPUs)
#   ./train_base.sh --depth=12   # Smaller model
#   NGPU=1 ./train_base.sh       # Force single GPU

set -e

# -----------------------------------------------------------------------------
# Configuration
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Navigate to project root
cd "$(dirname "$0")"
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must run from nanochat project root"
    exit 1
fi

# -----------------------------------------------------------------------------
# Activate environment (assumes train_tokenizer.sh already ran)
source .venv/bin/activate

# Verify tokenizer exists
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Error: Tokenizer not found. Run train_tokenizer.sh first!"
    exit 1
fi

# -----------------------------------------------------------------------------
# Download full dataset for pretraining

echo "Downloading dataset shards for pretraining..."
echo "**CHanged This downloads ~2.4GB of data (20 shards @ ~100MB each)"
python -m nanochat.dataset -n 20

# -----------------------------------------------------------------------------
# Detect GPU configuration

# Auto-detect number of GPUs (or use NGPU env var)
if [ -z "$NGPU" ]; then
    NGPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
fi

echo ""
echo "=============================================="
echo "GPU Configuration"
echo "=============================================="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)')
"
echo "Using $NGPU GPU(s) for training"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Set training parameters based on GPU count

if [ "$NGPU" -eq 0 ]; then
    echo "No GPU detected! Training on CPU (very slow, tiny model only)"
    TRAIN_CMD="python -m scripts.base_train"
    # Tiny model for CPU testing
    EXTRA_ARGS="--depth=4 --max_seq_len=512 --device_batch_size=1 --total_batch_size=512 --num_iterations=100 --core_metric_every=-1"
elif [ "$NGPU" -eq 1 ]; then
    echo "Single GPU mode"
    TRAIN_CMD="python -m scripts.base_train"
    # Adjust batch size for single GPU (smaller to avoid OOM)
    # depth=12 is ~200M params, more manageable for single GPU
    EXTRA_ARGS="--depth=8 --device_batch_size=2 --max_seq_len=1024"
else
    echo "Multi-GPU mode with $NGPU GPUs"
    TRAIN_CMD="torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train --"
    # Full d20 model (561M params) for multi-GPU
    EXTRA_ARGS="--depth=20 --device_batch_size=32"
fi

# -----------------------------------------------------------------------------
# Run base model pretraining

echo ""
echo "Starting base model pretraining..."
echo "Command: $TRAIN_CMD  $EXTRA_ARGS $@"
echo ""

$TRAIN_CMD  $EXTRA_ARGS "$@"

# -----------------------------------------------------------------------------
# Run evaluation

echo ""
echo "Running loss evaluation..."
if [ "$NGPU" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_loss
    torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_eval
else
    python -m scripts.base_loss
    python -m scripts.base_eval
fi

echo ""
echo "✅ Base model training complete!"
echo "   Checkpoint saved to: $NANOCHAT_BASE_DIR/base_checkpoints/"