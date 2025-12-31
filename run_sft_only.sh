#!/bin/bash
# Run SFT only (assumes base and midtraining are already complete)
# Optimized for RTX 4060 (8GB)

set -e

cd /mnt/c/code/nanochat/nanochat
source .venv/bin/activate

# Disable torch.compile to avoid OOM
export TORCH_COMPILE_DISABLE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# =============================================================================
# SFT (Supervised Fine-Tuning)
# =============================================================================
echo ""
echo "=============================================="
echo "SUPERVISED FINE-TUNING"
echo "=============================================="
echo ""
echo "This will load the model from midtraining checkpoint"
echo "and run SFT for 200 steps (~20-30 minutes)"
echo ""

python -u -m scripts.chat_sft \
    --device_batch_size=2 \
    --max_tokens=1024 \
    --eval_every=-1 \
    --num_iterations=200 \
    --save_every=100

echo ""
echo "=============================================="
echo "✅ SFT COMPLETE!"
echo "=============================================="
echo ""
echo "Your model is ready! Chat with it using:"
echo ""
echo "  python -m scripts.chat_cli"
echo ""
echo "Or start the web UI:"
echo ""
echo "  python -m scripts.chat_web"
echo ""

