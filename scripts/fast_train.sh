#!/bin/bash

# Fastest Path to Chatting with nanochat
# Total time: ~13-14 hours on RTX 4060

set -e

# -----------------------------------------------------------------------------
# Configuration
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export TORCH_COMPILE_DISABLE=1

# Navigate to project root
cd /mnt/c/code/nanochat/nanochat
source .venv/bin/activate

echo "=============================================="
echo "FASTEST PATH TO CHATTING"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# PHASE 1: Base Training (500 steps, ~9 hours)
echo "=============================================="
echo "PHASE 1: Base Training (500 steps)"
echo "Estimated time: ~9 hours"
echo "=============================================="

python -u -m scripts.base_train \
    --depth=8 \
    --device_batch_size=2 \
    --max_seq_len=1024 \
    --save_every=250 \
    --core_metric_every=-1 \
    --sample_every=500 \
    --num_iterations=500

echo ""
echo "✅ Base training complete!"
echo ""

# -----------------------------------------------------------------------------
# PHASE 2: Midtraining (~2-3 hours)
echo "=============================================="
echo "PHASE 2: Midtraining"
echo "Estimated time: ~2-3 hours"
echo "=============================================="

# Download identity conversations
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -u -m scripts.mid_train

echo ""
echo "✅ Midtraining complete!"
echo ""

# -----------------------------------------------------------------------------
# PHASE 3: SFT (~1-2 hours)
echo "=============================================="
echo "PHASE 3: Supervised Fine-Tuning"
echo "Estimated time: ~1-2 hours"
echo "=============================================="

python -u -m scripts.chat_sft

echo ""
echo "✅ SFT complete!"
echo ""

# -----------------------------------------------------------------------------
# DONE! Launch chat
echo "=============================================="
echo "🎉 ALL TRAINING COMPLETE!"
echo "=============================================="
echo ""
echo "Your model is ready! Launch chat with:"
echo ""
echo "  python -m scripts.chat_cli"
echo ""
echo "Or try a quick test:"
echo ""

python -m scripts.chat_cli -p "Hello! What can you help me with today?"

echo ""
echo "=============================================="
echo "To chat interactively, run:"
echo "  python -m scripts.chat_cli"
echo ""
echo "For web UI, run:"
echo "  python -m scripts.chat_web"
echo "=============================================="