#!/bin/bash
# Resume from base checkpoint step 250 -> midtraining -> SFT -> chat
# Optimized for RTX 4060 (8GB)

set -e

cd /mnt/c/code/nanochat/nanochat
source .venv/bin/activate

# Disable torch.compile to avoid OOM
export TORCH_COMPILE_DISABLE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Download identity conversations
echo "Downloading identity conversations..."
curl -L -o ~/.cache/nanochat/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# =============================================================================
# PHASE 1: Midtraining (uses step 250 base checkpoint automatically)
# =============================================================================
echo ""
echo "=============================================="
echo "MIDTRAINING (from base step 250)"
echo "=============================================="

python -u -m scripts.mid_train \
    --device_batch_size=4 \
    --max_seq_len=1024 \
    --num_iterations=500 \
    --eval_every=-1 \
    --save_every=100

# =============================================================================
# PHASE 2: SFT
# =============================================================================
echo ""
echo "=============================================="
echo "SUPERVISED FINE-TUNING"
echo "=============================================="

python -u -m scripts.chat_sft \
    --device_batch_size=2 \
    --max_tokens=1024 \
    --eval_every=-1 \
    --num_iterations=200 \
    --save_every=100

# =============================================================================
# PHASE 3: Chat!
# =============================================================================
echo ""
echo "=============================================="
echo "🎉 TRAINING COMPLETE! Testing chat..."
echo "=============================================="

python -m scripts.chat_cli -p "Hello! What can you help me with today?"

echo ""
echo "To chat interactively, run:"
echo "  python -m scripts.chat_cli"
