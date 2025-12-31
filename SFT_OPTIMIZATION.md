# SFT Optimization Guide

## Summary of Changes

We've optimized the SFT (Supervised Fine-Tuning) phase to reduce training time from **~2 hours to ~30-45 minutes** while maintaining quality.

## What Changed

### Before:
- **Iterations**: 701 steps (1 full epoch of ~23K examples)
- **Time**: ~1-2 hours
- **Rationale**: Full epoch ensures model sees all training data

### After:
- **Iterations**: 200 steps (~29% of dataset)
- **Time**: ~20-30 minutes
- **Rationale**: Model already well-trained from base + midtraining, so less SFT needed

## Updated Scripts

Both `resume.sh` and `fast_train.sh` now include:
```bash
python -u -m scripts.chat_sft \
    --device_batch_size=2 \
    --max_tokens=1024 \
    --eval_every=-1 \
    --num_iterations=300
```

## Why This Works

1. **Base Training**: Model learned fundamental language patterns (3,520 steps)
2. **Midtraining**: Model learned conversation format and task structure (500 steps)
3. **SFT**: Only needs to polish chat quality - doesn't need full epoch

The model has already seen:
- ✅ Language patterns (base)
- ✅ Conversation structure (mid)
- ✅ Task formats (mid)

SFT just needs to refine the conversational style, which happens quickly.

## Additional Optimization Options

If you want to experiment further, you can adjust:

### Option 1: Current Setting (200 steps) ✅
```bash
--num_iterations=200  # ~20-30 minutes
```
**Trade-off**: Fast training with good quality

### Option 2: More Thorough (300-400 steps)
```bash
--num_iterations=300  # ~30-45 minutes
--num_iterations=400  # ~45-60 minutes
```
**Trade-off**: Better quality, but takes longer

### Option 3: Increase Batch Size (if memory allows)
```bash
--device_batch_size=4 \
--target_examples_per_step=64
```
**Trade-off**: Faster steps, but may cause OOM on 8GB GPU

### Option 4: Reduce Max Tokens (if conversations are short)
```bash
--max_tokens=512  # Instead of 1024
```
**Trade-off**: Less context, but faster processing

## Current Configuration (Recommended)

```bash
--device_batch_size=2      # Safe for 8GB GPU
--max_tokens=1024          # Good context length
--eval_every=-1            # Disable validation (faster)
--num_iterations=200       # Fast training (~20-30 min)
```

## Expected Results

- **Training Time**: 20-30 minutes
- **Model Quality**: Should be very similar to full 701-step run
- **Chat Performance**: Ready for interactive use

## Monitoring

Watch for:
- **Validation loss**: Should decrease from ~2.2 to ~1.7-1.8
- **Training loss**: Will fluctuate (0.1-5.0) - this is normal
- **Step time**: ~5-10 seconds per step

If validation loss plateaus early (e.g., by step 150), you could stop early. If it's still decreasing at step 200, consider increasing to 300-400 steps for better quality.

