# Training Safety Guide

## ✅ Checkpointing Strategy

### Base Training
- **Checkpoint frequency**: Every 250 steps (via `--save_every=250`)
- **Location**: `~/.cache/nanochat/base_checkpoints/d8/`
- **What's saved**: Model weights, optimizer states, metadata, dataloader state
- **Resume capability**: ✅ Yes - can resume from any saved step

### Midtraining
- **Checkpoint frequency**: Every 100 steps (via `--save_every=100`)
- **Location**: `~/.cache/nanochat/mid_checkpoints/d8/`
- **What's saved**: Model weights, optimizer states, metadata
- **Resume capability**: ✅ Yes - can resume from any saved step

### SFT (Supervised Fine-Tuning)
- **Checkpoint frequency**: Every 100 steps (via `--save_every=100`)
- **Location**: `~/.cache/nanochat/chatsft_checkpoints/d8/`
- **What's saved**: Model weights, metadata (no optimizer state - not needed for SFT)
- **Resume capability**: ⚠️ Limited - SFT doesn't save optimizer state, but can load final checkpoint

### Checkpoint File Format
- `model_XXXXXX.pt` - Model weights
- `meta_XXXXXX.json` - Training metadata (step, loss, config)
- `optim_XXXXXX_rank0.pt` - Optimizer state (base/mid only)

## 🛡️ OOM (Out of Memory) Prevention

### Current Safe Configuration (RTX 4060 8GB)

#### Base Training
```bash
--depth=8                    # Reduced from 12 (saves ~50% memory)
--device_batch_size=2       # Reduced from 8 (saves ~75% memory)
--max_seq_len=1024           # Reduced from 2048 (saves ~50% memory)
--core_metric_every=-1       # Disabled CORE evaluation (saves memory)
```

**Memory usage**: ~4-4.5GB / 8GB (safe margin)

#### Midtraining
```bash
--device_batch_size=4        # Safe for midtraining (smaller model context)
--max_seq_len=1024          # Matches base training
--eval_every=-1              # Disabled evaluation (saves memory)
```

**Memory usage**: ~4-5GB / 8GB (safe margin)

#### SFT
```bash
--device_batch_size=2        # Conservative for variable-length conversations
--max_tokens=1024            # Matches previous phases
--eval_every=-1              # Disabled validation (saves memory)
```

**Memory usage**: ~3-4GB / 8GB (safe margin)

### Memory Safety Measures

1. **PyTorch Memory Management**
   ```bash
   export PYTORCH_ALLOC_CONF="expandable_segments:True"
   ```
   - Allows dynamic memory allocation
   - Reduces fragmentation

2. **Mixed Precision (bfloat16)**
   - Default dtype: `bfloat16`
   - Saves ~50% memory vs float32
   - Maintains training stability

3. **Gradient Accumulation**
   - Automatically calculated to maintain effective batch size
   - Allows small device batches while keeping training stable

4. **No torch.compile for Mid/SFT**
   - Disabled via `export TORCH_COMPILE_DISABLE=1`
   - Reduces memory overhead during compilation
   - Prevents OOM during kernel compilation

### If OOM Occurs

1. **Immediate Actions**:
   - Training will crash with `RuntimeError: CUDA driver error: out of memory`
   - Checkpoint from last `save_every` interval is safe

2. **Recovery Steps**:
   ```bash
   # For base training - resume from last checkpoint
   python -m scripts.base_train \
       --depth=8 \
       --device_batch_size=2 \
       --max_seq_len=1024 \
       --resume_from_step=250  # Use last saved step
   
   # For midtraining - reduce batch size further
   python -m scripts.mid_train \
       --device_batch_size=2 \  # Reduced from 4
       --max_seq_len=1024 \
       --resume_from_step=100   # Use last saved step
   
   # For SFT - reduce batch size
   python -m scripts.chat_sft \
       --device_batch_size=1 \  # Reduced from 2
       --max_tokens=512          # Reduced from 1024
       # Note: SFT doesn't support resume, but can load final checkpoint
   ```

3. **Prevention**:
   - Monitor with `watch -n 1 nvidia-smi`
   - If memory > 7GB, reduce `device_batch_size` or `max_seq_len`
   - Keep `save_every` frequent (100 steps) to minimize loss

## 🔍 Monitoring & Diagnostics

### GPU Memory Monitoring
```bash
# Continuous monitoring
watch -n 1 nvidia-smi

# Check specific process
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### Training Progress Monitoring
```bash
# Check if training is running
ps aux | grep python

# Monitor checkpoint directory
ls -lh ~/.cache/nanochat/*/d8/
```

### Expected Memory Usage
- **Idle**: ~0-100MB
- **Base Training**: ~4-4.5GB
- **Midtraining**: ~4-5GB
- **SFT**: ~3-4GB
- **Evaluation**: +1-2GB (if enabled)

## ⚠️ Other Safety Considerations

### 1. Disk Space
- **Checkpoint size**: ~350MB per checkpoint (base/mid), ~350MB (SFT)
- **Total storage needed**: ~2-3GB for all checkpoints
- **Monitor**: `df -h ~/.cache/nanochat/`

### 2. Training Interruption
- **Ctrl+C**: Safe - last checkpoint is preserved
- **System crash**: Safe - last `save_every` checkpoint is preserved
- **Power loss**: Safe - last `save_every` checkpoint is preserved

### 3. WSL2 Specific Issues
- **Line endings**: Use `sed -i 's/\r$//' script.sh` if needed
- **nvidia-smi**: May not show processes (WSL2 quirk) - use `ps aux` instead
- **CUDA initialization**: May take 30-60 seconds on first run

### 4. Checkpoint Cleanup
```bash
# List all checkpoints
ls ~/.cache/nanochat/base_checkpoints/d8/
ls ~/.cache/nanochat/mid_checkpoints/d8/
ls ~/.cache/nanochat/chatsft_checkpoints/d8/

# Keep only last N checkpoints (example: keep last 3)
cd ~/.cache/nanochat/base_checkpoints/d8/
ls -t model_*.pt | tail -n +4 | xargs rm -f
```

## 📊 Checkpoint Strategy Summary

| Phase | Save Every | Total Steps | Checkpoints | Max Loss if Crash |
|-------|------------|-------------|-------------|-------------------|
| Base | 250 | 3,520 | ~14 | 250 steps (~7%) |
| Mid | 100 | 500 | ~5 | 100 steps (~20%) |
| SFT | 100 | 200 | ~2 | 100 steps (~50%) |

**Note**: SFT has higher risk because it's shorter, but checkpoints every 100 steps still provide good safety.

## ✅ Verification Checklist

Before starting training, verify:
- [ ] GPU memory available: `nvidia-smi` shows > 7GB free
- [ ] Disk space available: `df -h` shows > 5GB free
- [ ] Checkpoint directories exist: `ls ~/.cache/nanochat/`
- [ ] `save_every` is set appropriately (100-250 steps)
- [ ] `device_batch_size` is conservative (2-4 for 8GB GPU)
- [ ] `max_seq_len`/`max_tokens` is reasonable (1024)
- [ ] Memory allocation config is set: `PYTORCH_ALLOC_CONF="expandable_segments:True"`

## 🚨 Emergency Recovery

If training crashes:

1. **Identify last checkpoint**:
   ```bash
   ls -t ~/.cache/nanochat/*/d8/model_*.pt | head -1
   ```

2. **Extract step number from filename**:
   - `model_000250.pt` → step 250

3. **Resume training** (base/mid only):
   ```bash
   python -m scripts.base_train --resume_from_step=250 ...
   ```

4. **For SFT**: Load final checkpoint manually if needed (no resume support)

## 📝 Best Practices

1. **Always use `save_every`**: Never set to -1 or 0 for long training
2. **Monitor regularly**: Check `nvidia-smi` every 30-60 minutes
3. **Keep checkpoints**: Don't delete old checkpoints until training is complete
4. **Test batch size**: Start conservative, increase if memory allows
5. **Use resume scripts**: The provided `resume.sh` handles checkpoint loading automatically

