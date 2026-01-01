
**Observations:**

- The model loads successfully and generates text
- It demonstrates understanding of the prompt (mentions AI, learning, adaptation)
- The response shows the model's limitations: it's repetitive and doesn't fully answer the question about "three stages of learning."
- This is expected for a 92M parameter model trained for only 200 SFT steps

For a model of this size, the responses are reasonable. With more training or a larger model, quality would improve significantly.

## Making Interactive Chat Work: PyTorch Compatibility Fixes

Getting chat_cli working required a couple of small compatibility shims for older PyTorch:

- RMSNorm: some versions don’t have torch.nn.functional.rms_norm(), so I added a lightweight fallback.
- GQA: Some versions don’t accept enable_gqa in scaled_dot_product_attention(), so I only pass it when supported (otherwise the code falls back to the non-enable_gqa path).

This is one of those real-world ML lessons: training and running models isn’t just theory—it’s also library/version wrinkles.

## Conclusion

By adapting nanochat to constrained hardware, it becomes an excellent tool for understanding how LLM training really works—end-to-end.

You get to see (and feel) the practical parts that are usually hidden behind big infrastructure:

- How text turns into tokens, and why tokenization matters
- What “predict the next word” training actually looks like in practice
- how the stages build on each other (pretrain → midtrain → SFT)
- The fundamental trade-offs between model size, sequence length, batch size, and quality
- the unglamorous engineering work: memory limits, checkpoints, training runs that go overnight, and the little fixes needed to make things run reliably

If you want to demystify LLMs, doing a small version of the full pipeline yourself is hard to beat.

## Resources

- nanochat repository: github.com/karpathy/nanochat
- My optimized scripts: https://github.com/vlele/nanochat/tree/master
- Training logs: Documented checkpoint locations and configurations
- Chat interface: python -m scripts.chat_cli to interact with the trained model

## Technical Details

### Final Model Specifications:

- Parameters: 92,274,688 (~92M)
- Layers: 8
- Model dimension: 512
- Attention heads: 4
- Vocabulary: 65,536 tokens
- Max sequence length: 1,024 tokens

### Training Configuration:

- Base: 3,520 steps, ~11-12 hours
- Mid: 500 steps, ~6-7 hours
- SFT: 200 steps, ~20-30 minutes
- Total: 4,220 steps, ~18-20 hours

### Hardware:

- GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
- CUDA: 12.8
- Environment: WSL2 on Windows
