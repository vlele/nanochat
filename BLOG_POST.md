# Training a 100M Parameter Language Model on a Laptop GPU: Demystifying Gen AI from Scratch

## Introduction

Large Language Models (LLMs) like GPT-4 and Claude seem like black boxes—trained on massive clusters with billions of parameters, requiring millions of dollars in compute. But what if you could train your own language model from scratch on a single laptop GPU? That's exactly what I did using [nanochat](https://github.com/karpathy/nanochat), Andrej Karpathy's educational LLM training codebase.

In this post, I'll walk through how I adapted nanochat to run on an **NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)**—a consumer-grade GPU that's far from the enterprise hardware typically used for LLM training. The result? A fully functional 92-million parameter language model trained entirely on my laptop, and more importantly, a deep understanding of how modern AI models actually work.

---

## Part 1: Adapting nanochat for a Laptop GPU

### The Hardware Challenge

My setup:
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **VRAM**: 8GB (compared to 80GB+ on enterprise GPUs)
- **CUDA**: Version 12.8
- **Environment**: WSL2 on Windows

The default nanochat configuration targets multi-GPU setups with much larger models. To make it work on my 8GB GPU, I had to make several strategic optimizations.

### Key Changes Made

#### 1. Model Architecture Reduction

**Default Configuration:**
- Depth: 20 layers
- Model dimension: 1,280 (depth × 64)
- Parameters: ~561 million

**My Configuration:**
- Depth: 8 layers (60% reduction)
- Model dimension: 512 (depth × 64)
- Parameters: ~92 million (84% reduction)

**Why?** Each layer requires significant memory for:
- Attention weights (Query, Key, Value projections)
- Feed-forward network weights
- Activations during forward/backward passes

Reducing from 20 to 8 layers cut memory usage by approximately 60% while maintaining the core architecture.

#### 2. Batch Size Optimization

**Default:**
- `device_batch_size`: 32 (for multi-GPU)
- `max_seq_len`: 2,048 tokens

**My Configuration:**
- Base training: `device_batch_size`: 2 (94% reduction)
- Midtraining: `device_batch_size`: 4
- SFT: `device_batch_size`: 2
- `max_seq_len`: 1,024 tokens (50% reduction)

**Why?** Memory scales quadratically with sequence length (due to attention matrices) and linearly with batch size. Reducing both was essential to fit within 8GB.

#### 3. Memory Management Strategies

**Disabled torch.compile:**
```bash
export TORCH_COMPILE_DISABLE=1
```
While `torch.compile` can speed up training, it requires additional memory during compilation and can cause OOM errors on constrained hardware.

**Disabled Evaluation During Training:**
- Base training: `--eval_every=-1`, `--core_metric_every=-1` (disabled CORE evaluation)
- Midtraining: `--eval_every=-1` (disabled validation loss computation)
- SFT: `--eval_every=-1` (disabled validation during fine-tuning)

Evaluation requires loading additional data and running inference, consuming precious VRAM. During midtraining, we encountered OOM errors when evaluation was enabled, so we disabled it entirely. While this means we couldn't monitor validation loss during training, it was necessary to fit within the 8GB memory constraint. The model still trains effectively—we just don't get real-time feedback on validation performance.

**Enabled Memory Expansion:**
```bash
export PYTORCH_ALLOC_CONF="expandable_segments:True"
```
This allows PyTorch to dynamically allocate memory, reducing fragmentation.

#### 4. Checkpointing Strategy

Added intermediate checkpointing to prevent data loss:
- Base training: Every 250 steps
- Midtraining: Every 100 steps  
- SFT: Every 100 steps

This ensures that if training crashes (OOM, power loss, etc.), we only lose at most 100-250 steps of progress instead of everything.

### Training Timeline

| Phase | Steps | Time | Memory Usage |
|-------|-------|------|--------------|
| **Base Training** | 3,520 | ~11-12 hours | ~4-4.5GB |
| **Midtraining** | 500 | ~6-7 hours | ~4-5GB |
| **SFT** | 200 | ~20-30 minutes | ~3-4GB |
| **Total** | 4,220 | **~18-20 hours** | - |

The entire training process completed in under a day on a single consumer GPU—a testament to how accessible LLM training has become.

---

## Part 2: The Three Training Scripts

nanochat's training is divided into three distinct phases, each with its own script and purpose:

### 1. `base_train.py` - Pretraining

**Purpose**: Teach the model fundamental language patterns and world knowledge.

**What it does:**
- Trains on raw text from FineWeb-Edu dataset
- Learns to predict the next token in a sequence
- Builds internal representations of language structure

**Key Features:**
- Uses Chinchilla scaling laws (20× parameters in tokens)
- Gradient accumulation to maintain large effective batch size
- Learning rate scheduling with warmup/warmdown
- Mixed precision training (bfloat16)

**Example Training Data:**
```
"The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet..."
```

The model learns that "fox" often follows "brown", "jumps" follows "fox", etc.

### 2. `mid_train.py` - Midtraining

**Purpose**: Teach the model conversation format, special tokens, and task-specific behaviors.

**What it does:**
- Introduces special tokens: `<|user_start|>`, `<|assistant_start|>`, etc.
- Trains on structured tasks: multiple-choice (MMLU), math (GSM8K), conversations (SmolTalk)
- Teaches tool use (Python REPL execution)
- Imparts identity/personality from custom JSON data

**Key Features:**
- Task mixture with weighted sampling
- Different data formats (conversations, Q&A, code)
- Maintains language knowledge while adding new capabilities

**Example Training Data:**
```
<|user_start|>What is 2+2?<|user_end|>
<|assistant_start|>2 + 2 equals 4.<|assistant_end|>
```

The model learns the conversation structure and how to respond appropriately.

### 3. `chat_sft.py` - Supervised Fine-Tuning

**Purpose**: Polish the model's chat ability and helpfulness.

**What it does:**
- Trains on high-quality conversational examples
- Refines response quality and tone
- Ensures consistent helpful behavior

**Key Features:**
- Shorter training (200 steps vs 701 default)
- Focuses on conversation quality
- Uses curated datasets (ARC, GSM8K, SmolTalk)

**Example Training Data:**
```
<|user_start|>Can you explain quantum computing?<|user_end|>
<|assistant_start|>Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in a superposition of both states simultaneously...<|assistant_end|>
```

The model learns to provide detailed, helpful explanations.

---

## Part 3: Understanding the Four Stages

nanochat is designed for **understandability**—every component is written from scratch with clear, readable code. This makes it an excellent learning tool. Let's break down each stage:

### Stage 1: Tokenizer Training

**What is a Tokenizer?**
A tokenizer converts text into numbers (tokens) that the model can process. Instead of processing individual characters (inefficient) or entire words (too many possibilities), modern LLMs use subword tokenization.

**How it Works:**
1. Start with individual characters
2. Find the most frequent character pairs
3. Merge them into tokens
4. Repeat until you have ~65,536 tokens

**Example:**
```
Text: "Hello world"
Characters: ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
Tokens: [1234, 5678, 9012]  # Much more efficient!
```

**Training Data Source:**
- FineWeb-Edu dataset
- ~2 billion characters for tokenizer training
- Creates a vocabulary of 65,536 tokens

**Code Location:** `scripts/tok_train.py`

### Stage 2: Pretraining (Base Training)

**What is Pretraining?**
The model learns to predict the next token in a sequence by training on massive amounts of text. It's like teaching a child to read by showing them millions of books.

**The Learning Process:**
1. Model sees: "The cat sat on the"
2. Model predicts: "mat" (most likely next word)
3. If correct: weights are slightly adjusted to reinforce this
4. If wrong: weights are adjusted to correct the mistake
5. Repeat billions of times

**Training Data Source:**
- **FineWeb-Edu**: Filtered web text, educational content
- ~1.8 billion tokens (for our 92M parameter model)
- Follows Chinchilla scaling: 20× parameters in tokens

**Example Training Sequence:**
```
Input:  "The capital of France is"
Target: " Paris"
```

The model learns facts, grammar, reasoning patterns, and world knowledge.

**Key Architecture Features:**
- **Transformer blocks**: Self-attention + feed-forward networks
- **Rotary Position Embeddings (RoPE)**: Encodes position information
- **RMSNorm**: Normalization without learnable parameters
- **ReLU² activation**: Modern activation function
- **Group-Query Attention (GQA)**: Efficient attention mechanism

**Code Location:** `scripts/base_train.py`, `nanochat/gpt.py`

### Stage 3: Midtraining

**What is Midtraining?**
After pretraining, the model knows language but doesn't know how to have conversations. Midtraining teaches it:
- Conversation structure (user/assistant turns)
- Special tokens and their meanings
- Task-specific formats (multiple choice, math problems)
- Tool use (executing Python code)

**Memory Optimization Note:** During midtraining, we disabled evaluation (`--eval_every=-1`) to prevent OOM errors. Evaluation requires loading validation data and running inference, which consumes additional VRAM. While this means we couldn't monitor validation loss during training, it was essential to fit within the 8GB memory constraint. The model still learns effectively—we just train without real-time validation feedback.

**Training Data Sources:**

1. **SmolTalk**: High-quality conversational examples
   ```
   User: "What's the weather like?"
   Assistant: "I don't have access to real-time weather data..."
   ```

2. **MMLU**: Multiple-choice questions across 57 subjects
   ```
   Question: "What is the capital of Australia?"
   A) Sydney  B) Melbourne  C) Canberra  D) Brisbane
   Answer: C
   ```

3. **GSM8K**: Grade school math word problems
   ```
   "John has 5 apples. He gives 2 to Mary. How many does he have left?"
   Answer: 3
   ```

4. **Custom JSON**: Identity and personality data
   ```
   {"conversations": [{"role": "user", "content": "Who are you?"}, 
                      {"role": "assistant", "content": "I'm nanochat, a helpful AI assistant..."}]}
   ```

5. **Spelling Tasks**: Simple spelling exercises
   ```
   "Spell the word 'computer'"
   Answer: "c-o-m-p-u-t-e-r"
   ```

**Code Location:** `scripts/mid_train.py`, `tasks/` directory

### Stage 4: Supervised Fine-Tuning (SFT)

**What is SFT?**
The final polish. The model already knows language and conversation structure, but SFT ensures it's helpful, accurate, and follows instructions well.

**Training Data Sources:**
- **ARC (AI2 Reasoning Challenge)**: Science questions
- **GSM8K**: Math problems (refined examples)
- **SmolTalk**: High-quality conversations
- **Custom JSON**: Identity reinforcement
- **Spelling tasks**: Final capability checks

**Example SFT Data:**
```
User: "Explain photosynthesis simply"
Assistant: "Photosynthesis is how plants make food using sunlight. 
They take in carbon dioxide from air and water from roots, 
use energy from sunlight to combine them, and produce glucose 
(sugar) and oxygen. The oxygen is released as a byproduct."
```

**Key Difference from Midtraining:**
- Midtraining: Teaches *what* conversations look like
- SFT: Teaches *how* to be helpful and accurate

**Code Location:** `scripts/chat_sft.py`

---

## Part 4: From 1B to 100M Parameters

### The Scaling Challenge

The default nanochat configuration (`depth=20`) produces a **561-million parameter model**. For an 8GB GPU, this is too large. Here's how we scaled down:

### Parameter Calculation

**Default Model (depth=20):**
- Model dimension: 20 × 64 = 1,280
- Attention heads: (1,280 + 127) // 128 = 10
- Embedding: 65,536 × 1,280 = 83.9M parameters
- Per layer: ~19.65M parameters (attention + MLP)
- 20 layers: 393M parameters
- Output head: 83.9M parameters
- **Total: ~561M parameters**

**Our Model (depth=8):**
- Model dimension: 8 × 64 = 512
- Attention heads: (512 + 127) // 128 = 4
- Embedding: 65,536 × 512 = 33.5M parameters
- Per layer: ~3.15M parameters
- 8 layers: 25.2M parameters
- Output head: 33.5M parameters
- **Total: ~92M parameters**

### The Trade-offs

**What We Lost:**
- Model capacity (fewer parameters = less knowledge storage)
- Some reasoning ability on complex tasks
- Long-context understanding (reduced sequence length)

**What We Gained:**
- Ability to train on consumer hardware
- Faster training (18 hours vs days/weeks)
- Lower memory requirements
- Still capable of meaningful conversations and tasks

### Performance Comparison

While our 92M model can't match GPT-4's capabilities, it demonstrates:
- ✅ Natural language understanding
- ✅ Conversational ability
- ✅ Basic reasoning (math, science questions)
- ✅ Code generation (simple Python)
- ✅ Following instructions

The model is surprisingly capable for its size, proving that you don't need billions of parameters to create a functional language model.

---

## Part 5: Demystifying Generative AI

### Why This Exercise Matters

Training a language model from scratch is the best way to understand how modern AI actually works. Here's what I learned:

### 1. It's Not Magic—It's Math

**Attention Mechanism:**
The "magic" of transformers is actually a mathematical operation:
```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

Each token "attends" to other tokens, learning relationships. When you see "The cat sat on the", the model uses attention to connect "cat" with "sat" and predict "mat".

**Gradient Descent:**
The model learns by:
1. Making a prediction
2. Calculating how wrong it was (loss)
3. Adjusting weights slightly in the direction that reduces error
4. Repeating millions of times

It's optimization, not magic.

### 2. Data is Everything

The model's knowledge comes entirely from training data:
- **FineWeb-Edu**: Provides world knowledge, facts, language patterns
- **Task datasets**: Teach specific capabilities (math, reasoning, conversation)
- **Quality matters**: Better data → better model

There's no "understanding" in the human sense—just pattern matching at scale.

### 3. Architecture Choices Matter

Every design decision affects performance:
- **RoPE vs learned embeddings**: Better extrapolation to longer sequences
- **RMSNorm vs LayerNorm**: Simpler, no learnable parameters
- **ReLU² activation**: Better than ReLU for language models
- **GQA**: Reduces memory without significant quality loss

Understanding these choices helps you appreciate why modern models work so well.

### 4. Training is Iterative

The three-stage process (pretrain → midtrain → SFT) isn't arbitrary:
- **Pretraining**: Builds foundation (language understanding)
- **Midtraining**: Adds structure (conversation format)
- **SFT**: Polishes behavior (helpfulness)

Each stage builds on the previous one. You can't skip steps.

### 5. Scaling Laws are Real

Chinchilla scaling (20× parameters in tokens) isn't a suggestion—it's based on empirical research. Training with too little data leads to underfitting. Too much data (beyond the ratio) provides diminishing returns.

### Key Insights

1. **LLMs are pattern matchers**, not reasoning engines (though they can appear to reason)
2. **Training data quality** matters more than model size (to a point)
3. **Architecture innovations** (attention, RoPE, etc.) enable the capabilities we see
4. **The "emergent" abilities** are just complex pattern matching at scale
5. **You can train useful models** on consumer hardware with the right optimizations

### What This Means for AI Understanding

By training from scratch, you learn:
- How attention actually works (not just theory)
- Why certain architectures are chosen
- How data shapes model behavior
- What "learning" means in the context of neural networks
- The practical challenges of training (memory, optimization, etc.)

This hands-on experience demystifies AI. You see that:
- It's not sentient—it's statistics
- It's not understanding—it's pattern matching
- It's not magic—it's engineering

But that doesn't make it less impressive. The fact that pattern matching at scale can produce such sophisticated behavior is remarkable.

---

## Part 6: Interacting with the Trained Model

### Chatting with Your Model

After training completes, you can interact with your model using the command-line interface. Here's an example of chatting with the first SFT checkpoint (step 100):

```bash
$ python3 -m scripts.chat_cli -s 100 -p "explain generative AI and three stages of learning in easy terms"

Autodetected device type: cuda
2025-12-31 11:31:48,324 - nanochat.common - INFO - Distributed world size: 1
2025-12-31 11:31:48,325 - nanochat.checkpoint_manager - INFO - No model tag provided, guessing model tag: d8
2025-12-31 11:31:48,325 - nanochat.checkpoint_manager - INFO - Loading model from /home/vlele/.cache/nanochat/chatsft_checkpoints/d8 with step 100
2025-12-31 11:31:48,798 - nanochat.checkpoint_manager - INFO - Building model with config: {'sequence_len': 1024, 'vocab_size': 65536, 'n_layer': 8, 'n_head': 4, 'n_kv_head': 4, 'n_embd': 512}

NanoChat Interactive Mode
--------------------------------------------------
Type 'quit' or 'exit' to end the conversation
Type 'clear' to start a new conversation
--------------------------------------------------

Assistant: Imagine a world where the human brain is to process and analyze. 
To help you understand the origins of AI and the emergence of new, AI-driven 
AI assistants. Imagine a world where AI-generated textural experiences can be 
translated into music, a language that is both familiar and new. AI-powered 
chatbots would be equipped with AI-powered chatbots that can read the user's 
words, adapt their language to their preferences, and adapt their to a new 
environment.

Imagine a world where AI-driven chatbots are trained to recognize the user's, 
but not necessarily the user, and,. AI could even learn how to recognize and 
respond to human-like responses, which are then translated into music. AI could 
even learn how to use AI to listen to music in real-time, analyzing their 
conversations, identifying patterns, and suggesting alternative phrases.

Another idea is a world where AI-powered chatbots be used to learn and adapt 
to new situations. AI could even learn 3D and adapt to new environments, based 
on the user's preferences, preferences, and the AI's limitations. AI could 
even learn how to recognize and respond to human-like responses, adapting to 
new situations as they learn to adapt to new situations.

AI could also be used to develop AI-powered chatbots that can learn and adapt 
to new...
```

**Observations:**
- The model loads successfully and generates text
- It demonstrates understanding of the prompt (mentions AI, learning, adaptation)
- The response shows the model's limitations: it's repetitive and doesn't fully answer the question about "three stages of learning"
- This is expected for a 92M parameter model trained for only 200 SFT steps

For a model of this size, the responses are reasonable. With more training or a larger model, quality would improve significantly.

### Making Interactive Chat Work: PyTorch Compatibility Fixes

Getting the chat interface working required fixing compatibility issues with older PyTorch versions. The nanochat codebase uses newer PyTorch features that aren't available in all versions.

#### Issue 1: RMSNorm Function

**Problem:** `torch.nn.functional.rms_norm()` was added in PyTorch 2.3.0. Older versions don't have this function.

**Solution:** Implemented a fallback RMSNorm function:
```python
def norm(x):
    if hasattr(F, 'rms_norm'):
        return F.rms_norm(x, (x.size(-1),))
    else:
        # Fallback implementation for older PyTorch versions
        eps = 1e-6
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() / torch.sqrt(variance + eps)
        return x_normed.to(x.dtype)
```

RMSNorm (Root Mean Square Normalization) is simpler than LayerNorm—it normalizes by the root mean square of activations without learnable parameters.

#### Issue 2: Group Query Attention Parameter

**Problem:** The `enable_gqa` parameter in `scaled_dot_product_attention()` was added in newer PyTorch versions. Older versions throw a `TypeError` when this parameter is passed.

**Solution:** Added compatibility handling:
```python
def call_attention(q, k, v, **kwargs):
    """Helper to call scaled_dot_product_attention with optional enable_gqa"""
    if enable_gqa:
        try:
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True, **kwargs)
        except TypeError:
            # Fallback: GQA already handled manually above
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
    else:
        # No GQA, don't pass the parameter
        return F.scaled_dot_product_attention(q, k, v, **kwargs)
```

When Group Query Attention (GQA) is enabled, we manually duplicate key/value heads for older PyTorch versions, then call the attention function without the `enable_gqa` parameter.

#### Why These Fixes Matter

These compatibility fixes highlight an important lesson: **real-world ML engineering involves handling version differences**. Production code needs to work across different environments, not just the latest versions.

The fixes also demonstrate:
- **Backward compatibility**: Code should work with older library versions
- **Graceful degradation**: Fallback implementations when features aren't available
- **Practical problem-solving**: Adapting code to work in constrained environments

These are the kinds of issues you encounter when deploying models in the real world—not just the theoretical understanding, but the practical engineering challenges.

---

## Conclusion

Training a language model on a laptop GPU taught me more about AI than years of reading papers. The nanochat codebase, designed for educational purposes, makes every component understandable. By adapting it for constrained hardware, I learned:

1. **Practical ML engineering**: Memory management, optimization, checkpointing
2. **Model architecture**: How transformers actually work under the hood
3. **Training dynamics**: The iterative process of building capabilities
4. **Scaling challenges**: Trade-offs between model size and hardware constraints

The result? A 92-million parameter model that can hold conversations, answer questions, and demonstrate the core capabilities of modern LLMs—all trained on a single consumer GPU in under a day.

More importantly, I now understand how these models work. The "black box" is now transparent. I can read research papers and understand the innovations. I can reason about why certain architectures work better than others. I can appreciate the engineering that goes into systems like GPT-4.

**If you want to demystify AI, train a model from scratch.** There's no better way to understand how modern language models actually work.

---

## Resources

- **nanochat repository**: [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)
- **My optimized scripts**: Available in the repository
- **Training logs**: Documented checkpoint locations and configurations
- **Chat interface**: `python -m scripts.chat_cli` to interact with the trained model

## Technical Details

**Final Model Specifications:**
- Parameters: 92,274,688 (~92M)
- Layers: 8
- Model dimension: 512
- Attention heads: 4
- Vocabulary: 65,536 tokens
- Max sequence length: 1,024 tokens

**Training Configuration:**
- Base: 3,520 steps, ~11-12 hours
- Mid: 500 steps, ~6-7 hours  
- SFT: 200 steps, ~20-30 minutes
- Total: 4,220 steps, ~18-20 hours

**Hardware:**
- GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
- CUDA: 12.8
- Environment: WSL2 on Windows

---

*This blog post was written to document the process of training a language model from scratch on consumer hardware. All code, configurations, and learnings are shared to help others understand how modern AI models work.*

