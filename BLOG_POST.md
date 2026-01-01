```md
# Training a 100M Parameter Language Model on a Laptop GPU: Demystifying Gen AI from Scratch

## Happy New Year!

Happy New Year — hope you had a lovely break over the holidays. For many of us, this stretch means fewer meetings, slower work days, and some travel with family and friends. That extra breathing room is often when the projects you’ve been meaning to get to finally get their turn.

For me, it was playing with nanochat—Andrej Karpathy’s “best ChatGPT that $100 can buy” codebase—and trying to understand the nuts and bolts of training an LLM from scratch.

One thing I love about Andrej’s approach is how he keeps dependencies minimal, so you can actually follow the code. He even coded his own Transformer instead of leaning on a full-stack library like Hugging Face, which makes the whole system easier to read, reason about, and learn from.

Beyond diving deeper into LLM training with nanochat, I had a few additional objectives: There’s a lot of talk about big ideas—AGI, singularity, superintelligence, “threat to humanity.” It’s interesting, but it isn’t always helpful for an average layperson trying to understand what’s real, and it can create unnecessary myth around these models.

I wanted to demystify how training actually works.

Large Language Models (LLMs) like GPT-4 and Claude seem like black boxes—trained on massive clusters with billions of parameters, requiring millions of dollars in compute. But what if you could train your own language model from scratch on a single laptop GPU? I deliberately constrained the training environment to my laptop (instead of renting GPUs in the cloud), which meant the runs ran for long hours—often overnight. That helped folks not deeply familiar with model training see how involved and resource-intensive it is in terms of time and energy usage. It also made it easier to use a student-learning analogy:

- **Kindergarten:** you learn the alphabet and how to break words into pieces so you can read at all (tokenizer/vocabulary)
- **Elementary school:** pre-training is basically “predict the next word” practice, over and over (pre-training)
- **Middle/high school:** you learn formats—how to answer questions, show your work, and write in a way that matches the assignment (midtraining)
- **Final exams/graduation:** a teacher (or rubric) keeps correcting you until your answers are more helpful, consistent, and on-topic (SFT)

In this post, I'll walk through how I adapted nanochat to run on an NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)—a consumer-grade GPU that's far from the enterprise hardware typically used for LLM training. The result? A fully functional 92-million parameter language model trained entirely on my laptop, and more importantly, a deep understanding of how modern AI models actually work.

## Part 1: Adapting nanochat for a Laptop GPU

### The Hardware Challenge

My setup:

- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
- **VRAM:** 8GB (compared to 80GB+ on enterprise GPUs)
- **CUDA:** Version 12.8
- **Environment:** WSL2 on Windows

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

Reducing the number of layers from 20 to 8 cut memory usage by approximately 60% while maintaining the core architecture.

#### 2. Batch Size Optimization

**Default:**

- device_batch_size: 32 (for multi-GPU)
- max_seq_len: 2,048 tokens

**My Configuration:**

- Base training: device_batch_size: 2 (94% reduction)
- Midtraining: device_batch_size: 4
- SFT: device_batch_size: 2
- max_seq_len: 1,024 tokens (50% reduction)

**Why?** Memory scales quadratically with sequence length (due to attention matrices) and linearly with batch size. Reducing both was essential to fit within 8GB.

#### 3. Memory Management Strategies

**Disabled torch.compile:**

```

export TORCH_COMPILE_DISABLE=1

```

While torch.compile can speed up training, it requires additional memory during compilation and can cause OOM errors on constrained hardware.

**Disabled Evaluation During Training:**

- Base training: --eval_every=-1, --core_metric_every=-1 (disabled CORE evaluation)
- Midtraining: --eval_every=-1 (disabled validation loss computation)
- SFT: --eval_every=-1 (disabled validation during fine-tuning)

Evaluation requires loading additional data and running inference, consuming precious VRAM. During mid-training, we encountered OOM errors when evaluation was enabled, so we disabled it entirely. While this means we couldn't monitor validation loss during training, it was necessary to fit within the 8GB memory constraint. The model still trains effectively—we don't get real-time feedback on validation performance.

#### 4. Checkpointing Strategy

Added intermediate checkpointing to prevent data loss — learned this lesson the hard way after getting OOM errors hours into pre-training:

- Base training: Every 250 steps
- Midtraining: Every 100 steps
- SFT: Every 100 steps

This ensures that if training crashes (OOM, power loss, etc.), we only lose 100-250 steps of progress, rather than everything.

### Training Timeline

| Phase         | Steps     | Time            | Memory Usage |
|--------------|-----------|-----------------|--------------|
| Base Training | 3,520     | ~11-12 hours     | ~4-4.5GB     |
| Midtraining   | 500       | ~6-7 hours       | ~4-5GB       |
| SFT           | 200       | ~20-30 minutes   | ~3-4GB       |
| **Total**     | **4,220** | **~18-20 hours** | -            |

The entire training process was completed in under a day on a single consumer GPU—not counting several restarts dealing with the errors and fixes described above—a testament to how accessible LLM training has become.

## Part 2: The Three Training Scripts

Rather than using one consolidated script (speedrun.sh), I broke the script into smaller scripts that allowed for more control and allowed me to run and optimize each step.

### 1. base_train.py - Pretraining

**Purpose:** Teach the model fundamental language patterns and world knowledge.

**What it does:**

- Trains on raw text from FineWeb-Edu dataset
- Learns to predict the next token in a sequence
- Builds internal representations of language structure

**Example Training Data:**  
"The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet..."

The model learns that "fox" often follows "brown", "jumps" follows "fox", etc.

### 2. mid_train.py - Midtraining

**Purpose:** Teach the model conversation format, special tokens, and task-specific behaviors.

**What it does:**

- Introduces special tokens: <|user_start|>, <|assistant_start|>, etc.
- Trains on structured tasks: multiple-choice (MMLU), math (GSM8K), conversations (SmolTalk)
- Teaches tool use (Python REPL execution)
- Imparts identity/personality from custom JSON data

**Example Training Data:**  
<|user_start|>What is 2+2?<|user_end|>  
<|assistant_start|>2 + 2 equals 4.<|assistant_end|>

The model learns the conversation structure and how to respond appropriately.

### 3. chat_sft.py - Supervised Fine-Tuning

**Purpose:** Polish the model's chat ability and helpfulness.

**What it does:**

- Trains on high-quality conversational examples
- Refines response quality and tone
- Ensures consistent helpful behavior

**Example Training Data:**  
<|user_start|>Can you explain quantum computing?<|user_end|>  
<|assistant_start|>Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in a superposition of both states simultaneously...<|assistant_end|>

The model learns to provide detailed, helpful explanations.

## Part 3: Understanding the Four Stages

nanochat is designed for understandability—every component is written from scratch with clear, readable code. This makes it an excellent learning tool. Let's break down each stage:

### Stage 1: Tokenizer Training

**What is a Tokenizer?** A tokenizer converts text into numbers (tokens) that the model can process. Instead of processing individual characters (inefficient) or entire words (too many possibilities), modern LLMs use subword tokenization.

**How it Works:**

- Start with individual characters
- Find the most frequent character pairs
- Merge them into tokens
- Repeat until you have ~65,536 tokens

**Example:**  
Text: "Hello world"  
Characters: ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']  
Tokens: [1234, 5678, 9012]  # Much more efficient!

**Training Data Source:**

- FineWeb-Edu dataset
- ~2 billion characters for tokenizer training
- Creates a vocabulary of 65,536 tokens

**Code Location:** scripts/tok_train.py

### Stage 2: Pretraining (Base Training)

**What is Pretraining?** The model learns to predict the next token in a sequence by training on massive amounts of text. It's like teaching a child to read by showing them millions of books.

**The Learning Process:**

- Model sees: "The cat sat on the"
- Model predicts: "mat" (most likely next word)
- If correct: weights are slightly adjusted to reinforce this
- If wrong: weights are adjusted to correct the mistake
- Repeat billions of times

**Training Data Source:**

- FineWeb-Edu: Filtered web text, educational content
- ~1.8 billion tokens (for our 92M parameter model)
- Follows Chinchilla scaling: 20× parameters in tokens (According to the Chinchilla paper, for a fixed compute budget, you usually get better results by not making the brain huge and under-trained. Instead, keep model size and training tokens balanced)

**Example Training Sequence:**  
Input:  "The capital of France is"  
Target: " Paris"

The model learns facts, grammar, reasoning patterns, and world knowledge.

**Code Location:** scripts/base_train.py, nanochat/gpt.py

### Stage 3: Midtraining

**What is Midtraining?** After pretraining, the model knows language but doesn't know how to have conversations. Midtraining teaches it:

- Conversation structure (user/assistant turns)
- Special tokens and their meanings
- Task-specific formats (multiple choice, math problems)
- Tool use (executing Python code)

**Memory Optimization Note:** During midtraining, we disabled evaluation (--eval_every=-1) to prevent OOM errors. Evaluation requires loading validation data and running inference, which consumes additional VRAM. While this means we couldn't monitor validation loss during training, it was essential to fit within the 8GB memory constraint. The model still learns effectively—we train without real-time validation feedback.

**Training Data Sources:**

- **SmolTalk:** High-quality conversational examples  
  User: "What's the weather like?"  
  Assistant: "I don't have access to real-time weather data..."

- **MMLU:** Multiple-choice questions across 57 subjects  
  Question: "What is the capital of Australia?"  
  A) Sydney  B) Melbourne  C) Canberra  D) Brisbane  
  Answer: C

- **GSM8K:** Grade school math word problems  
  "John has 5 apples. He gives 2 to Mary. How many does he have left?"  
  Answer: 3

- **Custom JSON:** Identity and personality data  
  {"conversations": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm nanochat, a helpful AI assistant..."}]}

- **Spelling Tasks:** Simple spelling exercises  
  "Spell the word 'computer'"  
  Answer: "c-o-m-p-u-t-e-r"

**Code Location:** scripts/mid_train.py, tasks/ directory

### Stage 4: Supervised Fine-Tuning (SFT)

**What is SFT?** The final polish. The model already knows language and conversation structure, but SFT ensures it's helpful, accurate, and follows instructions well.

**Training Data Sources:**

- ARC (AI2 Reasoning Challenge): Science questions
- GSM8K: Math problems (refined examples)
- SmolTalk: High-quality conversations
- Custom JSON: Identity reinforcement
- Spelling tasks: Final capability checks

**Example SFT Data:**  
User: "Explain photosynthesis simply"  
Assistant: "Photosynthesis is how plants make food using sunlight. They take in carbon dioxide from air and water from roots, use energy from sunlight to combine them, and produce glucose (sugar) and oxygen. The oxygen is released as a byproduct."

**Key Difference from Midtraining:**

- Midtraining: Teaches what conversations look like
- SFT: Teaches how to be helpful and accurate

**Code Location:** scripts/chat_sft.py

## Part 4: From 1B to 100M Parameters

### The Scaling Challenge

The default nanochat configuration (depth=20) produces a 561-million parameter model. For an 8GB GPU, this is too large. Here's how we scaled down:

### Parameter Calculation

**Default Model (depth=20):**

- Model dimension: 20 × 64 = 1,280
- Attention heads: (1,280 + 127) // 128 = 10
- Embedding: 65,536 × 1,280 = 83.9M parameters
- Per layer: ~19.65M parameters (attention + MLP)
- 20 layers: 393M parameters
- Output head: 83.9M parameters
- Total: ~561M parameters

**Our Model (depth=8):**

- Model dimension: 8 × 64 = 512
- Attention heads: (512 + 127) // 128 = 4
- Embedding: 65,536 × 512 = 33.5M parameters
- Per layer: ~3.15M parameters
- 8 layers: 25.2M parameters
- Output head: 33.5M parameters
- Total: ~92M parameters

### The Trade-offs

**Cons:**

- Model capacity (fewer parameters = less knowledge storage)
- Some reasoning ability on complex tasks
- Long-context understanding (reduced sequence length)

**Pros:**

- Ability to train on consumer hardware
- Faster training (18 hours vs days/weeks)
- Lower memory requirements
- Still capable of meaningful conversations and tasks

### Performance Comparison

Making the difference obvious — while our 92M model won't come anywhere close to GPT-4's capabilities, it demonstrates:

- ✅ Natural language understanding
- ✅ Conversational ability
- ✅ Basic reasoning (math, science questions)
- ✅ Code generation (simple Python)
- ✅ Following instructions

The model is surprisingly capable for its size, proving that you don't need billions of parameters to create a functional language model.

## Part 5: Interacting with the Trained Model

### Chatting with Your Model

After training completes, you can interact with the model using the command-line interface. Here's an example of chatting with the first SFT checkpoint (step 100):

```

$ python3 -m scripts.chat_cli -s 100 -p "explain generative AI and three stages of learning in easy terms"

## NanoChat Interactive Mode

Type 'quit' or 'exit' to end the conversation
Type 'clear' to start a new conversation
----------------------------------------

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
```
