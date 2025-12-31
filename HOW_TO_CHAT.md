# How to Chat with Your Trained Model

After SFT training completes, you can interact with your model in two ways:

## 🖥️ Method 1: Command Line Interface (CLI)

### Quick Test (Single Prompt)
```bash
python -m scripts.chat_cli -p "Hello! What can you help me with today?"
```

### Interactive Chat Mode
```bash
python -m scripts.chat_cli
```

This will start an interactive session where you can:
- Type messages and get responses
- Type `clear` to start a new conversation
- Type `quit` or `exit` to end the session

### CLI Options

```bash
python -m scripts.chat_cli [OPTIONS]
```

**Options:**
- `-i, --source`: Model source (`sft`|`mid`|`rl`) - **default: `sft`** ✅
- `-g, --model-tag`: Model tag (e.g., `d8`) - auto-detected if not specified
- `-s, --step`: Specific step to load - uses latest if not specified
- `-p, --prompt`: Single prompt mode (non-interactive)
- `-t, --temperature`: Temperature (0.0-2.0) - **default: 0.6**
- `-k, --top-k`: Top-k sampling - **default: 50**
- `--device-type`: Force device (`cuda`|`cpu`|`mps`) - auto-detects if not specified
- `-d, --dtype`: Data type (`bfloat16`|`float32`) - **default: bfloat16**

### Examples

**Basic chat (uses SFT checkpoint automatically):**
```bash
python -m scripts.chat_cli
```

**Chat with specific temperature:**
```bash
python -m scripts.chat_cli -t 0.8
```

**Chat with midtraining checkpoint instead:**
```bash
python -m scripts.chat_cli -i mid
```

**Single prompt (non-interactive):**
```bash
python -m scripts.chat_cli -p "Explain quantum computing in simple terms"
```

**More creative responses (higher temperature):**
```bash
python -m scripts.chat_cli -t 1.0 -k 100
```

**More focused responses (lower temperature):**
```bash
python -m scripts.chat_cli -t 0.3 -k 20
```

---

## 🌐 Method 2: Web UI (ChatGPT-style Interface)

### Start the Web Server
```bash
python -m scripts.chat_web
```

This will:
1. Load your SFT model
2. Start a FastAPI server
3. Print a URL (usually `http://localhost:8000`)
4. Open that URL in your browser

### Web UI Features
- **ChatGPT-style interface** with message history
- **Streaming responses** (tokens appear as they're generated)
- **Adjustable settings**: temperature, top-k, max tokens
- **Multi-GPU support** (if you have multiple GPUs)

### Web UI Options

```bash
python -m scripts.chat_web [OPTIONS]
```

**Options:**
- `--num-gpus`: Number of GPUs to use (default: 1)
- `--port`: Port to run server on (default: 8000)
- `--host`: Host to bind to (default: 0.0.0.0)
- `-i, --source`: Model source (`sft`|`mid`|`rl`) - **default: `sft`** ✅
- `-g, --model-tag`: Model tag (e.g., `d8`)
- `-s, --step`: Specific step to load

### Accessing the Web UI

1. **Local machine**: Open `http://localhost:8000` in your browser
2. **WSL2**: Use `http://localhost:8000` (WSL2 forwards ports automatically)
3. **Remote server**: Use the server's IP address: `http://YOUR_IP:8000`

### Web UI API Endpoint

The web server also exposes a Chat API compatible with OpenAI's format:

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

## 📍 Checkpoint Location

Your SFT checkpoints are saved at:
```
~/.cache/nanochat/chatsft_checkpoints/d8/
```

The model automatically loads the **latest checkpoint** (highest step number) unless you specify otherwise.

### Check Available Checkpoints
```bash
ls -lh ~/.cache/nanochat/chatsft_checkpoints/d8/
```

You'll see files like:
- `model_000200.pt` - Model weights at step 200
- `meta_000200.json` - Metadata (step, loss, config)

### Load Specific Checkpoint
```bash
# Load checkpoint from step 200
python -m scripts.chat_cli -s 200

# Or with model tag explicitly
python -m scripts.chat_cli -g d8 -s 200
```

---

## 🎛️ Generation Parameters

### Temperature (`-t`)
Controls randomness in responses:
- **0.0-0.3**: Very focused, deterministic (good for factual questions)
- **0.6**: Balanced (default, good for most conversations)
- **0.8-1.0**: More creative, varied responses
- **1.0-2.0**: Very creative, sometimes unpredictable

### Top-K (`-k`)
Limits the number of candidate tokens:
- **10-20**: Very focused, conservative
- **50**: Balanced (default)
- **100-200**: More diverse vocabulary

### Recommended Settings

**For factual questions:**
```bash
python -m scripts.chat_cli -t 0.3 -k 20
```

**For creative writing:**
```bash
python -m scripts.chat_cli -t 0.9 -k 100
```

**For coding help:**
```bash
python -m scripts.chat_cli -t 0.5 -k 50
```

**For general conversation:**
```bash
python -m scripts.chat_cli -t 0.6 -k 50  # Default
```

---

## 🔧 Troubleshooting

### Model Not Found
If you get an error about model not found:
```bash
# Check if checkpoint exists
ls ~/.cache/nanochat/chatsft_checkpoints/d8/

# If empty, SFT training may not have completed
# Check the training logs for errors
```

### Out of Memory
If you get OOM errors during chat:
```bash
# Use CPU instead (slower but works)
python -m scripts.chat_cli --device-type cpu

# Or reduce max tokens in web UI settings
```

### Slow Responses
- **First response is slow**: Model is loading (one-time cost)
- **Subsequent responses**: Should be faster
- **If consistently slow**: Check GPU utilization with `nvidia-smi`

### WSL2 Port Forwarding
If web UI doesn't work in WSL2:
```bash
# Check if port is accessible
curl http://localhost:8000/health

# If not, try binding to 0.0.0.0 explicitly
python -m scripts.chat_web --host 0.0.0.0 --port 8000
```

---

## 📝 Example Conversation

```bash
$ python -m scripts.chat_cli

NanoChat Interactive Mode
--------------------------------------------------
Type 'quit' or 'exit' to end the conversation
Type 'clear' to start a new conversation
--------------------------------------------------

User: What is machine learning?

Assistant: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on that data.

User: Can you give me a simple example?

Assistant: Sure! A simple example is email spam detection. The system learns from thousands of emails labeled as "spam" or "not spam" and then automatically categorizes new emails based on patterns it discovered, like certain words or sender addresses.

User: clear

Conversation cleared.

User: Write a haiku about coding

Assistant: 
Lines of logic flow,
Bugs hide in the shadows deep,
Debug until dawn.

User: quit

Goodbye!
```

---

## 🚀 Quick Start Commands

**After SFT completes, just run:**

```bash
# Interactive chat
python -m scripts.chat_cli

# Or web UI
python -m scripts.chat_web
```

That's it! The model automatically loads from your SFT checkpoint.

---

## 💡 Tips

1. **First run is slow**: Model loading takes 10-30 seconds (one-time)
2. **Use `clear`**: Start fresh conversations when context gets too long
3. **Adjust temperature**: Experiment with different values for different use cases
4. **Check GPU memory**: Use `nvidia-smi` to monitor if responses are slow
5. **Save conversations**: Copy interesting exchanges from the CLI or web UI

Enjoy chatting with your trained model! 🎉

