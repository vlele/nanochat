# Chat with First SFT Checkpoint

## Step 1: Check Available Checkpoints

First, see what SFT checkpoints you have:

```bash
ls -lh ~/.cache/nanochat/chatsft_checkpoints/d8/
```

You should see files like:
- `model_000100.pt` - First checkpoint (step 100)
- `model_000200.pt` - Final checkpoint (step 200)
- `meta_000100.json` - Metadata for step 100
- `meta_000200.json` - Metadata for step 200

## Step 2: Chat with First Checkpoint

Since checkpoints are saved every 100 steps, the **first checkpoint is at step 100**.

### Interactive Chat Mode

```bash
python -m scripts.chat_cli -i sft -s 100
```

Or if you want to be explicit about the model tag:

```bash
python -m scripts.chat_cli -i sft -g d8 -s 100
```

### Quick Test (Single Prompt)

```bash
python -m scripts.chat_cli -i sft -s 100 -p "Hello! How are you?"
```

## Step 3: Compare with Final Checkpoint

To compare the first checkpoint (step 100) with the final one (step 200):

```bash
# First checkpoint
python -m scripts.chat_cli -i sft -s 100 -p "Explain machine learning"

# Final checkpoint (default, or explicitly)
python -m scripts.chat_cli -i sft -s 200 -p "Explain machine learning"
```

## Notes

- **Step 100** is the first saved checkpoint (since `save_every=100`)
- **Step 200** is the final checkpoint (end of training)
- If you don't specify `-s`, it loads the latest checkpoint automatically
- The `-i sft` flag specifies the source (SFT checkpoints), which is the default anyway

## Example Session

```bash
$ python -m scripts.chat_cli -i sft -s 100

NanoChat Interactive Mode
--------------------------------------------------
Type 'quit' or 'exit' to end the conversation
Type 'clear' to start a new conversation
--------------------------------------------------

User: What is 2+2?

Assistant: 2 + 2 equals 4.

User: quit

Goodbye!
```

