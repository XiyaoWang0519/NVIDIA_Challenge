# autoresearch-nemotron

Adapted autoresearch framework for the **NVIDIA Nemotron Model Reasoning Challenge**.

This project adapts karpathy/autoresearch for fine-tuning **NVIDIA Nemotron-3-Nano-30B** with LoRA adapters for reasoning tasks.

## Overview

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch), this project modifies the original framework to:

- Use **Nemotron-3-Nano-30B** as the base model (instead of training from scratch)
- Support **LoRA fine-tuning** (rank ≤ 32 as per competition rules)
- Implement **GRPO/RLVR** training for reasoning tasks
- Support **Chain-of-Thought** prompting data
- Optimize for **accuracy** (competition metric)

## Competition Requirements

| Requirement | Value |
|------------|-------|
| Base Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| LoRA Rank | ≤ 32 |
| Max Tokens | 7680 |
| Temperature | 0.0 |
| Submission | `submission.zip` with LoRA adapter |

## Project Structure

```
autoresearch-nemotron/
├── train.py           # Main training script (adapts karpathy/autoresearch)
├── prepare.py         # Data preparation for reasoning tasks
├── config.py          # Competition-specific configuration
├── program.md         # Research agent instructions
├── requirements.txt   # Dependencies
├── pyproject.toml     # Project configuration
└── README.md
```

## Quick Start

### Option 1: Google Cloud VM (Recommended - $240 credit)

```bash
# 1. Connect and create VM with GPU
./gcp_connect.sh

# 2. On the VM, run setup
./setup_gcp.sh

# 3. Prepare data
python prepare.py

# 4. Train (5 min experiment)
./run_training.sh

# 5. Sync results locally
./sync_results.sh

# 6. Stop VM when done (save money!)
./gcp_stop.sh
```

### Option 2: Local (requires GPU with 16GB+ VRAM)

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # Optional

# 2. Prepare data
python prepare.py

# 3. Train (5 min experiment)
python train.py

# 4. Create submission
python create_submission.py
```

## Key Differences from Original autoresearch

| Aspect | Original | This Project |
|--------|----------|--------------|
| Base Model | Train from scratch | Load Nemotron-3-Nano-30B |
| Training | Full model pretraining | LoRA fine-tuning |
| Optimizer | Muon + AdamW | AdamW with LoRA |
| Data | Raw text | Reasoning/CoT data |
| Metric | val_bpb | Accuracy |
| Time Budget | 5 minutes | Configurable |

## Configuration

Key settings in `config.py`:

```python
# Model
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
MAX_SEQ_LEN = 7680  # Competition max_tokens
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
GRPO_GROUPS = 8
TIME_BUDGET = 300  # 5 minutes per experiment

# Data (reasoning datasets)
DATASETS = ["openmathreasoning", "lightweight-sft-reasoning"]
```

## Training Methods Supported

1. **Supervised Fine-Tuning (SFT)** - Standard instruction tuning
2. **GRPO** - Group Relative Policy Optimization for reasoning
3. **RLVR** - Reinforcement Learning with Verifiable Rewards
4. **Best-of-N** - Sampling and selection

## Data Format

Training data should be in the following format:

```json
{
  "prompt": "Solve this problem: What is 2 + 2? Think step by step.",
  "response": "Let me solve this step by step...\n\nStep 1: Identify the numbers\nStep 2: Add them together\n2 + 2 = 4\n\nTherefore, the answer is 4.",
  "answer": "4"
}
```

## Competition Submission

Generate your submission.zip:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./lora_output")
model.eval()

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./submission")
tokenizer.save_pretrained("./submission")

# Create submission.zip
import zipfile
with zipfile.ZipFile("submission.zip", "w") as zf:
    zf.write("submission/adapter_config.json", "adapter_config.json")
    # ... other files
```

## References

- [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [NVIDIA Nemotron-3-Nano-30B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [GRPO Paper](https://arxiv.org/abs/2505.07686)

## License

MIT
