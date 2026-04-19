# Kaggle Setup & Submission Guide

## Quick Reference

| Item | Value |
|---|---|
| Platform | Kaggle Notebooks (free) |
| GPU | RTX Pro 6000 Blackwell (96GB VRAM) |
| GPU quota | 30 hours/week (free tier) |
| Session limit | 12 hours |
| Submissions | 5/day max |
| Competition URL | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge |
| Submission demo | https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo |

## Setting Up a Training Notebook

### Option A: Fork the demo (fastest)

1. Open https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo
2. Click "Copy & Edit"
3. Settings → Accelerator → **RTX Pro 6000**
4. Settings → Internet → **On**
5. Add your training code where the demo says `# YOUR CODE HERE`

### Option B: New notebook

1. Go to the competition → Code → New Notebook
2. Settings → Accelerator → RTX Pro 6000
3. Settings → Internet → On
4. Add Input → search "Nemotron-3-Nano-30B-A3B-BF16" → Add

### Required installs

```python
!pip install -U transformers>=5.3.0
!pip install peft accelerate
```

## Model Loading

```python
import kagglehub
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,  # Required for this model
    torch_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
```

## Data Access

```python
import polars as pl
train = pl.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv')
# Columns: id, prompt, answer
```

## Submission Flow

### 1. Save adapter

```python
OUTPUT_DIR = "/kaggle/working"
model.save_pretrained(OUTPUT_DIR)
# Creates adapter_config.json and adapter_model.safetensors
```

### 2. Create submission.zip

```python
import subprocess
subprocess.run("zip -m submission.zip *", shell=True, check=True, cwd="/kaggle/working")
```

### 3. Submit

- **UI**: Save & Run All (commit) → Output tab → "Submit to Competition"
- **API**: `kaggle kernels push -p ./notebook-folder --accelerator NvidiaRtxPro6000`

## Programmatic Notebook Execution (Kaggle API)

```bash
# Setup (one-time)
pip install kaggle
# Place kaggle.json from https://www.kaggle.com/settings → Create New API Token → ~/.kaggle/

# Create kernel-metadata.json in notebook folder:
{
  "id": "YOUR_USERNAME/notebook-slug",
  "title": "Nemotron Training",
  "code_file": "train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "competition_sources": ["nvidia-nemotron-model-reasoning-challenge"],
  "kernel_sources": []
}

# Push and run
kaggle kernels push -p ./notebook-folder

# Check status
kaggle kernels status YOUR_USERNAME/notebook-slug

# Download output
kaggle kernels output YOUR_USERNAME/notebook-slug -p ./output
```

## Experiment Strategy

### Priority order:

1. **Fix the basics** — ensure model loads, LoRA applies, training runs on Kaggle
2. **Generate CoT training data** — write solvers for each puzzle category (numeral, cipher, gravity, unit_conversion, bit_manipulation)
3. **SFT with crafted CoT traces** — train model to follow deterministic solution patterns
4. **Prompt format** — test `\boxed{}` answer format matching evaluation
5. **Learning rate / epochs sweep** — tune training hyperparameters
6. **Category-specific tuning** — focus on hard categories (cryptarithm, equation)

### What NOT to bother with:

- RL/GRPO — too slow, SFT with CoT traces is sufficient
- Quantization — 96GB VRAM makes it unnecessary
- External data distillation — the puzzles are 100% solvable with programmatic traces

## Scoring Details

| Parameter | Value (Overview tab, authoritative) |
|---|---|
| max_tokens | 7680 |
| temperature | 0.0 |
| top_p | 1.0 |
| max_num_seqs | 64 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 8192 |

Answer matching: exact string match OR numeric match within `rel_tol=1e-2`. Binary strings must match exactly.

## Key Technical Notes

1. **KV cache bug**: The model's `modeling_nemotron_h.py` has a parameter name mismatch causing 20x generation slowdown. Fix: `transformers >= 5.3.0` + remove `trust_remote_code=True` during training. Keep `trust_remote_code=True` for submission (competition eval uses it).
2. **bitsandbytes broken**: Both 4-bit and 8-bit fail on Mamba SSM layers. Use bf16 only.
3. **Gradient checkpointing**: Incompatible with NemotronH native transformers implementation. Don't use it.
4. **Target modules**: `.*\.(in_proj|out_proj|up_proj|down_proj)$` is correct — covers both Mamba and MoE layers. ~880M trainable params with rank 32.
5. **Data column mismatch**: CSV has `prompt`/`answer` columns, not `question`/`answer`.
