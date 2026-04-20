"""
NVIDIA Nemotron Reasoning Challenge — Kaggle Training Notebook
Paste each cell (separated by # %% markers) into a Kaggle notebook cell.

Setup:
  1. Fork https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo
  2. Settings → Accelerator → RTX Pro 6000, Internet → On
  3. Replace all cells with the contents below
"""

# %% Cell 1: Install dependencies

import subprocess
subprocess.run("pip install -q peft accelerate", shell=True, check=True)

# %% Cell 2: Imports and config

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import math
import time
import torch
import polars as pl
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm.auto import tqdm

# Competition constraints (do not change)
LORA_RANK = 32
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"

# Training hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_SEQ_LEN = 2048
NUM_EPOCHS = 1
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
SEED = 42

OUTPUT_DIR = "/kaggle/working"

# %% Cell 3: Load data and model path

import kagglehub

MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
DATA_PATH = "/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv"

train_df = pl.read_csv(DATA_PATH)
print(f"Training data: {len(train_df)} samples")
print(train_df.head())

# %% Cell 4: Load model and tokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %% Cell 5: Prepare training dataset

class ReasoningDataset(Dataset):
    """
    Formats competition data as chat messages matching the eval pipeline:
    - User: {prompt}\nPlease put your final answer inside \boxed{}
    - Assistant: \boxed{ANSWER}

    Loss is computed ONLY on assistant tokens (user tokens masked with -100).
    """

    def __init__(self, df, tokenizer, max_length=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        prompts = df["prompt"].to_list()
        answers = df["answer"].to_list()

        for prompt, answer in zip(prompts, answers):
            user_content = f"{prompt}\nPlease put your final answer inside \\boxed{{}}"
            assistant_content = f"\\boxed{{{answer}}}"

            # Tokenize user-only to find the boundary
            user_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            user_ids = tokenizer(user_text, add_special_tokens=False)["input_ids"]
            user_len = len(user_ids)

            # Tokenize full conversation (user + assistant)
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )

            self.samples.append({"text": full_text, "user_len": user_len})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask user tokens and padding — only train on assistant response
        labels[:sample["user_len"]] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


dataset = ReasoningDataset(train_df, tokenizer, MAX_SEQ_LEN)
print(f"Dataset: {len(dataset)} samples")

# Sanity check
sample = dataset[0]
non_masked = (sample["labels"] != -100).sum().item()
print(f"Sample non-masked tokens (assistant): {non_masked}")

# %% Cell 6: Train

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

set_seed(SEED)
torch.cuda.manual_seed(SEED)
model.train()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
)

num_steps = len(dataloader) * NUM_EPOCHS // GRAD_ACCUM
warmup_steps = int(num_steps * WARMUP_RATIO)

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, num_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"Training {num_steps} steps ({len(dataloader)} batches x {NUM_EPOCHS} epoch / {GRAD_ACCUM} accum)")
print(f"Estimated time: ~{num_steps * 15 // 60} min (assuming ~15s/step)")

step = 0
running_loss = 0.0
t0 = time.time()

for epoch in range(NUM_EPOCHS):
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        running_loss += loss.item() * GRAD_ACCUM

        if (batch_idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0:
                avg = running_loss / step
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(f"  step {step:4d} | loss {avg:.4f} | lr {lr:.2e} | {elapsed:.0f}s")

        # Free memory
        del outputs, loss
        if (batch_idx + 1) % GRAD_ACCUM == 0:
            torch.cuda.empty_cache()

print(f"\nTraining done in {time.time()-t0:.0f}s, {step} steps, avg loss {running_loss/max(step,1):.4f}")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

# %% Cell 7: Save and submit

model.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

import subprocess
subprocess.run("zip -m submission.zip *", shell=True, check=True, cwd=OUTPUT_DIR)
print("submission.zip created!")
