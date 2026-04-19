"""
Nemotron-3-Nano-30B LoRA fine-tuning for reasoning.

This is the file the agent modifies. All hyperparameters, training logic,
data formatting, and model configuration are fair game.

Usage: uv run train.py

Competition constraints (do not change these fixed values):
  - LoRA rank max: 32
  - max_tokens: 7680
  - temperature: 0.0
  - top_p: 1.0
  - max_num_seqs: 64
  - gpu_memory_utilization: 0.85
  - max_model_len: 8192
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
import random
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    set_seed,
)

from prepare import (
    MODEL_NAME,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    MAX_TOKENS,
    TOP_P,
    TEMPERATURE,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    TIME_BUDGET,
    make_sft_dataset,
    make_reasoning_dataset,
    evaluate_accuracy,
    extract_answer,
)

# ---------------------------------------------------------------------------
# Hyperparameters (agent tunes these)
# ---------------------------------------------------------------------------

# LoRA config
LORA_RANK = 32  # Must be <= 32 per competition rules
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"

# Training
DEVICE_BATCH_SIZE = 1  # Per-device, keep small for GPU memory
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 1
MAX_SEQ_LEN = 2048  # Sequence length for training
WARMUP_RATIO = 0.1

# Data
MAX_SAMPLES = None  # None = use all available, or set to e.g. 1000 for quick testing
USE_SYNTHETIC_DATA = True  # Whether to augment with synthetic reasoning data

# Model
BASE_MODEL = MODEL_NAME
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def format_reasoning_prompt(question: str, answer: str = None) -> str:
    """
    Format a reasoning puzzle as a training example.
    Agent may experiment with different prompt formats.
    """
    if answer is not None:
        return f"<|user|>\n{question}\n<|assistant|>\nLet's think step by step. "
    else:
        return f"<|user|>\n{question}\n<|assistant|>\n"


def collate_fn_reasoning(batch, tokenizer, max_length=MAX_SEQ_LEN):
    """
    Collate function for reasoning data.
    Formats: question -> model generates reasoning + answer.
    """
    prompts = []
    for item in batch:
        question = item["question"]
        answer = item.get("answer", "")
        text = format_reasoning_prompt(question, answer)
        prompts.append(text)
    
    encodings = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    
    encodings["labels"] = encodings["input_ids"].clone()
    # Mask padding tokens in labels
    encodings["labels"][encodings["labels"] == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["labels"],
    }


def build_model():
    """Build and load the base model + LoRA."""
    print(f"Loading base model: {BASE_MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with vLLM-style memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Faster attention
    )
    
    # Apply LoRA
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
    
    return model, tokenizer


def setup_optimizer(model, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    """Set up optimizer. Agent may experiment with different optimizers."""
    from torch.optim import AdamW
    
    # Separate LoRA params from frozen params for differential LR
    lora_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = AdamW(
        lora_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    return optimizer


def get_lr_scheduler(optimizer, num_training_steps, warmup_ratio=WARMUP_RATIO):
    """Linear warmup + cosine decay."""
    warmup_steps = int(num_training_steps * warmup_ratio)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    """Main training function — agent modifies this extensively."""
    t_start = time.time()
    set_seed(42)
    torch.cuda.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = build_model()
    
    # Load data
    print("Loading training data...")
    train_dataset = make_sft_dataset(max_samples=MAX_SAMPLES)
    
    if train_dataset is None:
        print("Warning: No training data. Using a minimal dummy dataset for testing.")
        # Create a minimal dummy dataset
        class DummyDataset:
            def __init__(self, size=100):
                self.size = size
                self.questions = [
                    "If all cats are animals, and all animals need water, does this mean all cats need water?",
                    "What is the next number in the sequence: 2, 4, 8, 16, ?",
                    "If A > B and B > C, what is the relationship between A and C?",
                ] * (size // 3 + 1)
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    "question": self.questions[idx % len(self.questions)],
                    "answer": "42",
                    "text": f"<|user|>\n{self.questions[idx % len(self.questions)]}\n<|assistant|>\nThe answer is 42.",
                }
        
        train_dataset = DummyDataset()
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Create dataloader
    def collate_fn(batch):
        return collate_fn_reasoning(batch, tokenizer, MAX_SEQ_LEN)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=DEVICE_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Optimizer and scheduler
    optimizer = setup_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    
    num_training_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    lr_scheduler = get_lr_scheduler(optimizer, num_training_steps, WARMUP_RATIO)
    
    print(f"Training steps: {num_training_steps}")
    print(f"Time budget: {TIME_BUDGET}s")
    
    # Training loop
    model.train()
    total_loss = 0.0
    step = 0
    t_train_start = time.time()
    
    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(train_loader):
            # Check time budget
            elapsed = time.time() - t_train_start
            if elapsed >= TIME_BUDGET:
                print(f"\nTime budget reached ({elapsed:.1f}s). Stopping training.")
                break
            
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step += 1
                
                # Logging
                if step % 10 == 0:
                    elapsed = time.time() - t_train_start
                    avg_loss = total_loss / step
                    lr = lr_scheduler.get_last_lr()[0]
                    remaining = TIME_BUDGET - elapsed
                    print(f"step {step:04d} | loss: {avg_loss:.6f} | lr: {lr:.2e} | elapsed: {elapsed:.1f}s | remaining: {remaining:.1f}s")
    
    # Final summary
    t_end = time.time()
    total_train_time = t_end - t_train_start
    avg_loss = total_loss / max(step, 1)
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  avg_loss: {avg_loss:.6f}")
    print(f"  training_seconds: {total_train_time:.1f}")
    print(f"  peak_vram_mb: {peak_vram:.1f}")
    print(f"  num_steps: {step}")
    print(f"  lora_rank: {LORA_RANK}")
    print(f"  learning_rate: {LEARNING_RATE}")
    print(f"  batch_size: {DEVICE_BATCH_SIZE}")
    print(f"  gradient_accumulation_steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  max_seq_len: {MAX_SEQ_LEN}")
    print(f"  num_samples: {len(train_dataset)}")
    
    # Save adapter
    output_dir = os.path.expanduser("~/.cache/autorae/output")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"\nLoRA adapter saved to: {output_dir}")
    
    return {
        "avg_loss": avg_loss,
        "training_seconds": total_train_time,
        "peak_vram_mb": peak_vram,
        "num_steps": step,
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate():
    """Evaluate the trained model on reasoning puzzles."""
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load base model and LoRA
    print("Loading model and adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    output_dir = os.path.expanduser("~/.cache/autorae/output")
    if os.path.exists(output_dir):
        model = PeftModel.from_pretrained(base_model, output_dir)
        model.eval()
    else:
        print("Warning: No adapter found. Evaluating base model.")
        base_model.eval()
        model = base_model
    
    # Load validation data
    val_dataset = make_reasoning_dataset(split="train")
    
    if val_dataset is None or len(val_dataset) == 0:
        print("No validation dataset available. Skipping evaluation.")
        return None
    
    # Evaluate
    accuracy = evaluate_accuracy(model, tokenizer, val_dataset, num_samples=100, max_new_tokens=MAX_TOKENS)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    return accuracy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = train()
    accuracy = evaluate()
    
    if accuracy is not None:
        results["accuracy"] = accuracy
        print(f"\nFinal accuracy: {accuracy:.4f}")
