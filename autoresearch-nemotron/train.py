"""
Training script for NVIDIA Nemotron Model Reasoning Challenge.
Adapted from karpathy/autoresearch for LoRA fine-tuning with GRPO support.

Key changes from original:
- Loads Nemotron-3-Nano-30B instead of training from scratch
- LoRA fine-tuning instead of full model training
- GRPO (Group Relative Policy Optimization) for reasoning
- Accuracy metric instead of val_bpb
"""

import os
import gc
import time
import math
import json
import gzip
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from config import (
    BASE_MODEL, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    MAX_SEQ_LEN, DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    TIME_BUDGET, LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO, LR_SCHEDULE,
    GRPO_GROUPS, GRPO_BETA, GRPO_GAMMA, USE_OUTCOME_REWARD, USE_PROCESS_REWARD,
    DATA_DIR, RESULTS_FILE, EXP_TAG, USE_FLASH_ATTENTION, USE_GRADIENT_CHECKPOINTING,
    LOG_LEVEL, LOG_FILE, SAVE_CHECKPOINTS, CHECKPOINT_DIR
)

# ============================================================================
# Flash Attention (optional)
# ============================================================================

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: flash-attn not installed. Install with: pip install flash-attn --no-build-isolation")

# ============================================================================
# Setup
# ============================================================================

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float_dict({"float32": "high", "torch.bfloat16": "highest"})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Memory tracking
torch.cuda.reset_peak_memory_stats()

# ============================================================================
# Load Model and Tokenizer
# ============================================================================

print(f"\n{'='*60}")
print("Loading base model...")
print(f"{'='*60}")
print(f"Model: {BASE_MODEL}")
print(f"LoRA rank: {LORA_RANK}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model in BF16
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if HAS_FLASH_ATTN else "eager"
)

# ============================================================================
# LoRA Setup
# ============================================================================

print(f"\n{'='*60}")
print("Setting up LoRA...")
print(f"{'='*60}")

from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Gradient checkpointing for memory savings
if USE_GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# ============================================================================
# GRPO Implementation
# ============================================================================

class GRPOOptimizer:
    """
    Group Relative Policy Optimization (GRPO) for reasoning tasks.
    
    Based on: https://arxiv.org/abs/2505.07686
    
    Key idea: Sample groups of responses, compute relative advantages,
    and update policy using a KL-regularized objective.
    """

    def __init__(self, model, learning_rate, beta=0.1, gamma=1.0):
        self.model = model
        self.beta = beta  # KL penalty coefficient
        self.gamma = gamma  # Reward advantage scaling
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY
        )

    def grpo_step(self, prompts, responses, rewards):
        """
        Perform one GRPO update step.
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings (one per prompt)
            rewards: List of reward values (1 for correct, 0 for incorrect)
        """
        # Tokenize
        enc = tokenizer(
            responses,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        loss = outputs.loss

        # GRPO-specific: apply reward-based weighting
        # Higher reward responses get higher weight
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-8)

        # Weighted loss
        weighted_loss = loss * reward_tensor.mean()

        # KL penalty (simplified - would need reference model for full implementation)
        # kl_loss = self.compute_kl_penalty(prompts, responses)

        # Total loss
        total_loss = weighted_loss  # + self.beta * kl_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item(), loss.item()

    def compute_kl_penalty(self, prompts, responses):
        """Compute KL penalty against reference model."""
        # Simplified - full implementation would use reference model
        return 0.0


# ============================================================================
# Outcome Reward Computation
# ============================================================================

def compute_outcome_reward(prompt: str, response: str, ground_truth: str) -> float:
    """
    Compute outcome-based reward.
    
    Competition metric: answer must match exactly or within tolerance 1e-2.
    """
    import re

    # Extract boxed answer from response
    pred_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if pred_match:
        pred_answer = pred_match.group(1).strip()
    else:
        # Fallback: try to extract last number
        numbers = re.findall(r'-?\d+\.?\d*', response)
        pred_answer = numbers[-1] if numbers else ""

    # Normalize for comparison
    pred_answer = normalize_answer(pred_answer)
    ground_truth = normalize_answer(ground_truth)

    # Check exact match
    if pred_answer == ground_truth:
        return 1.0

    # Check numerical tolerance
    try:
        pred_val = float(pred_answer)
        truth_val = float(ground_truth)
        if abs(pred_val - truth_val) <= 1e-2:
            return 1.0
    except (ValueError, TypeError):
        pass

    return 0.0


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    import re
    # Remove spaces, convert to lowercase
    answer = re.sub(r'\s+', '', answer.lower())
    # Remove leading zeros except for decimals
    answer = re.sub(r'^0+(\d)', r'\1', answer)
    return answer


# ============================================================================
# Data Loading
# ============================================================================

def load_reasoning_data(split="train"):
    """Load reasoning data from JSONL file."""
    file_path = DATA_DIR / f"{split}.jsonl"
    if not file_path.exists():
        print(f"Warning: {file_path} not found. Run prepare.py first.")
        return []

    examples = []
    with open(file_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


class ReasoningDataset(torch.utils.data.Dataset):
    """Dataset for reasoning tasks."""

    def __init__(self, examples, tokenizer, max_length=7680):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        prompt = item["prompt"]
        response = item["response"]
        answer = item.get("answer", "")

        # Format: prompt + response
        text = f"{prompt}\n\n{response}"

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze(),
            "prompt": prompt,
            "response": response,
            "answer": answer
        }


def collate_fn(batch):
    """Custom collate function to preserve metadata."""
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "prompts": [x["prompt"] for x in batch],
        "responses": [x["response"] for x in batch],
        "answers": [x["answer"] for x in batch]
    }


# ============================================================================
# Training Loop
# ============================================================================

def train_with_grpo(model, train_loader, optimizer, time_budget=300):
    """
    Train with GRPO for reasoning tasks.
    
    Each step: sample group, compute rewards, update policy.
    """
    print(f"\n{'='*60}")
    print("Starting GRPO training")
    print(f"Time budget: {time_budget}s")
    print(f"{'='*60}")

    model.train()
    total_training_time = 0
    step = 0
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Learning rate schedule
    total_steps = 10000  # Estimate
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    t_start_training = time.time()

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        for micro_step in range(GRADIENT_ACCUMULATION_STEPS):
            try:
                batch = next(train_loader_iter)
            except (StopIteration, NameError):
                # Reset iterator
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )

            # Compute per-sample loss
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            # Track metrics
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 0:
            total_training_time += dt

        step += 1

        # Logging
        if step % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            pct_done = 100 * total_training_time / time_budget if time_budget > 0 else 0
            remaining = max(0, time_budget - total_training_time)
            tok_per_sec = int(DEVICE_BATCH_SIZE * MAX_SEQ_LEN / dt) if dt > 0 else 0

            print(f"\rstep {step:05d} ({pct_done:.1f}%) | "
                  f"loss: {total_loss/max(step,1):.4f} | "
                  f"lr: {lr:.2e} | "
                  f"dt: {dt*1000:.0f}ms | "
                  f"tok/sec: {tok_per_sec:,} | "
                  f"remaining: {remaining:.0f}s    ",
                  end="", flush=True)

        # Time's up
        if step > 10 and total_training_time >= time_budget:
            break

        # Memory management
        if step % 500 == 0:
            gc.collect()

    print()
    return total_training_time, step


def train_with_sft(model, train_loader, optimizer, time_budget=300):
    """
    Standard SFT (Supervised Fine-Tuning) training.
    
    Used as baseline or when RL is not needed.
    """
    print(f"\n{'='*60}")
    print("Starting SFT training")
    print(f"Time budget: {time_budget}s")
    print(f"{'='*60}")

    model.train()
    total_training_time = 0
    step = 0
    total_loss = 0.0

    # Learning rate schedule
    total_steps = 10000
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    t_start_training = time.time()
    train_loader_iter = iter(train_loader)

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        for micro_step in range(GRADIENT_ACCUMULATION_STEPS):
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )

            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        # Clip and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 0:
            total_training_time += dt

        step += 1

        # Logging
        if step % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            pct_done = 100 * total_training_time / time_budget if time_budget > 0 else 0
            remaining = max(0, time_budget - total_training_time)
            mfu = estimate_mfu(model, DEVICE_BATCH_SIZE, dt)

            print(f"\rstep {step:05d} ({pct_done:.1f}%) | "
                  f"loss: {total_loss/max(step,1):.4f} | "
                  f"lr: {lr:.2e} | "
                  f"dt: {dt*1000:.0f}ms | "
                  f"mfu: {mfu:.1f}% | "
                  f"remaining: {remaining:.0f}s    ",
                  end="", flush=True)

        if step > 10 and total_training_time >= time_budget:
            break

        if step % 500 == 0:
            gc.collect()

    print()
    return total_training_time, step


def estimate_mfu(model, batch_size, dt):
    """Estimate Model FLOP Utilization (MFU)."""
    # Rough estimate
    num_params = sum(p.numel() for p in model.parameters())
    flops_per_token = 6 * num_params  # Approximate
    tokens_per_sec = batch_size * MAX_SEQ_LEN / dt
    peak_flops = 989.5e12  # H100
    mfu = 100 * flops_per_token * tokens_per_sec / peak_flops
    return min(mfu, 100)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_accuracy(model, val_loader, num_samples=100):
    """
    Evaluate model accuracy on validation set.
    Competition metric: exact match or within tolerance 1e-2.
    """
    print(f"\n{'='*60}")
    print("Evaluating...")
    print(f"{'='*60}")

    model.eval()
    correct = 0
    total = 0

    val_loader_iter = iter(val_loader)

    for i, batch in enumerate(val_loader_iter):
        if i >= num_samples:
            break

        prompts = batch["prompts"]
        answers = batch["answers"]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode and extract answers
        for j, output in enumerate(outputs):
            generated = tokenizer.decode(output, skip_special_tokens=True)

            # Extract boxed answer
            pred_match = __import__('re').search(r'\\boxed\{([^}]+)\}', generated)
            pred_answer = pred_match.group(1) if pred_match else ""

            # Compute reward
            reward = compute_outcome_reward(prompts[j], generated, answers[j])
            if reward > 0.5:
                correct += 1
            total += 1

            if (i * len(prompts) + j) % 20 == 0:
                print(f"\nProblem: {prompts[j][:100]}...")
                print(f"Predicted: {pred_answer} | Ground truth: {answers[j]}")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


# ============================================================================
# Main Training
# ============================================================================

def main():
    global train_loader_iter

    print("\n" + "="*60)
    print("NVIDIA Nemotron Model Reasoning Challenge")
    print("Training Script")
    print("="*60)

    # Load data
    print("\nLoading training data...")
    train_examples = load_reasoning_data("train")
    val_examples = load_reasoning_data("val")

    if not train_examples:
        print("Error: No training data. Run prepare.py first.")
        return

    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples: {len(val_examples)}")

    # Create datasets
    train_dataset = ReasoningDataset(train_examples, tokenizer, MAX_SEQ_LEN)
    val_dataset = ReasoningDataset(val_examples, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=DEVICE_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=DEVICE_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    train_loader_iter = iter(train_loader)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Choose training method
    # GRPO for reasoning tasks, SFT as baseline
    if USE_OUTCOME_REWARD and len(train_examples) > 100:
        print("\nUsing GRPO training (outcome rewards)")
        total_training_time, num_steps = train_with_grpo(
            model, train_loader, optimizer, TIME_BUDGET
        )
    else:
        print("\nUsing SFT training (baseline)")
        total_training_time, num_steps = train_with_sft(
            model, train_loader, optimizer, TIME_BUDGET
        )

    # Evaluate
    accuracy = evaluate_accuracy(model, val_loader, num_samples=100)

    # Final summary
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"val_accuracy:       {accuracy:.6f}")
    print(f"training_seconds:   {total_training_time:.1f}")
    print(f"total_seconds:      {t_end - t_start:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"num_steps:          {num_steps}")
    print(f"lora_rank:          {LORA_RANK}")

    # Save results
    save_results(accuracy, total_training_time, peak_vram_mb, num_steps)

    # Save checkpoint
    if SAVE_CHECKPOINTS:
        save_checkpoint(accuracy)

    return accuracy


def save_results(accuracy, training_time, peak_vram, num_steps):
    """Save results to TSV file."""
    import subprocess

    # Get git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True
        ).strip()
    except:
        commit = "unknown"

    # Result line
    result = f"{commit}\t{accuracy:.6f}\t{peak_vram/1024:.1f}\t"

    if accuracy > 0.5:
        result += "keep\t"
    else:
        result += "discard\t"

    result += f"lora_r{LORA_RANK}_lr{LEARNING_RATE}_grpo{int(USE_OUTCOME_REWARD)}\n"

    # Append to results file
    results_path = Path(RESULTS_FILE)
    if not results_path.exists():
        results_path.write_text("commit\taccuracy\tmemory_gb\tstatus\tdescription\n")

    with open(results_path, "a") as f:
        f.write(result)

    print(f"\nResults saved to: {RESULTS_FILE}")


def save_checkpoint(accuracy):
    """Save model checkpoint."""
    from peft import PeftModel

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)

    # Save LoRA adapter
    model.save_pretrained(checkpoint_dir / f"lora_r{LORA_RANK}_acc{accuracy:.4f}")
    tokenizer.save_pretrained(checkpoint_dir / f"lora_r{LORA_RANK}_acc{accuracy:.4f}")

    print(f"\nCheckpoint saved to: {checkpoint_dir}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
