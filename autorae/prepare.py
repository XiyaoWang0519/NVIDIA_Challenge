"""
Data preparation and evaluation harness for the NVIDIA Nemotron competition.
This file is FIXED — do not modify.

Downloads competition data, sets up the validation split, and provides
the fixed accuracy evaluation metric used for all experiments.
"""

import os
import gc
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Competition constraints (fixed per Kaggle rules)
# ---------------------------------------------------------------------------
LORA_RANK = 32
LORA_ALPHA = 16
MAX_TOKENS = 7680
TOP_P = 1.0
TEMPERATURE = 0.0
MAX_NUM_SEQS = 64
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 8192

# Training constraints (from competition demo)
TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"
LORA_DROPOUT = 0.05

# Time budget per experiment (seconds) — agents may tune internal hyperparameters
# but the outer loop controls wall-clock time
TIME_BUDGET = 600  # 10 minutes training per experiment (excludes eval/compilation)

# Model name from competition demo
MODEL_NAME = "nvidia/Nemotron-3-Nano-30B-A3B-bf16"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def download_competition_data():
    """Download competition data from Kaggle if not present."""
    from datasets import load_dataset
    
    cache_dir = os.path.expanduser("~/.cache/autorae")
    data_dir = os.path.join(cache_dir, "competition_data")
    
    if os.path.exists(data_dir):
        print(f"Competition data already exists at {data_dir}")
        return data_dir
    
    print("Downloading competition data...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Load competition dataset (this handles both train.csv and test.csv)
    # If Kaggle credentials are not set, try HuggingFace mirror or mock for local dev
    try:
        dataset = load_dataset("kaggle", "nvidia-nemotron-model-reasoning-challenge", cache_dir=cache_dir)
        return dataset
    except Exception as e:
        print(f"Could not download from Kaggle: {e}")
        print("Please download competition data manually and place in ~/.cache/autorae/competition_data/")
        return None


def make_reasoning_dataset(split="train"):
    """Load and preprocess the reasoning puzzle dataset."""
    cache_dir = os.path.expanduser("~/.cache/autorae")
    data_path = os.path.join(cache_dir, "competition_data", f"{split}.csv")
    
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Run prepare.py first.")
        return None
    
    import pandas as pd
    df = pd.read_csv(data_path)
    
    class ReasoningDataset(Dataset):
        def __init__(self, df):
            self.df = df
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            return {
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
                "id": row.get("id", idx),
            }
    
    return ReasoningDataset(df)


def make_sft_dataset(text_column="text", max_samples=None):
    """
    Create a dataset for supervised fine-tuning.
    Uses competition data formatted as instruction-response pairs.
    
    Format: <|user|> {question} <|assistant|> {reasoning_chain} {answer}
    """
    from datasets import load_dataset
    
    cache_dir = os.path.expanduser("~/.cache/autorae")
    
    try:
        # Try to load from competition data
        dataset = load_dataset("kaggle", "nvidia-nemotron-model-reasoning-challenge", cache_dir=cache_dir)
        if "train" in dataset:
            ds = dataset["train"]
            if max_samples:
                ds = ds.select(range(min(max_samples, len(ds))))
            return ds
    except Exception as e:
        print(f"Could not load competition dataset: {e}")
    
    return None


# ---------------------------------------------------------------------------
# Evaluation harness (FIXED metric)
# ---------------------------------------------------------------------------

def evaluate_accuracy(model, tokenizer, val_dataset, num_samples=100, max_new_tokens=512):
    """
    Evaluate model accuracy on reasoning puzzles.
    This is the FIXED evaluation metric for all experiments.
    
    Returns accuracy (fraction of correct answers).
    """
    if val_dataset is None:
        print("Warning: No validation dataset, returning 0.0")
        return 0.0
    
    from tqdm import tqdm
    
    model.eval()
    correct = 0
    total = min(num_samples, len(val_dataset))
    
    for i in tqdm(range(total), desc="Evaluating"):
        item = val_dataset[i]
        question = item["question"]
        true_answer = str(item["answer"]).strip().upper()
        
        # Format prompt
        prompt = f"<|user|>\n{question}\n<|assistant|>\n"
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=(TEMPERATURE > 0),
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Extract answer from response (simple heuristic: last token or explicitly marked)
        pred_answer = extract_answer(response)
        
        if pred_answer == true_answer:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def extract_answer(response: str) -> str:
    """
    Extract the final answer from a reasoning chain.
    Override this in train.py to experiment with different answer extraction methods,
    but note that the FINAL evaluation uses the ground truth comparison.
    """
    # Simple extraction: look for answer markers or use last line
    response = response.strip()
    
    # Look for explicit answer markers
    for marker in ["answer:", "Answer:", "ANSWER:", "final answer:", "Final Answer:"]:
        if marker in response:
            answer = response.split(marker)[-1].strip()
            return answer.strip().upper()[:50]  # Normalize and limit length
    
    # Fallback: last line
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if lines:
        return lines[-1].strip().upper()[:50]
    
    return response.upper()[:50]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("NVIDIA Nemotron Reasoning Challenge — Data Preparation")
    print("=" * 60)
    
    print(f"\nCompetition constraints:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  LoRA rank: {LORA_RANK} (max)")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
    
    print("\nDownloading competition data...")
    data = download_competition_data()
    
    if data is not None:
        print(f"\nDataset loaded: {list(data.keys())}")
        for split, ds in data.items():
            print(f"  {split}: {len(ds)} samples")
    
    print("\nData preparation complete!")
    print("Run 'train.py' to start training.")
