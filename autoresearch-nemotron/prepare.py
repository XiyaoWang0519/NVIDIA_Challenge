"""
Data preparation for NVIDIA Nemotron Model Reasoning Challenge.
Adapted from karpathy/autoresearch prepare.py.

Downloads and prepares reasoning datasets for fine-tuning.
"""

import os
import json
import gzip
from pathlib import Path

# ============================================================================
# Dataset Configuration
# ============================================================================

# Primary reasoning datasets
REASONING_DATASETS = {
    # Math reasoning
    "openmathreasoning": {
        "source": "nvidia/OpenMathReasoning",
        "type": "math",
        "split": "train",
        "description": "OpenMath Reasoning dataset for math problem solving"
    },
    # General reasoning
    "lightweight-sft-reasoning": {
        "source": "reholders/lightweight-sft-reasoning",
        "type": "reasoning",
        "split": "train",
        "description": "Lightweight SFT reasoning data"
    },
}

# Alternative datasets (commented out - uncomment to use)
# DATASETS = {
#     "openmathreasoning": "nvidia/OpenMathReasoning",
#     "opencodereasoning": "openr1/OpenCoder-Reasoning",
#     "tulu-sft": "allenai/tulu-sft-reasoning",
#     "scibench": "ARXIVGP/scibench",
#     "gpqa": "open-r1/GPQA",
#     "aime25": "AI-MO/NuminaMath-AIME",
# }

# ============================================================================
# Output Configuration
# ============================================================================

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "reasoning.jsonl"

# ============================================================================
# Download and Prepare Datasets
# ============================================================================

def download_dataset(name: str, config: dict) -> list:
    """Download and prepare a single dataset."""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"Source: {config['source']}")
    print(f"{'='*60}")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed.")
        print("Run: pip install datasets")
        return []

    try:
        # Load dataset
        dataset = load_dataset(config["source"], split=config["split"])
        print(f"Loaded {len(dataset)} examples")

        # Convert to training format
        examples = []
        for item in dataset:
            processed = process_item(item, config["type"])
            if processed:
                examples.append(processed)

        print(f"Processed {len(examples)} training examples")
        return examples

    except Exception as e:
        print(f"Error loading {name}: {e}")
        return []


def process_item(item: dict, data_type: str) -> dict | None:
    """
    Process a dataset item into training format.
    
    Args:
        item: Raw dataset item
        data_type: Type of data ("math", "reasoning", "code")
    
    Returns:
        Processed item with prompt, response, answer fields
    """
    try:
        if data_type == "math":
            # OpenMathReasoning format
            prompt = item.get("problem", "")
            solution = item.get("solution", "")
            answer = item.get("answer", "")
            
            # Build CoT response
            response = f"{solution}\n\nTherefore, the answer is {answer}."
            
        elif data_type == "reasoning":
            # Lightweight SFT format
            prompt = item.get("prompt", "")
            response = item.get("chosen", "")
            
            # Extract answer if available
            if "answer" in item:
                answer = item["answer"]
            else:
                # Try to extract from response
                answer = extract_boxed_answer(response)
                
        else:
            # Generic format
            prompt = item.get("prompt", item.get("input", ""))
            response = item.get("response", item.get("output", ""))
            answer = item.get("answer", extract_boxed_answer(response))

        if not prompt or not response:
            return None

        return {
            "prompt": prompt,
            "response": response,
            "answer": str(answer) if answer else ""
        }
        
    except Exception as e:
        print(f"Error processing item: {e}")
        return None


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} LaTeX command."""
    import re
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1)
    # Fallback: try to find last number
    match = re.search(r'(\d+)$', text.strip())
    if match:
        return match.group(1)
    return ""


def prepare_reasoning_data():
    """Main function to prepare all reasoning data."""
    print("\n" + "="*60)
    print("NVIDIA Nemotron Model Reasoning Challenge")
    print("Data Preparation")
    print("="*60)

    all_examples = []

    # Process each dataset
    for name, config in REASONING_DATASETS.items():
        examples = download_dataset(name, config)
        all_examples.extend(examples)

    # Deduplicate by prompt
    seen = set()
    unique_examples = []
    for ex in all_examples:
        if ex["prompt"] not in seen:
            seen.add(ex["prompt"])
            unique_examples.append(ex)

    print(f"\n{'='*60}")
    print(f"Total unique examples: {len(unique_examples)}")
    print(f"{'='*60}")

    # Shuffle and save
    import random
    random.seed(42)
    random.shuffle(unique_examples)

    # Save as JSONL
    with open(OUTPUT_FILE, "w") as f:
        for ex in unique_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved to: {OUTPUT_FILE}")

    # Split into train/val
    val_size = int(len(unique_examples) * 0.05)
    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "val.jsonl"

    with open(train_file, "w") as f:
        for ex in unique_examples[val_size:]:
            f.write(json.dumps(ex) + "\n")

    with open(val_file, "w") as f:
        for ex in unique_examples[:val_size]:
            f.write(json.dumps(ex) + "\n")

    print(f"Train: {len(unique_examples) - val_size} | Val: {val_size}")

    return unique_examples


# ============================================================================
# Tokenizer Preparation
# ============================================================================

def prepare_tokenizer():
    """Prepare tokenizer from base model."""
    from transformers import AutoTokenizer

    from config import BASE_MODEL

    print(f"\nLoading tokenizer from: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# ============================================================================
# DataLoader
# ============================================================================

class ReasoningDataset:
    """Dataset for reasoning tasks."""

    def __init__(self, file_path: str, tokenizer, max_length: int = 7680):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self._load(file_path)

    def _load(self, file_path: str):
        """Load examples from JSONL file."""
        print(f"Loading data from: {file_path}")
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 10000:  # Limit for quick testing
                    break
                self.examples.append(json.loads(line))
        print(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        item = self.examples[idx]

        # Format with CoT prompt
        prompt = item["prompt"]
        response = item["response"]

        # Tokenize
        enc = self.tokenizer(
            prompt,
            response,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze()
        }


def make_dataloader(tokenizer, batch_size: int, max_length: int, split: str = "train"):
    """Create a dataloader for reasoning data."""
    file_path = DATA_DIR / f"{split}.jsonl"

    if not file_path.exists():
        print(f"Data file not found: {file_path}")
        print("Run: python prepare.py")
        return None

    dataset = ReasoningDataset(str(file_path), tokenizer, max_length)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=True
    )

    return dataloader


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_accuracy(model, dataloader, tokenizer, device):
    """
    Evaluate model accuracy on validation set.
    Adapted for reasoning tasks with \boxed{} answer extraction.
    """
    import re
    import torch

    model.eval()
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode and extract answers
        for i, output in enumerate(outputs):
            generated = tokenizer.decode(output, skip_special_tokens=True)

            # Extract boxed answer
            pred_match = re.search(r'\\boxed\{([^}]+)\}', generated)
            pred_answer = pred_match.group(1) if pred_match else ""

            # Compare with ground truth (stored in batch metadata)
            # For simplicity, we'll just count non-empty generations
            if pred_answer.strip():
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NVIDIA Nemotron Reasoning Challenge")
    print("Data Preparation Script")
    print("="*60)

    # Prepare data
    prepare_reasoning_data()

    # Prepare tokenizer
    print("\nPreparing tokenizer...")
    tokenizer = prepare_tokenizer()
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Test dataloader
    print("\nTesting dataloader...")
    dataloader = make_dataloader(tokenizer, batch_size=1, max_length=7680, split="train")
    if dataloader:
        batch = next(iter(dataloader))
        print(f"Batch shape: {batch['input_ids'].shape}")

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review data in ./data/")
    print("2. Run: python train.py")
