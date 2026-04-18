"""
Create submission.zip for NVIDIA Nemotron Model Reasoning Challenge.

This script:
1. Loads the base model (Nemotron-3-Nano-30B)
2. Loads the trained LoRA adapter
3. Merges LoRA weights into base model
4. Creates submission.zip in competition format
"""

import os
import zipfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig

# Configuration
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
LORA_PATH = "./checkpoints"  # Path to your trained LoRA adapter
OUTPUT_DIR = "./submission"
OUTPUT_ZIP = "submission.zip"


def create_submission():
    """Create competition submission.zip."""
    print("=" * 60)
    print("Creating Competition Submission")
    print("=" * 60)

    # Find the best LoRA checkpoint
    lora_path = find_best_checkpoint()
    if not lora_path:
        print("Error: No LoRA checkpoint found. Train a model first.")
        return

    print(f"\nUsing LoRA checkpoint: {lora_path}")

    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load on CPU first for merging
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    # Merge and unload
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Save merged model
    print(f"\nSaving to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Create submission.zip
    print(f"\nCreating {OUTPUT_ZIP}...")
    create_zip()

    print("\n" + "=" * 60)
    print("Submission created successfully!")
    print("=" * 60)
    print(f"\nSubmit: {OUTPUT_ZIP}")
    print(f"Model: {OUTPUT_DIR}")


def find_best_checkpoint():
    """Find the best (highest accuracy) LoRA checkpoint."""
    checkpoint_dir = Path(LORA_PATH)
    if not checkpoint_dir.exists():
        return None

    # Find all lora_* directories
    checkpoints = list(checkpoint_dir.glob("lora_r*"))
    if not checkpoints:
        return None

    # Sort by accuracy (extracted from directory name)
    def get_accuracy(path):
        name = path.name
        if "acc" in name:
            try:
                acc_str = name.split("acc")[1].split("_")[0]
                return float(acc_str)
            except:
                pass
        return 0.0

    checkpoints.sort(key=get_accuracy, reverse=True)
    return str(checkpoints[0])


def create_zip():
    """Create submission.zip with model files."""
    output_dir = Path(OUTPUT_DIR)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # Required files for competition submission
        required_files = [
            "adapter_config.json",
            "adapter_model.safetensors",  # or pytorch_model.bin
        ]

        # Add all files from output directory
        for file in output_dir.rglob("*"):
            if file.is_file():
                arcname = file.name
                zf.write(file, arcname)
                print(f"  Adding: {arcname}")

    print(f"\nCreated: {OUTPUT_ZIP}")
    print(f"Size: {Path(OUTPUT_ZIP).stat().st_size / 1024 / 1024:.2f} MB")


def verify_submission():
    """Verify submission.zip contents."""
    print("\n" + "=" * 60)
    print("Verifying Submission")
    print("=" * 60)

    if not Path(OUTPUT_ZIP).exists():
        print(f"Error: {OUTPUT_ZIP} not found")
        return False

    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
        files = zf.namelist()
        print(f"\nFiles in {OUTPUT_ZIP}:")
        for f in files:
            info = zf.getinfo(f)
            print(f"  {f}: {info.file_size / 1024:.2f} KB")

        # Check required files
        required = ["adapter_config.json"]
        has_required = all(f in files for f in required)

        if has_required:
            print("\n✓ Submission format valid")
        else:
            print("\n✗ Missing required files")
            return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create competition submission")
    parser.add_argument("--verify", action="store_true", help="Verify existing submission")
    args = parser.parse_args()

    if args.verify:
        verify_submission()
    else:
        create_submission()
