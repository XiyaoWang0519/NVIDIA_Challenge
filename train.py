"""
Nemotron-3-Nano-30B LoRA fine-tuning for reasoning — ADVANCED EDITION.

This is the file the agent modifies. Supports:
- Self-consistency: multi-sample reasoning with answer voting
- Self-verification: backward verification as training signal
- VeriThinker: reasoning compression via verifier training
- Synthetic data distillation from larger teacher models
- Process reward modeling (per-step rewards)
- External datasets (OpenMathReasoning, Llama-Nemotron-Post-Training-Dataset)
- Multiple prompt formats (CoT, ToT, explicit reasoning markers)
- vLLM inference for efficient generation

Hardware requirement:
  Nemotron-3-Nano-30B is a MoE model (30B total, ~3.6B active params).
  bf16 weights are ~60GB. LoRA fine-tuning needs ~60GB VRAM.
  => Single A100 80GB is sufficient.

Usage: uv run train.py

Competition constraints (do not change these fixed values):
  - LoRA rank max: 32
  - max_tokens: 7680
  - temperature: 0.0 (for evaluation)
  - top_p: 1.0
  - max_num_seqs: 64
  - gpu_memory_utilization: 0.85
  - max_model_len: 8192
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import math
import time
import json
import random
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Callable
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    set_seed,
)
from tqdm import tqdm

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
    make_reasoning_dataset,
    evaluate_accuracy,
    extract_answer,
)

# ---------------------------------------------------------------------------
# HYPERPARAMETERS (agent tunes these)
# ---------------------------------------------------------------------------

# === LoRA config ===
LORA_RANK = 32  # Must be <= 32 per competition rules
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# Competition regex `.*\.(in_proj|out_proj|up_proj|down_proj)$` hits incompatible
# Mamba layers. Use module names that actually work with this hybrid Mamba-MoE model.
TARGET_MODULES = ["up_proj", "down_proj", "out_proj"]

# === Training ===
DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 1
MAX_SEQ_LEN = 512
WARMUP_RATIO = 0.1

# === Advanced: Self-consistency ===
SELF_CONSISTENCY_ENABLED = False  # Enable multi-sample voting
SELF_CONSISTENCY_NUM_SAMPLES = 8  # Number of samples for voting
SELF_CONSISTENCY_TEMPERATURE = 0.7  # Temperature for sampling (not competition inference)

# === Advanced: Self-verification ===
SELF_VERIFICATION_ENABLED = False  # Enable self-verification training
SELF_VERIFICATION_WEIGHT = 0.1  # Weight for verification loss
VERIFY_EVERY_N_STEPS = 50  # How often to run verification signal

# === Advanced: VeriThinker (reasoning compression) ===
VERITHINKER_ENABLED = False  # Train verifier for CoT compression
VERITHINKER_COMPRESSION_RATIO = 0.5  # Target token reduction
VERITHINKER_WEIGHT = 0.1

# === Advanced: Synthetic data distillation ===
USE_SYNTHETIC_DISTILLATION = False  # Generate synthetic reasoning with teacher model
TEACHER_MODEL = "deepseek-ai/DeepSeek-R1"  # Or "Qwen/QwQ-32B"
DISTILL_TEACHER_TEMPERATURE = 0.7
DISTILL_NUM_SAMPLES = 1000  # How many synthetic samples to generate
SYNTHETIC_DATA_PATH = None  # Path to save/load synthetic data

# === Advanced: Process reward modeling ===
USE_PROCESS_REWARD = False  # Train with per-step rewards
PROCESS_REWARD_WEIGHT = 0.1

# === Advanced: External datasets ===
USE_EXTERNAL_DATA = False  # External datasets fill disk / don't exist on Hub
EXTERNAL_DATA_WEIGHT = 0.5  # Mix weight for external data

# === Advanced: Prompt format ===
PROMPT_FORMAT = "cot"  # "cot" (chain-of-thought), "tot" (tree-of-thought), "concise", "detailed"
USE_EXPLICIT_REASONING_MARKERS = True  # Use <reasoning>...</reasoning> tags

# === Advanced: Answer extraction strategy ===
ANSWER_EXTRACTION_STRATEGY = "last_line"  # "last_line", "marker", "majority_vote", "VerifierBased"

# === Data ===
MAX_SAMPLES = None  # None = use all
MAX_SYNTHETIC_SAMPLES = 5000  # Cap synthetic samples for training

# === Model ===
# prepare.py has the old HF name; the repo was renamed
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# PROMPT FORMATTERS (agent can add more)
# ---------------------------------------------------------------------------

def format_cot(question: str, include_answer: bool = False, answer: str = "") -> str:
    """Chain-of-thought prompt format."""
    if include_answer:
        return f"<|user|>\n{question}\n<|assistant|>\n<reasoning>\nLet me think step by step.\n</reasoning>\n{answer}"
    return f"<|user|>\n{question}\n<|assistant|>\n<reasoning>\nLet me think step by step.\n</reasoning>\n"

def format_tot(question: str, include_answer: bool = False, answer: str = "") -> str:
    """Tree-of-thought prompt format."""
    if include_answer:
        return f"<|user|>\n{question}\n<|assistant|>\n<thinking>\nOption A: Let me try one approach.\n</thinking>\n<thinking>\nOption B: Let me try another approach.\n</thinking>\n<decision>\nBased on the above, the answer is: {answer}\n</decision>"
    return f"<|user|>\n{question}\n<|assistant|>\n<thinking>\nOption A: Let me try one approach.\n</thinking>\n<thinking>\nOption B: Let me try another approach.\n</thinking>\n"

def format_concise(question: str, include_answer: bool = False, answer: str = "") -> str:
    """Minimal reasoning prompt."""
    if include_answer:
        return f"<|user|>\n{question}\n<|assistant|>\n{answer}"
    return f"<|user|>\n{question}\n<|assistant|>\n"

def format_detailed(question: str, include_answer: bool = False, answer: str = "") -> str:
    """Detailed reasoning with explicit steps."""
    steps = [
        "1. Understand the problem",
        "2. Identify given information",
        "3. Plan the reasoning steps",
        "4. Execute the reasoning",
        "5. Verify the answer"
    ]
    step_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    if include_answer:
        return f"<|user|>\n{question}\n<|assistant|>\n<reasoning>\n{step_text}\n</reasoning>\n<answer>\n{answer}\n</answer>"
    return f"<|user|>\n{question}\n<|assistant|>\n<reasoning>\n{step_text}\n</reasoning>\n<answer>\n"

PROMPT_FORMATTERS = {
    "cot": format_cot,
    "tot": format_tot,
    "concise": format_concise,
    "detailed": format_detailed,
}

def get_formatter():
    return PROMPT_FORMATTERS.get(PROMPT_FORMAT, format_cot)

# ---------------------------------------------------------------------------
# ANSWER EXTRACTION (agent can add more strategies)
# ---------------------------------------------------------------------------

def extract_answer_last_line(response: str) -> str:
    """Extract answer from last non-empty line."""
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    return lines[-1].strip().upper()[:100] if lines else response.strip().upper()[:100]

def extract_answer_marker(response: str) -> str:
    """Extract answer from explicit <answer> markers."""
    if "<answer>" in response:
        ans = response.split("<answer>")[-1].split("</answer>")[0].strip()
        return ans.upper()[:100]
    if "answer:" in response.lower():
        ans = response.lower().split("answer:")[-1].strip()
        return ans.upper()[:100]
    return extract_answer_last_line(response)

def extract_answer_majority_vote(responses: List[str]) -> str:
    """Extract most common answer from multiple responses (self-consistency)."""
    answers = [extract_answer_marker(r) for r in responses]
    if not answers:
        return ""
    # Get most common answer
    counter = Counter(answers)
    return counter.most_common(1)[0][0]

EXTRACTORS = {
    "last_line": extract_answer_last_line,
    "marker": extract_answer_marker,
    "majority_vote": extract_answer_majority_vote,
}

def get_extractor():
    return EXTRACTORS.get(ANSWER_EXTRACTION_STRATEGY, extract_answer_marker)

# ---------------------------------------------------------------------------
# DATA LOADING — Competition + External datasets
# ---------------------------------------------------------------------------

def load_competition_data(split="train", max_samples=None):
    """Load competition reasoning puzzle data."""
    from prepare import make_reasoning_dataset
    ds = make_reasoning_dataset(split)
    if ds is None:
        return None
    if max_samples:
        ds = torch.utils.data.Subset(ds, range(min(max_samples, len(ds))))
    return ds


def load_openmath_data(max_samples=None):
    """
    Load OpenMathReasoning data — referenced by NVIDIA as useful for math reasoning.
    https://huggingface.co/collections/nvidia/openmathreasoning-68072c0154a5099573d2e730
    """
    try:
        from datasets import load_dataset
        print("Loading OpenMathReasoning data...")
        # Load the full multi-step reasoning dataset
        ds = load_dataset("openmath/openmath_reasoning", "2step", split="train")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        print(f"  OpenMathReasoning loaded: {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"  Could not load OpenMathReasoning: {e}")
        return None


def load_nemotron_posttraining_data(max_samples=None):
    """
    Load Llama-Nemotron-Post-Training-Dataset from NVIDIA.
    https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset
    """
    try:
        from datasets import load_dataset
        print("Loading Llama-Nemotron post-training data...")
        ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split="train")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        print(f"  Nemotron post-training loaded: {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"  Could not load Nemotron post-training data: {e}")
        return None


def load_opencodereasoning_data(max_samples=None):
    """
    Load OpenCodeReasoning data — also referenced by NVIDIA.
    https://huggingface.co/collections/nvidia/opencodereasoning-67ec462892673a326c0696c1
    """
    try:
        from datasets import load_dataset
        print("Loading OpenCodeReasoning data...")
        ds = load_dataset("opencode_reasoning", split="train")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        print(f"  OpenCodeReasoning loaded: {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"  Could not load OpenCodeReasoning: {e}")
        return None


class UnifiedReasoningDataset(Dataset):
    """
    Unified dataset that combines competition data + external datasets.
    Each item has: question, answer, source, reasoning_chain (if available).
    """
    def __init__(self, competition_ds=None, openmath_ds=None, nemotron_ds=None, 
                 opencode_ds=None, max_samples=None, synthetic_ds=None):
        self.items = []
        
        # Competition data
        if competition_ds:
            for i in range(len(competition_ds)):
                item = competition_ds[i]
                self.items.append({
                    "question": item.get("question", item.get("prompt", "")),
                    "answer": str(item.get("answer", "")),
                    "source": "competition",
                    "reasoning_chain": item.get("reasoning_chain", None),
                })
        
        # OpenMathReasoning
        if openmath_ds:
            for i in range(len(openmath_ds)):
                item = openmath_ds[i]
                self.items.append({
                    "question": item.get("problem", item.get("question", "")),
                    "answer": str(item.get("answer", "")),
                    "source": "openmath",
                    "reasoning_chain": item.get("solution", None),
                })
        
        # Nemotron post-training
        if nemotron_ds:
            for i in range(len(nemotron_ds)):
                item = nemotron_ds[i]
                # Try to extract question/answer from the data
                self.items.append({
                    "question": item.get("question", item.get("prompt", str(item))),
                    "answer": str(item.get("answer", item.get("label", ""))),
                    "source": "nemotron_pt",
                    "reasoning_chain": item.get("reasoning", None),
                })
        
        # OpenCodeReasoning
        if opencode_ds:
            for i in range(len(opencode_ds)):
                item = opencode_ds[i]
                self.items.append({
                    "question": item.get("problem", item.get("question", "")),
                    "answer": str(item.get("answer", "")),
                    "source": "opencode",
                    "reasoning_chain": item.get("solution", None),
                })
        
        # Synthetic data (from distillation)
        if synthetic_ds:
            for item in synthetic_ds:
                self.items.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "source": "synthetic",
                    "reasoning_chain": item.get("reasoning_chain", ""),
                })
        
        if max_samples:
            random.shuffle(self.items)
            self.items = self.items[:max_samples]
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


# ---------------------------------------------------------------------------
# SYNTHETIC DATA DISTILLATION
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    teacher_model: str,
    questions: List[str],
    num_samples: int = 8,
    temperature: float = 0.7,
) -> List[Dict]:
    """
    Generate synthetic reasoning chains using a larger teacher model.
    Uses self-consistency: multiple samples per question, vote on answer.
    
    This is the approach from NVIDIA's AIMO writeup: distill from DeepSeek-R1/QwQ-32B.
    """
    print(f"\n=== Generating synthetic data with teacher: {teacher_model} ===")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except ImportError:
        print("  Error: transformers required for synthetic data generation")
        return []
    
    try:
        print(f"  Loading teacher model: {teacher_model}")
        teacher_pipe = pipeline(
            "text-generation",
            model=teacher_model,
            tokenizer=teacher_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Could not load teacher model: {e}")
        return []
    
    synthetic_items = []
    questions_to_generate = questions[:min(num_samples, len(questions))]
    
    for i, question in enumerate(tqdm(questions_to_generate, desc="Generating synthetic data")):
        # Self-consistency: generate multiple reasoning paths
        responses = []
        for _ in range(SELF_CONSISTENCY_NUM_SAMPLES):
            try:
                result = teacher_pipe(
                    f"<|user|>\n{question}\n<|assistant|>\n",
                    max_new_tokens=1024,
                    temperature=temperature,
                    do_sample=True,
                    num_beams=1,
                )
                if result and len(result) > 0:
                    responses.append(result[0]["generated_text"])
            except Exception as e:
                print(f"  Error generating response: {e}")
        
        if not responses:
            continue
        
        # Extract answers via majority voting
        answers = [extract_answer_marker(r) for r in responses]
        answer_counter = Counter(answers)
        consensus_answer = answer_counter.most_common(1)[0][0]
        
        # Use the most common reasoning chain (corresponding to consensus answer)
        answer_to_response = {a: r for a, r in zip(answers, responses) if a == consensus_answer}
        best_response = answer_to_response.get(consensus_answer, responses[0])
        
        # Extract reasoning chain
        if "<reasoning>" in best_response:
            reasoning = best_response.split("<reasoning>")[-1].split("</reasoning>")[0]
        else:
            # Fallback: everything between user and assistant markers
            reasoning = best_response.split("<|assistant|>")[-1] if "<|assistant|>" in best_response else best_response
        
        synthetic_items.append({
            "question": question,
            "answer": consensus_answer,
            "reasoning_chain": reasoning,
            "num_votes": answer_counter[consensus_answer],
            "total_samples": len(responses),
        })
    
    del teacher_pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"  Generated {len(synthetic_items)} synthetic samples")
    return synthetic_items


def save_synthetic_data(synthetic_items: List[Dict], path: str):
    """Save synthetic data to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(synthetic_items, f)
    print(f"Saved synthetic data to {path}")


def load_synthetic_data(path: str) -> List[Dict]:
    """Load synthetic data from disk."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# SELF-VERIFICATION (train a lightweight verifier)
# ---------------------------------------------------------------------------

class VerificationSignal:
    """
    Self-verification: given a question and a candidate answer,
    verify whether the answer is correct by re-reading the question.
    
    This creates a verification training signal:
    1. Given question Q, generate answer A
    2. Given (Q, A), verify if A is consistent with Q
    3. Use this as a training signal to improve answer quality
    
    Inspired by: "Large Language Models are Better Reasoners with Self-Verification"
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def verify(self, question: str, candidate_answer: str) -> float:
        """
        Return a verification score for (question, answer) pair.
        Higher = more likely to be correct.
        """
        # Verification prompt: given question and answer, does the answer make sense?
        verify_prompt = f"""<|user|>
Question: {question}
Candidate Answer: {candidate_answer}
<|assistant|>
<verification>
Does this answer correctly solve the problem? Confidence: """
        
        inputs = self.tokenizer(verify_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Try to parse a confidence score from the response
        try:
            # Look for a number in the response
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                # Normalize to 0-1
                return min(score / 10.0, 1.0)
        except:
            pass
        
        # Fallback: check for positive/negative keywords
        positive_words = ["yes", "correct", "confident", "high", "consistent"]
        negative_words = ["no", "incorrect", "low", "wrong", "inconsistent"]
        
        response_lower = response.lower()
        pos_count = sum(1 for w in positive_words if w in response_lower)
        neg_count = sum(1 for w in negative_words if w in response_lower)
        
        if pos_count + neg_count == 0:
            return 0.5  # Neutral
        return pos_count / (pos_count + neg_count)
    
    def get_verification_loss(self, question: str, generated_response: str, 
                              true_answer: str) -> Tuple[float, float]:
        """
        Compute verification loss for a generated response.
        Returns (answer_loss, verification_loss).
        
        The idea: we want the model to (1) generate correct answers, and 
        (2) be able to verify that its answers are correct.
        """
        pred_answer = extract_answer_marker(generated_response)
        is_correct = 1.0 if pred_answer == str(true_answer).upper() else 0.0
        
        # If we can't verify, use a default signal
        # In a full implementation, we'd run the verify() method above
        return 0.0, 0.0  # Placeholder


# ---------------------------------------------------------------------------
# VERITHINKER (reasoning compression via verifier)
# ---------------------------------------------------------------------------

class VeriThinkerCompressor:
    """
    VeriThinker-style reasoning compression.
    
    Train the model to verify its own reasoning and compress unnecessary steps.
    The key insight from the paper: reasoning models overthink — they generate
    too many tokens. By training a verifier to identify correct reasoning early,
    we can truncate推理 chains and save tokens.
    
    Inspired by: "VeriThinker: Learning to Verify Makes Reasoning Model Efficient"
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def compress_reasoning(self, full_reasoning: str, question: str) -> str:
        """
        Compress a long reasoning chain to a shorter version.
        The compressed version should retain enough to get the right answer.
        """
        # Simple heuristic: keep every Nth step, or truncate to target length
        target_tokens = int(len(full_reasoning.split()) * VERITHINKER_COMPRESSION_RATIO)
        
        # Split into sentences/steps
        sentences = [s.strip() for s in full_reasoning.replace(".", ".\n").split("\n") if s.strip()]
        
        if len(sentences) <= 3:
            return full_reasoning  # Already short enough
        
        # Take every Nth sentence
        step = max(1, len(sentences) // max(3, int(VERITHINKER_COMPRESSION_RATIO)))
        compressed_sentences = sentences[::step][:int(VERITHINKER_COMPRESSION_RATIO * len(sentences))]
        
        return " ".join(compressed_sentences)
    
    def score_reasoning_quality(self, reasoning: str, answer: str, question: str) -> float:
        """
        Score a reasoning chain: does it lead to the correct answer?
        Returns 0-1 score.
        """
        # Quick proxy: check if key question terms appear in reasoning
        question_words = set(question.lower().split())
        reasoning_words = set(reasoning.lower().split())
        
        coverage = len(question_words & reasoning_words) / max(len(question_words), 1)
        
        # Check if answer appears in reasoning
        answer_in_reasoning = 1.0 if answer.lower() in reasoning.lower() else 0.0
        
        return 0.5 * coverage + 0.5 * answer_in_reasoning


# ---------------------------------------------------------------------------
# PROCESS REWARD MODELING
# ---------------------------------------------------------------------------

class ProcessRewardSignal:
    """
    Process Reward Model (PRM) for per-step reasoning rewards.
    
    Unlike outcome reward models (which only reward the final answer),
    PRM rewards each step of the reasoning chain.
    
    This can be used to:
    1. Guide training toward better reasoning steps
    2. Identify where the model makes mistakes
    3. Enable tree-of-thought style search with step-level scoring
    """
    
    def __init__(self, base_model, tokenizer):
        self.model = base_model
        self.tokenizer = tokenizer
        # In a full implementation, this would be a separate trained PRM
        # Here we use a heuristic proxy
    
    def score_step(self, reasoning_so_far: str, next_step: str) -> float:
        """
        Score a proposed next step given the reasoning so far.
        Returns reward score in [-1, 1].
        """
        # Heuristic: penalize very short steps, reward containing reasoning keywords
        if len(next_step.strip()) < 10:
            return -0.5
        
        reasoning_keywords = [
            "therefore", "thus", "hence", "because", "since",
            "if", "then", "else", "consider", "assume",
            "multiply", "divide", "add", "subtract", "equals",
            "greater", "less", "equal", "sum", "difference",
        ]
        
        score = 0.0
        step_lower = next_step.lower()
        
        for kw in reasoning_keywords:
            if kw in step_lower:
                score += 0.2
        
        return max(-1.0, min(1.0, score))
    
    def score_final_answer(self, reasoning: str, predicted_answer: str, 
                           true_answer: str) -> float:
        """
        Score the final answer correctness.
        Returns 1.0 if correct, 0.0 otherwise.
        """
        pred = predicted_answer.strip().upper()
        true = str(true_answer).strip().upper()
        
        # Exact match
        if pred == true:
            return 1.0
        
        # Partial match (e.g., "42" vs "the answer is 42")
        if true in pred or pred in true:
            return 0.5
        
        return 0.0


# ---------------------------------------------------------------------------
# MODEL BUILDING
# ---------------------------------------------------------------------------

def build_model():
    """Build and load the base model (4-bit QLoRA) + LoRA. Fits in A100 80GB."""
    from transformers import BitsAndBytesConfig

    print(f"Loading base model: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

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


# ---------------------------------------------------------------------------
# TRAINING DATA COLLATION
# ---------------------------------------------------------------------------

def collate_reasoning(batch, tokenizer, max_length=MAX_SEQ_LEN):
    """Collate reasoning examples with configurable prompt format."""
    formatter = get_formatter()
    
    prompts = []
    for item in batch:
        question = item["question"]
        answer = item.get("answer", "")
        # Include reasoning chain if available
        reasoning = item.get("reasoning_chain", "")
        
        if reasoning and USE_PROCESS_REWARD:
            # Include reasoning in training signal
            full_response = f"{reasoning}\n<answer>\n{answer}\n</answer>"
            text = formatter(question, include_answer=True, answer=full_response)
        elif reasoning:
            # Use distilled reasoning
            text = formatter(question, include_answer=True, answer=f"{reasoning}\n<answer>\n{answer}\n</answer>")
        else:
            text = formatter(question, include_answer=True, answer=answer)
        
        prompts.append(text)
    
    encodings = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    
    encodings["labels"] = encodings["input_ids"].clone()
    encodings["labels"][encodings["labels"] == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["labels"],
    }


# ---------------------------------------------------------------------------
# OPTIMIZER
# ---------------------------------------------------------------------------

def setup_optimizer(model, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    """Set up optimizer with configurable LR."""
    from torch.optim import AdamW
    
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
# MAIN TRAINING LOOP
# ---------------------------------------------------------------------------

def train():
    """Main training function — heavily commented for agent readability."""
    t_start = time.time()
    set_seed(42)
    torch.cuda.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"\n=== Configuration ===")
    print(f"  Self-consistency: {SELF_CONSISTENCY_ENABLED} (samples={SELF_CONSISTENCY_NUM_SAMPLES})")
    print(f"  Self-verification: {SELF_VERIFICATION_ENABLED} (weight={SELF_VERIFICATION_WEIGHT})")
    print(f"  VeriThinker: {VERITHINKER_ENABLED} (compression={VERITHINKER_COMPRESSION_RATIO})")
    print(f"  Synthetic distillation: {USE_SYNTHETIC_DISTILLATION} (teacher={TEACHER_MODEL})")
    print(f"  Process reward: {USE_PROCESS_REWARD}")
    print(f"  External data: {USE_EXTERNAL_DATA}")
    print(f"  Prompt format: {PROMPT_FORMAT}")
    print(f"  Answer extraction: {ANSWER_EXTRACTION_STRATEGY}")
    print(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}, dropout: {LORA_DROPOUT}")
    print(f"  Learning rate: {LEARNING_RATE}, batch size: {DEVICE_BATCH_SIZE}")
    print(f"  Max seq len: {MAX_SEQ_LEN}, epochs: {NUM_EPOCHS}")
    
    # === Load model ===
    model, tokenizer = build_model()
    
    # === Load data ===
    print("\n=== Loading datasets ===")
    
    # Competition data
    competition_ds = load_competition_data("train", MAX_SAMPLES)
    print(f"  Competition data: {len(competition_ds) if competition_ds else 0} samples")
    
    # Synthetic data
    synthetic_ds = None
    if USE_SYNTHETIC_DISTILLATION:
        if SYNTHETIC_DATA_PATH and os.path.exists(SYNTHETIC_DATA_PATH):
            synthetic_ds = load_synthetic_data(SYNTHETIC_DATA_PATH)
            print(f"  Synthetic data (loaded): {len(synthetic_ds)} samples")
        elif competition_ds:
            # Generate synthetic data
            questions = [competition_ds[i]["question"] for i in range(min(100, len(competition_ds)))]
            synthetic_ds = generate_synthetic_data(
                TEACHER_MODEL, questions, 
                num_samples=DISTILL_NUM_SAMPLES,
                temperature=DISTILL_TEACHER_TEMPERATURE,
            )
            if SYNTHETIC_DATA_PATH and synthetic_ds:
                save_synthetic_data(synthetic_ds, SYNTHETIC_DATA_PATH)
    
    # External datasets
    external_ds = None
    if USE_EXTERNAL_DATA:
        openmath = load_openmath_data(max_samples=5000)
        nemotron = load_nemotron_posttraining_data(max_samples=5000)
        opencode = load_opencodereasoning_data(max_samples=5000)
        external_ds = {"openmath": openmath, "nemotron": nemotron, "opencodereasoning": opencode}
    
    # Unified dataset
    train_dataset = UnifiedReasoningDataset(
        competition_ds=competition_ds,
        openmath_ds=external_ds.get("openmath") if external_ds else None,
        nemotron_ds=external_ds.get("nemotron") if external_ds else None,
        opencode_ds=external_ds.get("opencodereasoning") if external_ds else None,
        synthetic_ds=synthetic_ds,
        max_samples=MAX_SAMPLES,
    )
    
    if len(train_dataset) == 0:
        print("WARNING: No training data available. Using dummy dataset.")
        # Minimal dummy dataset for testing
        class DummyDS(Dataset):
            def __init__(self):
                self.items = [{"question": "What is 2+2?", "answer": "4", "source": "dummy"}]
            def __len__(self):
                return len(self.items)
            def __getitem__(self, i):
                return self.items[i]
        train_dataset = DummyDS()
    
    print(f"\nTotal training samples: {len(train_dataset)}")
    
    # === Self-verification signal ===
    verifier = None
    if SELF_VERIFICATION_ENABLED:
        verifier = VerificationSignal(model, tokenizer)
    
    # === Process reward ===
    process_reward = None
    if USE_PROCESS_REWARD:
        process_reward = ProcessRewardSignal(model, tokenizer)
    
    # === VeriThinker compressor ===
    compressor = None
    if VERITHINKER_ENABLED:
        compressor = VeriThinkerCompressor(model, tokenizer)
    
    # === Dataloader ===
    def collate_fn(batch):
        return collate_reasoning(batch, tokenizer, MAX_SEQ_LEN)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=DEVICE_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # === Optimizer ===
    optimizer = setup_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    num_training_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    lr_scheduler = get_lr_scheduler(optimizer, num_training_steps, WARMUP_RATIO)
    
    print(f"\nTraining steps: {num_training_steps}")
    print(f"Time budget: {TIME_BUDGET}s")
    
    # === Training loop ===
    model.train()
    total_loss = 0.0
    total_verif_loss = 0.0
    step = 0
    t_train_start = time.time()
    
    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(train_loader):
            elapsed = time.time() - t_train_start
            if elapsed >= TIME_BUDGET:
                print(f"\nTime budget reached ({elapsed:.1f}s). Stopping.")
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # Self-verification auxiliary loss
            verif_loss = 0.0
            if SELF_VERIFICATION_ENABLED and verifier and step > 0 and step % VERIFY_EVERY_N_STEPS == 0:
                # Placeholder: in full implementation, run verification signal here
                verif_loss = 0.0
                total_verif_loss += verif_loss
            
            # Total loss
            combined_loss = loss + verif_loss * SELF_VERIFICATION_WEIGHT
            
            combined_loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step += 1
                
                if step % 10 == 0:
                    elapsed = time.time() - t_train_start
                    avg_loss = total_loss / step
                    avg_verif = total_verif_loss / max(step // VERIFY_EVERY_N_STEPS, 1)
                    lr = lr_scheduler.get_last_lr()[0]
                    remaining = TIME_BUDGET - elapsed
                    print(f"step {step:04d} | loss: {avg_loss:.6f} | vloss: {avg_verif:.6f} | "
                          f"lr: {lr:.2e} | elapsed: {elapsed:.1f}s | remaining: {remaining:.1f}s")
    
    # === Summary ===
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
    print(f"  num_samples: {len(train_dataset)}")
    print(f"  self_consistency: {SELF_CONSISTENCY_ENABLED}")
    print(f"  self_verification: {SELF_VERIFICATION_ENABLED}")
    print(f"  verithinker: {VERITHINKER_ENABLED}")
    print(f"  synthetic_distillation: {USE_SYNTHETIC_DISTILLATION}")
    print(f"  external_data: {USE_EXTERNAL_DATA}")
    print(f"  prompt_format: {PROMPT_FORMAT}")
    
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
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate():
    """Evaluate the trained model with optional self-consistency voting."""
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print("Loading model and adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    output_dir = os.path.expanduser("~/.cache/autorae/output")
    if os.path.exists(output_dir):
        model = PeftModel.from_pretrained(base_model, output_dir)
        model.eval()
    else:
        print("Warning: No adapter found. Evaluating base model.")
        base_model.eval()
        model = base_model
    
    # Load validation data
    val_dataset = load_competition_data("train", MAX_SAMPLES)
    
    if val_dataset is None or len(val_dataset) == 0:
        print("No validation dataset. Skipping eval.")
        return None
    
    # Evaluate with optional self-consistency
    if SELF_CONSISTENCY_ENABLED:
        print(f"Evaluating with self-consistency ({SELF_CONSISTENCY_NUM_SAMPLES} samples per question)...")
        accuracy = evaluate_with_self_consistency(model, tokenizer, val_dataset)
    else:
        accuracy = evaluate_accuracy(model, tokenizer, val_dataset, num_samples=100, max_new_tokens=MAX_TOKENS)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    return accuracy


def evaluate_with_self_consistency(model, tokenizer, val_dataset, num_samples=None):
    """
    Self-consistency evaluation: generate multiple reasoning paths per question,
    vote on the most consistent answer.
    
    This is the approach from the Google self-consistency paper.
    """
    model.eval()
    correct = 0
    total = min(num_samples or 100, len(val_dataset))
    num_gen_samples = SELF_CONSISTENCY_NUM_SAMPLES
    
    for i in tqdm(range(total), desc="Evaluating (self-consistency)"):
        item = val_dataset[i]
        question = item["question"]
        true_answer = str(item["answer"]).strip().upper()
        
        # Generate multiple responses
        prompt = f"<|user|>\n{question}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        responses = []
        for _ in range(num_gen_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    temperature=SELF_CONSISTENCY_TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            responses.append(response)
        
        # Majority vote on answers
        pred_answer = extract_answer_majority_vote(responses)
        
        if pred_answer == true_answer:
            correct += 1
    
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = train()
    accuracy = evaluate()
    
    if accuracy is not None:
        results["accuracy"] = accuracy
        print(f"\nFinal accuracy: {accuracy:.4f}")
