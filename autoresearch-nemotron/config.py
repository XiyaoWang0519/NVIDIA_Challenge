"""
Configuration for NVIDIA Nemotron Model Reasoning Challenge.
Adapts karpathy/autoresearch for LoRA fine-tuning of Nemotron-3-Nano-30B.
"""

# ============================================================================
# Model Configuration
# ============================================================================

# Base model - Nemotron-3-Nano-30B for competition
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
# Alternative: FP8 version for lower memory
# BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

# LoRA configuration (competition requires rank <= 32)
LORA_RANK = 32
LORA_ALPHA = 64  # Typically 2x rank
LORA_DROPOUT = 0.05

# Target modules for LoRA (standard for Nemotron/Mistral architectures)
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ============================================================================
# Training Configuration
# ============================================================================

# Sequence length (competition max_tokens = 7680)
MAX_SEQ_LEN = 7680

# Batch size (adjust based on GPU memory)
DEVICE_BATCH_SIZE = 1  # Per-device batch size
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 8

# Time budget per experiment in seconds (5 minutes default)
TIME_BUDGET = 300

# Learning rates
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LR_SCHEDULE = "cosine"

# ============================================================================
# GRPO / RL Configuration
# ============================================================================

# GRPO: Group Relative Policy Optimization
GRPO_GROUPS = 8  # Number of samples per group
GRPO_BETA = 0.1  # KL penalty coefficient
GRPO_GAMMA = 1.0  # Reward advantage scaling

# Use outcome reward (1 for correct, 0 for incorrect)
# Can be extended to process reward (per-step)
USE_OUTCOME_REWARD = True
USE_PROCESS_REWARD = False

# ============================================================================
# Data Configuration
# ============================================================================

# Reasoning datasets for fine-tuning
# See prepare.py for data preparation
DATASETS = [
    "openmathreasoning",      # Math reasoning
    "lightweight-sft-reasoning",  # General reasoning
]

# Alternative datasets to consider:
# - opencodereasoning
# - tulu-sft-reasoning
# - scibench
# - gpqa

# Data format: JSONL with fields: prompt, response, answer
DATA_PATH = "./data/reasoning.jsonl"

# Validation split ratio
VAL_RATIO = 0.05

# ============================================================================
# Prompt Templates
# ============================================================================

# Chain-of-Thought prompt template
COT_PROMPT_TEMPLATE = """Solve the following problem step by step. Show your reasoning in your response, and put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:
"""

# Wrapper prompt (competition requirement)
WRAPPER_PROMPT = """You are a helpful assistant. Solve the problem below.\n\n{problem}\n\nProvide your answer in \\boxed{{}} format."""

# ============================================================================
# Evaluation Configuration
# ============================================================================

# Competition evaluation settings
EVAL_MAX_TOKENS = 7680
EVAL_TEMPERATURE = 0.0  # Deterministic
EVAL_TOP_P = 1.0
EVAL_TOP_K = -1

# Metric: Accuracy (exact match or within tolerance 1e-2)
# See: https://www.kaggle.com/code/metric/nvidia-nemotron-metric

# ============================================================================
# Experiment Tracking
# ============================================================================

# Results file
RESULTS_FILE = "results.tsv"

# Experiment tag (set per run)
EXP_TAG = "nemotron-reasoning"

# ============================================================================
# Hardware / Memory
# ============================================================================

# Memory optimization
USE_FLASH_ATTENTION = True
USE_GRADIENT_CHECKPOINTING = True
BF16_MIXED_PRECISION = True

# Gradient checkpointing saves ~40% memory at cost of ~20% slower
GRADIENT_CHECKPOINTING_RATIO = 0.4

# ============================================================================
# HuggingFace Hub (optional)
# ============================================================================

# Push to hub after training (optional)
PUSH_TO_HUB = False
HUB_REPO = None  # e.g., "your-username/nemotron-reasoning"

# ============================================================================
# Logging
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FILE = "train.log"
SAVE_CHECKPOINTS = True
CHECKPOINT_DIR = "./checkpoints"
