# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA Nemotron Model Reasoning Challenge (Kaggle). The goal is to fine-tune a LoRA adapter for `nvidia/Nemotron-3-Nano-30B-A3B-bf16` that maximizes reasoning accuracy on logical puzzles. The project adapts [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for this competition.

## Training Platform: Kaggle Notebooks

Training runs on **Kaggle** (not GCP or RunPod). The competition provides free access to **RTX Pro 6000 Blackwell (96GB VRAM)** — enough to train bf16 LoRA without quantization or CPU offloading.

See `KAGGLE_SETUP.md` for notebook setup, submission flow, and automation.

## Competition Constraints (immutable)

| Parameter | Value |
|---|---|
| LoRA rank max | 32 |
| max_tokens | 7680 |
| temperature | 0.0 |
| top_p | 1.0 |
| max_model_len | 8192 |
| gpu_memory_utilization | 0.85 |
| target_modules | `.*\.(in_proj\|out_proj\|up_proj\|down_proj)$` |
| Max submissions/day | 5 |

External data IS allowed. Team size max 5.

## Architecture

### Core files (in `autorae/`)

- **`prepare.py`** — FIXED data prep and evaluation harness. **Do not modify.** Exports constants (`MODEL_NAME`, `LORA_RANK`, `evaluate_accuracy`, `extract_answer`, etc.) and datasets.
- **`train.py`** — The file the agent modifies. Imports constraints from `prepare.py`. Contains hyperparameter toggles, prompt formatters, LoRA training loop, and evaluation.
- **`program.md`** — Agent instructions for the autonomous experiment loop (what to try, what's off-limits).
- **`pyproject.toml`** — Dependencies managed with `uv`. No new packages should be installed.

### Key constraint: `prepare.py` is read-only

Everything in `prepare.py` is fixed — the evaluation metric, competition constants, and data loading. The agent only modifies `train.py`.

## Model Architecture Notes

Nemotron-3-Nano-30B-A3B is a **hybrid Mamba-MoE model** (NemotronH architecture):
- 52 layers alternating Mamba (M) and Expert/MoE (E) layers
- 128 routed experts per MoE layer, plus shared experts
- hidden_size: 2688, num_attention_heads: 32
- **NO** separate q_proj/k_proj/v_proj/o_proj — only `in_proj`/`out_proj` (Mamba) and `up_proj`/`down_proj` (MoE)
- The competition's target regex `.*\.(in_proj|out_proj|up_proj|down_proj)$` is correct for this architecture
- ~880M trainable params with LoRA rank 32 (2.7% of 32.5B total)

### Critical technical details

1. **`trust_remote_code=True`** is required for this model (custom `modeling_nemotron_h.py`)
2. **bitsandbytes quantization (4-bit and 8-bit) is broken** for Mamba SSM layers — use bf16 only
3. **The model has a KV cache bug** in the remote code — install `transformers >= 5.3.0` and remove `trust_remote_code=True` during training to use the native implementation
4. **Gradient checkpointing** is incompatible with NemotronH's native transformers implementation
5. **Evaluation uses vLLM** with `enable_lora=True`, not HuggingFace `generate()`
6. **Data columns**: CSV has `prompt`/`answer`, not `question`/`answer`

### Competition evaluation pipeline

1. Your `submission.zip` is extracted to find `adapter_config.json`
2. vLLM loads base model + your LoRA adapter on RTX Pro 6000 (96GB)
3. Each test prompt gets appended with `\nPlease put your final answer inside \boxed{}`
4. Tokenizer's `apply_chat_template` called with `enable_thinking=True`
5. Answers extracted from `\boxed{...}` — matched via exact string or numeric (rel_tol=1e-2)

## Winning Approach Reference

The Open Progress Prize winner (score 0.85) used **pure SFT with programmatic CoT traces**:
- Crafted deterministic solution code for each puzzle category (numeral, cipher, gravity, etc.)
- No RL needed — "the only job of the LLM is to follow the policy"
- Key categories: numeral/unit_conversion/gravity/cipher (100% solvable), cryptarithm (~8%)

## Research Cache

The `.firecrawl/` directory contains scraped competition pages, discussions, and reference papers. `NVIDIA_NEMOTRON_REASONING_WIKI.md` is a consolidated wiki.
