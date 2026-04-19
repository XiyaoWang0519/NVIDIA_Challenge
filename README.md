# Autorae — Autonomous Research on Nemotron Reasoning

Autonomous AI agent-driven research on LoRA fine-tuning for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).

This is adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for the competition setting.

## Project structure

```
autorae/
├── prepare.py      # Fixed data prep and evaluation (DO NOT MODIFY)
├── train.py        # LoRA fine-tuning — the file the agent modifies
├── program.md      # Agent instructions
├── pyproject.toml  # Dependencies
└── results.tsv     # Experiment log (untracked, created at runtime)
```

## Core idea

The same as original autoresearch: give an AI agent a training setup, let it experiment autonomously overnight, and wake up to a log of experiments and (hopefully) a better model.

The key difference: we fine-tune **Nemotron-3-Nano-30B with LoRA** for reasoning accuracy, rather than training a small GPT from scratch for bits/byte loss.

## Quick start

```bash
cd autorae

# Install dependencies
uv sync

# Download competition data
uv run python prepare.py

# Run a baseline experiment
uv run train.py
```

## Competition context

- **Model**: `nvidia/Nemotron-3-Nano-30B-A3B-bf16`
- **Task**: Logical reasoning puzzles
- **Submission**: A LoRA adapter (rank <= 32)
- **Metric**: Accuracy on hidden test set
- **Key constraints**: max_tokens=7680, temperature=0.0, top_p=1.0

## References

- [Competition overview](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview)
- [Submission demo](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo)
- [NVIDIA Nemotron developer page](https://developer.nvidia.com/nemotron)
- [Self-consistency paper](https://research.google/pubs/self-consistency-improves-chain-of-thought-reasoning-in-language-models/)
- [Self-verification paper](https://openreview.net/forum?id=s4xIeYimGQ)
- [VeriThinker paper](https://arxiv.org/html/2505.17941v1)
- [NVIDIA AIMO writeup](https://blogs.nvidia.com/blog/reasoning-ai-math-olympiad/)
- [Numina AIMO writeup](https://huggingface.co/blog/winning-aimo-progress-prize)
