# Autorae — Autonomous Research: NVIDIA Nemotron Reasoning Challenge

This is an autonomous research experiment. AI agents modify `train.py` to improve reasoning accuracy on the NVIDIA Nemotron model challenge.

## Goal

Fine-tune a LoRA adapter for `Nemotron-3-Nano-30B` that maximizes **reasoning accuracy** on logical reasoning puzzles. The competition evaluates on a hidden test set; we use our own validation split as the proxy metric.

## Competition constraints (do not change these in train.py)

| Parameter | Value |
|---|---|
| LoRA rank max | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 (for final evaluation) |
| max_model_len | 8192 |
| gpu_memory_utilization | 0.85 |
| target_modules | `.*\.(in_proj|out_proj|up_proj|down_proj)$` |

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr18`). The branch `autorae/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autorae/<tag>` from current main.
3. **Read the in-scope files**:
   - `README.md` (this file) — context
   - `prepare.py` — fixed data prep and evaluation. **Do not modify.**
   - `train.py` — the file you modify. Everything below is fair game.
4. **Verify data exists**: Check that `~/.cache/autorae/competition_data/` contains competition CSV files. If not, run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**.

## What you CAN do (modify in train.py)

### LoRA & Model Config
- LoRA rank (1-32), alpha, dropout, target modules
- Any other model configuration

### Training Hyperparameters
- Learning rate, batch size, sequence length, warmup, weight decay
- Optimizer choice (AdamW, Muon, etc.)
- Learning rate scheduler (linear, cosine, warmup+warmdown)
- Gradient accumulation steps

### Data & Prompting
- **Prompt formats**: CoT, ToT, concise, detailed, or your own design
- **External datasets** (competition allows these):
  - OpenMathReasoning (NVIDIA-referenced)
  - OpenCodeReasoning (NVIDIA-referenced)
  - Llama-Nemotron-Post-Training-Dataset (NVIDIA-referenced)
- **Synthetic data distillation** from larger models (DeepSeek-R1, QwQ-32B, etc.)
- Data filtering and curation strategies
- Answer extraction strategies (last_line, marker, majority_vote, verifier_based)

### Advanced Reasoning Methods

**Self-consistency** (Google paper):
- Generate N reasoning paths per question with temperature > 0
- Vote on the most consistent answer
- Can be used at training time (distillation) or eval time
- Toggle: `SELF_CONSISTENCY_ENABLED = True`

**Self-verification** (NUS/OpenReview paper):
- Train the model to verify its own answers by re-reading the question
- Creates an auxiliary training loss that rewards self-consistency
- Toggle: `SELF_VERIFICATION_ENABLED = True`

**VeriThinker** (reasoning compression):
- Train a verifier to identify correct reasoning early
- Compress reasoning chains to reduce token waste
- Paper claims 44% token reduction + 0.8% accuracy improvement on MATH500
- Toggle: `VERITHINKER_ENABLED = True`

**Process Reward Modeling**:
- Per-step rewards instead of just final answer reward
- Enables tree-of-thought style search with step-level scoring
- Toggle: `USE_PROCESS_REWARD = True`

### Training Techniques
- Curriculum learning (easy → hard)
- Mixture of experts routing
- Differential learning rates for different layers
- Knowledge distillation from larger models
- Multi-task learning (reasoning + other objectives)

### Evaluation
- Self-consistency evaluation (majority vote at test time)
- Verification-based reranking
- Different extraction strategies for final answer

## What you CANNOT do

- **Do not modify `prepare.py`** — it is read-only. The `evaluate_accuracy` function is the ground truth metric.
- **Do not change competition constraint constants** in final submission (LoRA rank max=32, temperature=0, etc.)
- **Do not install new packages** — only use packages in `pyproject.toml`
- Do not modify the fixed evaluation harness

## The experiment loop

The experiment runs on branch `autorae/<tag>`.

**LOOP FOREVER:**

1. Look at git state: current branch/commit
2. Modify `train.py` with an experimental idea
3. `git commit`
4. Run experiment: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^accuracy:" run.log`
6. If accuracy improved: keep the commit
7. If accuracy is equal or worse: `git reset` back to previous state
8. Log results in `results.tsv`

### Output format

```
commit	accuracy	memory_gb	status	description
```

Example:
```
commit	accuracy	memory_gb	status	description
a1b2c3d	0.7200	45.2	keep	baseline
b2c3d4e	0.7450	45.8	keep	increase LR to 5e-4 + add CoT prompting
c3d4e5f	0.7300	44.9	discard	add synthetic data (overfit)
d4e5f6g	0.0000	0.0	crash	batch size too large (OOM)
e5f6g7h	0.7600	46.1	keep	+ self-consistency distillation
f6g7h8i	0.7700	46.3	keep	+ external OpenMath data
```

**Note**: `results.tsv` should NOT be committed — leave it untracked.

### Timeout

Each experiment should complete within 20 minutes total. If it exceeds 30 minutes, treat as crash.

## Key papers for ideas

1. **Self-consistency** — https://research.google/pubs/self-consistency-improves-chain-of-thought-reasoning-in-language-models/
2. **Self-verification** — https://openreview.net/forum?id=s4xIeYimGQ
3. **VeriThinker** — https://arxiv.org/html/2505.17941v1 (reasoning compression + 0.8% improvement)
4. **NVIDIA AIMO writeup** — https://blogs.nvidia.com/blog/reasoning-ai-math-olympiad/ (distill from DeepSeek-R1 + QwQ-32B)
5. **Numina AIMO writeup** — https://huggingface.co/blog/winning-aimo-progress-prize (good data is all you need)

## Advanced methods quick reference

| Method | Toggle | What it does |
|---|---|---|
| Self-consistency | `SELF_CONSISTENCY_ENABLED` | Multi-sample vote at eval time |
| Self-consistency distillation | same + `SELF_CONSISTENCY_TEMPERATURE > 0` | Distill voting into training |
| Self-verification | `SELF_VERIFICATION_ENABLED` | Backward verification as training signal |
| VeriThinker | `VERITHINKER_ENABLED` | Compress reasoning chains via verifier |
| Process reward | `USE_PROCESS_REWARD` | Per-step rewards for tree search |
| Synthetic distillation | `USE_SYNTHETIC_DISTILLATION` | Generate with DeepSeek-R1/QwQ-32B |
| External data | `USE_EXTERNAL_DATA` | OpenMathReasoning + Nemotron dataset |

## Simplicity criterion

All else being equal, simpler is better. A small accuracy gain that adds significant complexity is less valuable. Removing something that improves accuracy is a simplification win.

## Quick start

```bash
cd autorae

# Install
uv sync

# First run (baseline, no advanced methods)
uv run train.py

# Enable self-consistency + external data
# Edit train.py:
#   SELF_CONSISTENCY_ENABLED = True
#   USE_EXTERNAL_DATA = True
#   uv run train.py
```
