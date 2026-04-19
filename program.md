# Autonomous Research: NVIDIA Nemotron Reasoning Challenge

This is an autonomous research experiment. AI agents modify `train.py` to improve reasoning accuracy on the NVIDIA Nemotron model challenge.

## Goal

Fine-tune a LoRA adapter for `Nemotron-3-Nano-30B` that maximizes **reasoning accuracy** on logical reasoning puzzles. The competition evaluates on a hidden test set; we use our own validation split as the proxy metric.

## Competition constraints (do not modify these in train.py)

| Parameter | Value |
|---|---|
| LoRA rank max | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 |
| max_model_len | 8192 |
| gpu_memory_utilization | 0.85 |
| target_modules | `.*\.(in_proj\|out_proj\|up_proj\|down_proj)$` |

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr18`). The branch `autorae/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autorae/<tag>` from current main.
3. **Read the in-scope files**:
   - `README.md` (this file) — context
   - `prepare.py` — fixed data prep and evaluation. **Do not modify.**
   - `train.py` — the file you modify. LoRA fine-tuning, hyperparameters, data formatting.
4. **Verify data exists**: Check that `~/.cache/autorae/competition_data/` contains competition CSV files. If not, run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

## What you CAN do (modify in train.py)

- Model hyperparameters: LoRA rank (1-32), alpha, dropout, target modules
- Training hyperparameters: learning rate, batch size, sequence length, warmup, weight decay, optimizer choice
- Data formatting: prompt templates, use of chain-of-thought, synthetic data augmentation
- Training techniques: curriculum learning, data filtering, answer extraction strategies
- Any technique from the referenced papers: self-consistency distillation, self-verification signals, VeriThinker-style compression
- **You may use external data/models** as allowed by competition rules (e.g., generate synthetic reasoning data with larger models)

## What you CANNOT do

- **Do not modify `prepare.py`** — it is read-only. The `evaluate_accuracy` function is the ground truth metric.
- **Do not change competition constraint constants** (LoRA rank max=32, temperature=0, etc.) in your final submission
- **Do not install new packages** — only use packages already in `pyproject.toml`
- Do not modify the evaluation harness

## The experiment loop

The experiment runs on branch `autorae/<tag>`.

**LOOP FOREVER:**

1. Look at the git state: current branch/commit
2. Modify `train.py` with an experimental idea
3. `git commit`
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^accuracy:" run.log` or extract from the evaluation section
6. If accuracy improved: keep the commit
7. If accuracy is equal or worse: `git reset` back to the previous state
8. Log results in `results.tsv`

### Output format

```
commit	accuracy	memory_gb	status	description
```

Example:
```
commit	accuracy	memory_gb	status	description
a1b2c3d	0.7200	45.2	keep	baseline
b2c3d4e	0.7450	45.8	keep	increase LR to 5e-4
c3d4e5f	0.7300	44.9	discard	add synthetic CoT data (overfit)
d4e5f6g	0.0000	0.0	crash	batch size too large (OOM)
```

**Note**: `results.tsv` should NOT be committed — leave it untracked by git.

### Timeout

Each experiment should complete training + eval within 20 minutes total. If it exceeds 30 minutes, treat as a failure (crash).

### Crash handling

If a run crashes (OOM, bug, etc.):
- If it's an easy fix (typo, missing import): fix it and re-run
- If the idea itself is broken: skip it, log "crash", move on
- Never spend more than 3 attempts on the same idea

## Key papers to reference (for ideas)

1. **Self-consistency** (Google): sample diverse reasoning paths, pick most consistent answer
2. **Self-verification** (新加坡国立): validate own answers via backward verification
3. **VeriThinker** (arXiv 2505.17941): train verifiers to compress reasoning chains
4. **NVIDIA AIMO writeup**: synthetic data distillation from larger models (DeepSeek-R1, QwQ-32B)
5. **Numina AIMO writeup**: good data is all you need; TIR with code execution feedback

## Synthetic data generation ideas

Since the competition allows external data:
- Generate reasoning chains using larger models (DeepSeek-R1, QwQ-32B, etc.)
- Use process reward models to score reasoning steps
- Filter/curate training data based on whether the model can verify its own answers
- Experiment with different reasoning formats (step-by-step, tree-of-thought, etc.)

## Simplicity criterion

All else being equal, simpler is better. A small accuracy gain that adds significant complexity is less valuable than expected. Conversely, removing something that improves or maintains accuracy is a simplification win.

## Quick start

```bash
cd autorae

# First time: install dependencies
uv sync

# First time: download competition data
uv run python prepare.py

# Run a single experiment (baseline first!)
uv run train.py

# After training, evaluation runs automatically
# Check results with:
grep "accuracy:" run.log
grep "val_bpb equivalent" run.log  # if using proxy metric
```
