# Research Agent Instructions: NVIDIA Nemotron Model Reasoning Challenge

You are an autonomous AI researcher working on the **NVIDIA Nemotron Model Reasoning Challenge**. Your goal is to improve reasoning accuracy on a novel benchmark developed by NVIDIA Research.

## Competition Context

- **Objective**: Improve reasoning accuracy using NVIDIA Nemotron models
- **Base Model**: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- **Submission**: LoRA adapter (rank ≤ 32) for the Nemotron-3-Nano-30B base model
- **Evaluation**: Accuracy on reasoning tasks (answers in `\boxed{}` LaTeX format)
- **Prizes**: $106,388 + 8 DGX Spark systems

## Key Techniques to Explore

### 1. Prompting Strategies
- Chain-of-Thought (CoT) prompting
- Self-consistency decoding
- Tree-of-Thought prompting
- Best-of-N sampling with verification

### 2. Reinforcement Learning (GRPO/RLVR)
- Group Relative Policy Optimization (GRPO)
- Reinforcement Learning with Verifiable Rewards (RLVR)
- Process Reward Models (PRM)
- Outcome-based rewards

### 3. Data Engineering
- Synthetic data generation for reasoning
- Data filtering and curation
- Math/Code/Science reasoning datasets

### 4. Fine-tuning Approaches
- LoRA rank optimization (≤ 32 as per rules)
- Learning rate scheduling
- Batch size and sequence length tuning

## What You CAN Do

- Modify `train.py` to experiment with:
  - Training methods (SFT, GRPO, RLVR)
  - Hyperparameters (LR, batch size, LoRA rank)
  - Data formatting (prompt templates, CoT patterns)
  - Reward computation (outcome vs process rewards)
  - Sampling strategies (temperature, top-p, best-of-n)

- Modify `config.py` to adjust:
  - Model configuration
  - Training parameters
  - Dataset selection
  - Evaluation settings

- Modify `prepare.py` to:
  - Add new datasets
  - Adjust data formatting
  - Implement data augmentation

## What You CANNOT Do

- Modify base model loading (must use Nemotron-3-Nano-30B)
- Use LoRA rank > 32
- Modify evaluation metric (competition metric is fixed)
- Install new dependencies (use only existing packages)

## Experiment Workflow

### Each experiment:
1. Modify `train.py` or `config.py` with your idea
2. Run training: `python train.py`
3. Observe accuracy on validation set
4. Log results to `results.tsv`
5. Keep or discard based on accuracy improvement

### Key Metrics:
- `val_accuracy`: Validation accuracy (higher is better)
- `training_seconds`: Training time
- `peak_vram_mb`: GPU memory usage
- `num_steps`: Number of training steps

### Time Budget:
Each training run has a **5-minute time budget** (configurable). Aim for:
- Fast iteration (12 experiments/hour)
- Meaningful accuracy gains
- Memory efficiency

## Suggested Experiment Directions

### High Priority:
1. **GRPO with outcome rewards**: Reward correct `\boxed{}` answers
2. **Process rewards**: Reward intermediate reasoning steps
3. **Best-of-N decoding**: Generate multiple responses, select best
4. **Chain-of-Thought formatting**: Optimize CoT prompt templates

### Medium Priority:
5. **Data curation**: Filter for challenging problems
6. **Synthetic data**: Generate more reasoning traces
7. **Learning rate scheduling**: Cosine annealing with warmup
8. **LoRA rank exploration**: Test ranks 8, 16, 24, 32

### Experimental:
9. **Ensemble methods**: Combine multiple LoRA adapters
10. **Reward shaping**: Design better reward functions
11. **Curriculum learning**: Start easy, progress hard
12. **Multi-task learning**: Combine math/code/science

## Data Format

Training data should follow this format:

```json
{
  "prompt": "Problem text with \\boxed{} answer expected",
  "response": "Step-by-step reasoning leading to answer",
  "answer": "Final numerical/text answer"
}
```

The model should be prompted to show reasoning and place final answer in `\boxed{}`.

## Evaluation

The competition metric extracts answers from `\boxed{}` and compares:
- Exact string match
- Numerical tolerance of 1e-2

```
prediction \boxed{42} matches ground truth \boxed{42} → correct
prediction \boxed{41.99} matches ground truth \boxed{42} → correct (within tolerance)
```

## References

- Competition: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- GRPO Paper: https://arxiv.org/abs/2505.07686
- s1 Test-time Scaling: https://arxiv.org/abs/2501.12597
- Chain-of-Thought: https://arxiv.org/abs/2201.11903
- OpenMathReasoning: https://huggingface.co/datasets/nvidia/OpenMathReasoning

## Success Criteria

Your experiments should aim to:
1. Beat the baseline accuracy (> 50% typical for reasoning tasks)
2. Stay within memory limits (LoRA rank ≤ 32)
3. Complete within time budget (5 minutes per run)
4. Produce reproducible improvements

Good luck! 🚀
