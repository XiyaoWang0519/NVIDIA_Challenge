# NVIDIA Nemotron Model Reasoning Challenge Wiki

Last refreshed: April 18, 2026

Raw source cache for this note lives in `.firecrawl/`.

## Official competition pages

- Overview: [Kaggle competition overview](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview)
- Data: [Kaggle data page](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data)
- Rules: [Kaggle rules](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules)
- Metric/demo: [NVIDIA Nemotron submission demo](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo)

> "Develop techniques that improve reasoning accuracy using NVIDIA Nemotron models."

> "Participants will experiment with prompting, data pipelines, and lightweight fine-tuning while evaluating their approaches on a new reasoning benchmark developed by NVIDIA Research."

> "The only requirement is that the final submission produces a compatible LoRA adapter for the Nemotron-3-Nano-30B base model."

> "Submissions are evaluated based on their Accuracy in solving the provided tasks."

### Fixed metric/runtime constraints copied from Kaggle

From the official overview:

| Parameter | Value |
| --- | --- |
| `max_lora_rank` | `32` |
| `max_tokens` | `7680` |
| `top_p` | `1.0` |
| `temperature` | `0.0` |
| `max_num_seqs` | `64` |
| `gpu_memory_utilization` | `0.85` |
| `max_model_len` | `8192` |

From the official data page:

> "This dataset comprises a collection of logical reasoning puzzles requiring the identification and application of underlying transformation rules."

> "`test.csv` ... will be replaced by a test set of several hundred problems."

From the official rules:

> "The maximum Team size is five (5)."

> "You may submit a maximum of five (5) Submissions per day."

> "You may select up to two (2) Final Submissions for judging."

> "You may use data other than the Competition Data ('External Data') to develop and test your Submissions."

> "To be eligible for any prize, teams must publish a public Kaggle notebook and solution Writeup documenting the methods used to generate their submission."

### Submission demo lines worth copying

From [NVIDIA Nemotron Submission Demo](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo):

```python
LORA_RANK = 32  # Can be set to a maximum of 32
...
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
...
model.save_pretrained(OUTPUT_DIR)
```

## Official NVIDIA / Nemotron sources

- Hub page: [NVIDIA Nemotron developer page](https://developer.nvidia.com/nemotron)
- Blog: [Advancing Agentic AI with NVIDIA Nemotron Open Reasoning Models](https://developer.nvidia.com/blog/advancing-agentic-ai-with-nvidia-nemotron-open-reasoning-models/)
- Paper: [Llama-Nemotron: Efficient Reasoning Models](https://arxiv.org/abs/2505.00949)
- Tutorials/repo entrypoint: [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron)

From the developer page:

> "NVIDIA Nemotron is a family of open models with open weights, training data, and recipes, delivering leading efficiency and accuracy for building specialized AI agents."

> "The new Nemotron 3 family provides the most efficient open models, powered by hybrid Mamba-Transformer MoE with 1M-token context, delivering top accuracy for complex, high-throughput agentic AI applications."

> "Nemotron 3 Nano offers 4x faster throughput compared to Nemotron 2 Nano."

> "Leading accuracy for coding, reasoning, math and long context tasks."

From the NVIDIA technical blog:

> "To create a Nemotron model, the team starts with an open frontier model and performs a series of key steps..."

> "Knowledge Distillation ... transfer[s] reasoning skills from large models to smaller, faster ones."

> "Supervised Fine-Tuning ... trains models with a mix of reasoning and nonreasoning data."

> "Reinforcement Learning ... reward[s] accurate, structured outputs."

Useful dataset links called out by NVIDIA in that post:

- [OpenMathReasoning](https://huggingface.co/collections/nvidia/openmathreasoning-68072c0154a5099573d2e730)
- [OpenCodeReasoning](https://huggingface.co/collections/nvidia/opencodereasoning-67ec462892673a326c0696c1)
- [Llama-Nemotron post-training dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset)

From the Llama-Nemotron paper abstract:

> "We introduce the Llama-Nemotron series of models, an open family of heterogeneous reasoning models that deliver exceptional reasoning capabilities, inference efficiency, and an open license for enterprise use."

> "The training procedure ... entails ... knowledge distillation ... followed by a reasoning-focused post-training stage consisting of two main parts: supervised fine-tuning and large scale reinforcement learning."

> "Llama-Nemotron models are the first open-source models to support a dynamic reasoning toggle..."

> "We release the complete post-training dataset ... [and] our training codebases: NeMo, NeMo-Aligner, and Megatron-LM."

## Reasoning methods worth stealing ideas from

### Self-consistency

Source: [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://research.google/pubs/self-consistency-improves-chain-of-thought-reasoning-in-language-models/)

> "We propose a new decoding strategy, self-consistency, to replace the naive greedy decoding used in chain-of-thought prompting."

> "It first samples a diverse set of reasoning paths ... and then selects the most consistent answer by marginalizing out the sampled reasoning paths."

### Self-verification

Source: [Large Language Models are Better Reasoners with Self-Verification](https://openreview.net/forum?id=s4xIeYimGQ)

> "LLMs also have similar self-verification abilities."

> "By performing a backward verification of the answers that LLM deduced for itself, we can obtain interpretable answer validation scores to select the candidate answer with the highest score."

Public code link from the paper page:

- [WENGSYX/Self-Verification](https://github.com/WENGSYX/Self-Verification)

### Verification as compression / anti-overthinking

Source: [VeriThinker: Learning to Verify Makes Reasoning Model Efficient](https://arxiv.org/html/2505.17941v1)

> "Large Reasoning Models ... tend[ ] to overthinking ... dramatically increasing inference costs."

> "We introduce VeriThinker, a novel approach for CoT compression."

> "By training LRMs to accurately verify the correctness of CoT solutions, the LRMs inherently become more discerning about the necessity of subsequent self-reflection steps."

> "Our approach reduces reasoning tokens on MATH500 from 3790 to 2125 while improving accuracy by 0.8%."

Code:

- [czg1225/VeriThinker](https://github.com/czg1225/VeriThinker)

## Repos and tooling

### vLLM

Source: [vllm-project/vllm](https://github.com/vllm-project/vllm)

> "vLLM is a fast and easy-to-use library for LLM inference and serving."

> "State-of-the-art serving throughput."

> "High-throughput serving with various decoding algorithms..."

> "Multi-lora support."

### Competition demo / PEFT path

Source: [NVIDIA Nemotron Submission Demo](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo)

This is the most concrete "starting notebook" source in the whole stack:

- it loads `metric/nemotron-3-nano-30b-a3b-bf16`
- it applies PEFT LoRA directly to the base model
- it saves only the adapter payload expected by the competition packaging flow

### Additional repo links to inspect

- [OpenPipe/vllm-lora](https://github.com/OpenPipe/vllm-lora)
- [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron)
- [janhq/verifiers-deepresearch](https://github.com/janhq/verifiers-deepresearch)

## Adjacent competition writeups that look transferable

These are not the same submission environment, but they contain tactics that map onto "reasoning under hard inference limits."

### NVIDIA’s AIMO writeup

Source: [NVIDIA Team Scores Kaggle Win With Reasoning Model](https://blogs.nvidia.com/blog/reasoning-ai-math-olympiad/)

> "Their winning model used Qwen2.5-14B-Base ... fine-tuned on millions of synthetically generated solutions to math problems."

> "These synthetic solutions were primarily generated by two larger reasoning models — DeepSeek-R1 and QwQ-32B — and used to teach the team’s foundation model via a form of knowledge distillation."

> "To further boost performance, the team’s solution reasons through multiple long-thinking responses in parallel before determining a final answer."

> "TensorRT-LLM also enabled the team to harness FP8 quantization ... [for] a 1.5x speedup."

### Numina / Hugging Face AIMO writeup

Source: [How NuminaMath Won the 1st AIMO Progress Prize](https://huggingface.co/blog/winning-aimo-progress-prize)

> "Our solution ... consisted of three main components."

> "A recipe to fine-tune DeepSeekMath-Base 7B to act as a 'reasoning agent'..."

> "A novel decoding algorithm for tool-integrated reasoning (TIR) with code execution feedback..."

> "A variety of internal validation sets that we used to guide model selection and avoid overfitting to the public leaderboard."

From the same writeup:

> "Good data is all you need."

> "We found that using MMOS improved performance but was still capped..."

> "At this point, we focused our efforts on producing a dataset similar to the one used by the DeepSeekMath Instruct / RL models..."

### Kaggle LLM Science 1st-place writeup

Source: [1st Place Solution — Kaggle LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam/writeups/team-h2o-llm-studio-1st-place-solution)

> "We were mostly focusing on improving the context retrieval part, while keeping the modeling part almost the same."

> "In the end, we had a local choice of about 300 combinations of Retrieval + LLM models + different Wikipedia dumps."

> "Our final ensemble is a combination of fine-tuned 7B & 13B LLMs."

This is less directly portable than the math/reasoning sources above, but still useful as a reminder to keep validation, ablations, and infrastructure broad while the final modeling change stays narrow.

## Minimal takeaways for this specific challenge

This is the only part of the note that is more synthesis than copying.

- The Kaggle evaluation path is fixed around a single submitted LoRA adapter, loaded into Nemotron-3-Nano-30B through vLLM with `temperature = 0.0`. Ideas like self-consistency and verifier reranking therefore look more useful as training-data generation, distillation, or behavior-shaping methods than as final test-time orchestration.
- NVIDIA is explicitly signaling that prompting, data curation, synthetic data generation, RL, and lightweight fine-tuning are all in-bounds, while the rules also allow external data/models if they are reasonably accessible.
- Prize eligibility is documentation-heavy. The public notebook and writeup are not optional if you care about prizes.
- The shortest "must-read" stack is: competition overview, rules, data page, submission demo, Nemotron developer page, Llama-Nemotron paper, self-consistency paper, self-verification paper, VeriThinker paper, AIMO/Numina writeups.

## Quick reading order

1. [Competition overview](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview)
2. [Competition rules](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules)
3. [Competition data page](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data)
4. [Submission demo](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo)
5. [NVIDIA Nemotron developer page](https://developer.nvidia.com/nemotron)
6. [Llama-Nemotron paper](https://arxiv.org/abs/2505.00949)
7. [Self-consistency paper](https://research.google/pubs/self-consistency-improves-chain-of-thought-reasoning-in-language-models/)
8. [Self-verification paper](https://openreview.net/forum?id=s4xIeYimGQ)
9. [VeriThinker paper](https://arxiv.org/html/2505.17941v1)
10. [NVIDIA AIMO writeup](https://blogs.nvidia.com/blog/reasoning-ai-math-olympiad/)
11. [Numina AIMO writeup](https://huggingface.co/blog/winning-aimo-progress-prize)
