menu

[Skip to\\
\\
content](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge#site-content)

[![Kaggle](https://www.kaggle.com/static/images/site-logo.svg)](https://www.kaggle.com/)

Create

search​

- [explore\\
\\
Home](https://www.kaggle.com/)

- [emoji\_events\\
\\
Competitions](https://www.kaggle.com/competitions)

- [table\_chart\\
\\
Datasets](https://www.kaggle.com/datasets)

- [tenancy\\
\\
Models](https://www.kaggle.com/models)

- [leaderboard\\
\\
Benchmarks](https://www.kaggle.com/benchmarks)

- [smart\_toy\\
\\
Game Arena](https://www.kaggle.com/game-arena)

- [code\\
\\
Code](https://www.kaggle.com/code)

- [comment\\
\\
Discussions](https://www.kaggle.com/discussions)

- [school\\
\\
Learn](https://www.kaggle.com/learn)


- [expand\_more\\
\\
More](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge#)


menu

[Skip to\\
\\
content](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge#site-content)

[![Kaggle](https://www.kaggle.com/static/images/site-logo.svg)](https://www.kaggle.com/)

search​

[Sign In](https://www.kaggle.com/account/login?phase=startSignInTab&returnUrl=%2Fcompetitions%2Fnvidia-nemotron-model-reasoning-challenge)

[Register](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2Fcompetitions%2Fnvidia-nemotron-model-reasoning-challenge)

[NVIDIA's profile](https://www.kaggle.com/organizations/nvidia) NVIDIA  · Featured Prediction Competition · 2 months to go

Join Competition

more\_horiz

# NVIDIA Nemotron Model Reasoning Challenge

Advance reasoning techniques using NVIDIA Nemotron open models on a novel benchmark

![](https://www.kaggle.com/competitions/129716/images/header)

## NVIDIA Nemotron Model Reasoning Challenge

[Overview](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview) [Data](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data) [Code](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/code) [Models](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/models) [Discussion](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion) [Leaderboard](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/leaderboard) [Rules](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules)

## Overview

Develop techniques that improve reasoning accuracy using NVIDIA Nemotron models.

Participants will experiment with prompting, data pipelines, and lightweight fine-tuning while evaluating their approaches on a new reasoning benchmark developed by NVIDIA Research.

Start

a month ago

Close

2 months to go

Merger & Entry

### Description

link

keyboard\_arrow\_up

Reasoning benchmarks are a useful way to measure progress on structured tasks. When approaches and results are shared openly, the community can compare methods, reproduce improvements, and iterate more effectively.

Today, reasoning improvements are explored across many independent efforts - often using different datasets, prompts, and evaluation setups - making direct comparison difficult. A shared benchmark and common baseline model allow techniques to be tested and compared more consistently.

While language models perform strongly on many tasks, structured reasoning benchmarks remain an active area of research and optimization.

In this competition, participants will work from a shared Nemotron 3 Nano baseline and a novel reasoning benchmark developed by NVIDIA Research. Nemotron provides an open foundation for this challenge, including openly available models, datasets, and training recipes that participants can build on or adapt within their own workflows.

You may experiment with:

- Prompting strategies
- Data filtering and curation
- Synthetic data generation
- Reinforcement learning
- Lightweight fine-tuning
- Or other approaches of your choice

Participants may use any training framework, tooling, or workflow to produce their LoRA adapter. NVIDIA-provided recipes are optional starting points - competitors are free to use other ecosystems and libraries (e.g., Hugging Face, Unsloth, Axolotl, TRL, or similar tooling).

The only requirement is that the final submission produces a compatible LoRA adapter for the Nemotron-3-Nano-30B base model.

Multiple valid solution paths are expected. Clear documentation - including notebooks and write-ups - is encouraged (and required for prize eligibility) to support reproducibility and communal learning.

By bringing models, datasets, and evaluation into an open, shared environment, this challenge creates an opportunity for collaborative iteration - strengthening open reasoning workflows that others can study, reuse, and extend.

### Evaluation

link

keyboard\_arrow\_up

Submissions are evaluated based on their **Accuracy** in solving the provided tasks. The [NVIDIA Nemotron-3-Nano-30B](https://www.kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16) model is loaded with your LoRA adapter (which must include an `adapter_config.json`) using the vLLM inference engine. For each test case, the model is prompted to generate a response and instructed to place its final answer within a `\boxed{}` LaTeX command. The metric extracts the final answer from the generated text, prioritizing content within the boxed format while falling back to other heuristic patterns or the last numeric value found. A prediction is graded as correct if it matches the ground truth either exactly as a string or within a relative numerical tolerance of 10−2. The final score is the proportion of correctly answered questions.

You may view the implementation of the metric here: [**NVIDIA Nemotron Metric**](https://www.kaggle.com/code/metric/nvidia-nemotron-metric). It is being run with the following parameters:

| Parameter | Value |
| --- | --- |
| max\_lora\_rank | 32 |
| max\_tokens | 7680 |
| top\_p | 1.0 |
| temperature | 0.0 |
| max\_num\_seqs | 64 |
| gpu\_memory\_utilization | 0.85 |
| max\_model\_len | 8192 |

## Submitting

You must submit a LoRA adapter of rank at most 32 for the NVIDIA Nemotron-3-Nano-30B model packaged into a `submission.zip` file. You may consider adapting the [**NVIDIA Nemotron Submission Demo**](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo) to produce your submission.

### Timeline

link

keyboard\_arrow\_up

- **March 16, 2026** \- Start Date.
- **April 9, 2026** \- Midpoint Cut-off Date
- **June 8, 2026** \- Entry Deadline. You must accept the competition rules before this date in order to compete.
- **June 8, 2026** \- Team Merger Deadline. This is the last day participants may join or merge teams.
- **June 15, 2026** \- Competition End Date & Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### Prizes

link

keyboard\_arrow\_up

To be eligible for any prize, teams must publish a public Kaggle notebook and solution write-up documenting the methods, datasets, and techniques used to produce the submission. Submissions without qualifying public documentation may be deemed ineligible for prizes.

**Final Leaderboard Prizes**

- 1st Place - $25,000 + 5 DGX Sparks
- 2nd Place - $15,000 + 2 DGX Sparks
- 3rd Place - $5,000 + 1 DGX Sparks

Note: A total of eight (8) NVIDIA DGX Spark systems (Approximate Retail Value: $4,699 per system) will be awarded based on final leaderboard placement. If any team has fewer eligible members than the number of DGX Spark systems allocated for that placement, or if any team member is ineligible to receive hardware due to export, shipping, or regional restrictions, any remaining units will cascade to the next highest-ranked teams on the final leaderboard until all eight (8) DGX Spark systems have been awarded.

_Each eligible participant may receive no more than one (1) DGX Spark, and only officially registered team members are eligible to receive hardware prizes. NVIDIA reserves the right to verify team membership and eligibility prior to awarding hardware prizes._

### **Open Progress Prize (Mid-Competition Milestone)**

Open Progress Prize: $5,000 + 1 DGX Sparks

- Awarded to the team with the highest leaderboard score as of the Midpoint Cut-off Date: **April 9, 2026**.
- Methodology submissions Cut-off Date: **April 16, 2026**.
- Winners will be announced during Cloud NEXT (April 22-24, 2026)

If the top-ranked submission at the cutoff date does not meet these requirements, the prize will be awarded to the next highest eligible submission.

In the event of a tie, the prize will be awarded to the team whose qualifying submission was submitted earliest.

Each eligible participant may receive no more than one (1) DGX Spark.

### **Open Contribution Awards**

The Open Contribution Awards recognize techniques that meaningfully advance reasoning performance using Nemotron models.

Three awards will be granted:

- Best Data/Synthetic Data Method - 1 DGX Spark
- Best RL Method - 1 DGX Spark
- Best Fine-tuning Method - 1 DGX Spark

Participants must submit their entry for these awards through **[this form](https://docs.google.com/forms/d/1O9XN6v0fCDkBVChbbcorBmT95rhca6_ELz8RwwJnx9c/preview?resourcekey=0-I6m8wE4Cap8m5NuOBLk_Cg)** linking to their notebook and clearly identifying the category being entered.

Only submissions ranking within the top 10% of the final leaderboard will be considered for Open Contribution Awards.

### Compute Powered by NVIDIA Blackwell on Google Cloud

link

keyboard\_arrow\_up

We're excited to partner with Google Cloud to offer the G4 VMs powered by NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs as the compute backbone for this challenge. G4 VMs are a strong fit for both fine‑tuning and high‑throughput inference with Nemotron models so you can iterate quickly on prompts, data pipelines, and tuning strategies while evaluating performance on real benchmarks.

NVIDIA RTX PRO 6000 Blackwell GPUs provide the performance and memory needed to serve open reasoning models efficiently at inference time, enabling responsive evaluation runs and rapid leaderboard iteration as you improve your Nemotron variants. Learn more about the underlying Google Cloud G4 VM and its capabilities [here](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series).

### Citation

link

keyboard\_arrow\_up

Jamil C Semaan, Jean-Francois Puget, Christof Henkel, Yi Dong, Addison Howard, Ashley Oldacre, Ryan Holbrook, Chris Alexiuk, and Rebecca Kao. NVIDIA Nemotron Model Reasoning Challenge. https://kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge, 2026. Kaggle.

Cite

## Competition Host

NVIDIA

[NVIDIA's profile](https://www.kaggle.com/organizations/nvidia)

## Prizes & Awards

$106,388

info

Awards Points & Medals

## Participation

11,864 Entrants

2,523 Participants

2,161 Teams

15,676 Submissions

## Tags

[Reasoning](https://www.kaggle.com/competitions?tagIds=17193-Reasoning) [Deep Learning](https://www.kaggle.com/competitions?tagIds=13310-Deep+Learning) [Pre-Trained Model](https://www.kaggle.com/competitions?tagIds=16668-PreTrained+Model)

Custom Metric

Table of Contents

collapse\_all

[Overview](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/abstract) [Description](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/description) [Evaluation](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/evaluation) [Timeline](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/timeline) [Prizes](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/prizes) [Compute Powered by NVIDIA Blackwell on Google Cloud](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/compute-powered-by-nvidia-blackwell-on-google-cloud) [Citation](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/citation)