# Autoresearch Runbook

## Current Platform: Kaggle (free, 96GB VRAM)

Training runs on Kaggle Notebooks with RTX Pro 6000 Blackwell (96GB VRAM). See **`KAGGLE_SETUP.md`** for full setup and submission instructions.

### Quick start

1. Fork https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo
2. Enable RTX Pro 6000 + Internet
3. Add training code where demo says `# YOUR CODE HERE`
4. Save & Run All → Submit from Output tab

### Experiment loop

1. Write training notebook on Kaggle (or locally, push via API)
2. Save & Run All (commits the notebook, runs on Kaggle GPU)
3. Submit from Output tab → get score
4. Iterate — 5 submissions/day, 30 GPU hours/week

---

## Archived: RunPod / GCP Setup

The following is kept for reference but **is no longer the active platform**. RunPod and GCP were abandoned because:
- A100 80GB is too tight for bf16 LoRA training (79/80GB for model weights alone)
- bitsandbytes quantization is broken for Mamba SSM layers
- CPU offloading makes training extremely slow (~2 steps in 10 min)
- Kaggle provides 96GB VRAM for free, eliminating all memory issues

### RunPod (archived)

- Pod was terminated. Pod ID `zvnhnt8hvzpi45` (stored in `.env`)
- Cost: $0.79/hr (community spot), but not worth it vs free Kaggle
- SSH required PTY workaround (expect wrapper)

### GCP (archived)

- Project: `nemotron-challenge`, authenticated as `wangxiyao2003@gmail.com`
- A100 80GB quota was 0 in most regions; preemptible quota=1 in us-east5 but stockout
- VM `nemotron-trainer` in us-east4-c was created once (preemptible, now stopped/deleted)
- Free credits available but GPU capacity was unreliable
