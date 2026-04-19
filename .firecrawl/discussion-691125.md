menu

[Skip to\\
\\
content](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/691125#site-content)

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
More](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/691125#)


menu

[Skip to\\
\\
content](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/691125#site-content)

[![Kaggle](https://www.kaggle.com/static/images/site-logo.svg)](https://www.kaggle.com/)

search​

[Sign In](https://www.kaggle.com/account/login?phase=startSignInTab&returnUrl=%2Fcompetitions%2Fnvidia-nemotron-model-reasoning-challenge%2Fdiscussion%2F691125)

[Register](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2Fcompetitions%2Fnvidia-nemotron-model-reasoning-challenge%2Fdiscussion%2F691125)

[NVIDIA's profile](https://www.kaggle.com/organizations/nvidia) NVIDIA  · Featured Prediction Competition · 2 months to go

Join Competition

more\_horiz

# NVIDIA Nemotron Model Reasoning Challenge

Advance reasoning techniques using NVIDIA Nemotron open models on a novel benchmark

![](https://www.kaggle.com/competitions/129716/images/header)

## NVIDIA Nemotron Model Reasoning Challenge

[Overview](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview) [Data](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data) [Code](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/code) [Models](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/models) [Discussion](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion) [Leaderboard](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/leaderboard) [Rules](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules)

[Deval Mukherjee1's profile](https://www.kaggle.com/devalmukherjee1)

[Deval Mukherjee1](https://www.kaggle.com/devalmukherjee1) · 335th in this Competition · Posted 5 days ago

arrow\_drop\_up0

more\_vert

### Is the evaluation deterministic ?

I saw the post from Tong about his attempt and he mentioned that he got a score of .84 twice and .85 once or something like that. If the evaluation is deterministic how does that happen ? am I missing something ?

Please [sign in](https://www.kaggle.com/account/login) to reply to this topic.

comment

## 2 Comments

Hotness

[Mark Cooper's profile](https://www.kaggle.com/markjcooper)

[**Mark Cooper**](https://www.kaggle.com/markjcooper)

Posted 3 days ago

· 559th in this Competition

arrow\_drop\_up-1

more\_vert

Yes, it's non-deterministic in practice even though Kaggle sets temperature=0. You're not missing anything. Sources of variance:

1. **CUDA kernel non-determinism.**`torch.backends.cudnn.deterministic=False` is the default, and several ops (attention, softmax reductions, batched matmul) have tiny float-order differences across runs. At temperature=0 these rarely matter, but over 7680 tokens × 34 puzzles a single


flipped greedy decision cascades into a completely different answer.

2. **Batch composition.** If Kaggle batches your 34 puzzles with different padding arrangements or ordering, logits for identical inputs can shift by ~1e-6. Usually harmless, but at exact greedy boundaries it flips the picked token.

3. **GPU hardware drift.** Kaggle's grading infrastructure isn't guaranteed to route you to the same physical GPU / driver version between submissions. Different CUDA versions produce subtly different results.

4. **vLLM's prefix caching & scheduling.** If prefix caching is enabled server-side, concurrent request arrival order affects KV cache hits.

**Practical impact:** THK's 0.84/0.84/0.85 pattern suggests ~1 puzzle out of 34 flipping correct/wrong between runs = ~3% score variance. Matches your 0.64→0.66 (≈1 puzzle).

**What to do about it:**

5. Kaggle lets you pick 2 submissions for the final private leaderboard. Submit your best adapter 3-5 times, pick the highest-scoring two.

6. For your own local benchmarking, expect ±1-2% just from this — don't chase changes smaller than that as if they're real.

7. If you really need determinism, set `torch.use_deterministic_algorithms(True)` and `torch.backends.cudnn.deterministic=True` in your training/inference stack — hurts speed but pins outputs.

It's a known feature of large-model inference at exact greedy, not a bug in the competition's grader.


[Trishanth Mellimi's profile](https://www.kaggle.com/trishanthmellimi)

[**Trishanth Mellimi**](https://www.kaggle.com/trishanthmellimi)

Posted 5 days ago

· 341st in this Competition

arrow\_drop\_up0

more\_vert

even i got non deterministic score, 0.64 and 0.66 for the same submission a week back