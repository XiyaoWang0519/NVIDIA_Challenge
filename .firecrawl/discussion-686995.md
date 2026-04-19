menu

[Skip to\\
\\
content](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/686995#site-content)

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
More](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/686995#)


menu

[Skip to\\
\\
content](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/686995#site-content)

[![Kaggle](https://www.kaggle.com/static/images/site-logo.svg)](https://www.kaggle.com/)

search​

[Sign In](https://www.kaggle.com/account/login?phase=startSignInTab&returnUrl=%2Fcompetitions%2Fnvidia-nemotron-model-reasoning-challenge%2Fdiscussion%2F686995)

[Register](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2Fcompetitions%2Fnvidia-nemotron-model-reasoning-challenge%2Fdiscussion%2F686995)

[NVIDIA's profile](https://www.kaggle.com/organizations/nvidia) NVIDIA  · Featured Prediction Competition · 2 months to go

Join Competition

more\_horiz

# NVIDIA Nemotron Model Reasoning Challenge

Advance reasoning techniques using NVIDIA Nemotron open models on a novel benchmark

![](https://www.kaggle.com/competitions/129716/images/header)

## NVIDIA Nemotron Model Reasoning Challenge

[Overview](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview) [Data](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data) [Code](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/code) [Models](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/models) [Discussion](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion) [Leaderboard](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/leaderboard) [Rules](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules)

[lucian kucera's profile](https://www.kaggle.com/luciankucera)

[lucian kucera](https://www.kaggle.com/luciankucera) · 867th in this Competition · Posted 17 days ago

arrow\_drop\_up0

more\_vert

### Why every bit problem always has INFINITE AMOUNT OF WRONG SOLUTIONS\[WHY IT DOESN'T TEST REASONING\]

This solution is consistently wrong it satisfies all examples:
a ^ b = 0000 0000: when they are equal
a ^ b = could be anything, if a all ones and b al zeros, than its all ones
if a is zeros and b is 0100 1010, than its result of xor is b

So we can assume xor can result into literally any byte.

So we need a way to detect if there is one in output of xor.

If there is one we now have all ones
(a ^ b) \| ((a ^ b) >> 1) …. \| ((a ^ b) >> 7) \| ((a ^ b) << 1) …. \| ((a ^ b) << 7)

So just apply not and u get == operator. Then we can just apply logical AND for each example pair.

So the bitwise problem isn't proble of finding solution for the examples, but guessing what nvdia thought when creating training data? So it is just a guessing game not reasoning problem.

This proofs that there is infite amount of bitwise formula that are wrong solutions, but satisfy condtiion. Imo there is not much point into focusing on this problem, since it is ALL LUCK.

This works for all problems which dont result into all zeros(SO MOST PROBLEMS)

TLDR:

If the evaluation was based on correctness: correct answer is always 0000 0000, no matter the task. So are we really sure if we train the model, solution will be right, but not NVDIA RIGHT? Hence the need is to reverse engineer the training set and see what transformations are used.

Please [sign in](https://www.kaggle.com/account/login) to reply to this topic.

comment

## 0 Comments

Hotness