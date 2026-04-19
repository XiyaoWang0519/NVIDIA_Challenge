menu

[Skip to\\
\\
content](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687798#site-content)

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
More](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687798#)


menu

[Skip to\\
\\
content](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687798#site-content)

[![Kaggle](https://www.kaggle.com/static/images/site-logo.svg)](https://www.kaggle.com/)

search​

[Sign In](https://www.kaggle.com/account/login?phase=startSignInTab&returnUrl=%2Fcompetitions%2Fnvidia-nemotron-model-reasoning-challenge%2Fdiscussion%2F687798)

[Register](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2Fcompetitions%2Fnvidia-nemotron-model-reasoning-challenge%2Fdiscussion%2F687798)

[NVIDIA's profile](https://www.kaggle.com/organizations/nvidia) NVIDIA  · Featured Prediction Competition · 2 months to go

Join Competition

more\_horiz

# NVIDIA Nemotron Model Reasoning Challenge

Advance reasoning techniques using NVIDIA Nemotron open models on a novel benchmark

![](https://www.kaggle.com/competitions/129716/images/header)

## NVIDIA Nemotron Model Reasoning Challenge

[Overview](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview) [Data](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data) [Code](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/code) [Models](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/models) [Discussion](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion) [Leaderboard](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/leaderboard) [Rules](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules)

[Ryan Holbrook's profile](https://www.kaggle.com/ryanholbrook)

[Ryan Holbrook](https://www.kaggle.com/ryanholbrook) · Posted 15 days ago

· Kaggle Staff

arrow\_drop\_up14

more\_vert

### Rescore After Metric Update

Hi everyone,

Recently, there was an update to the [evaluation metric](https://www.kaggle.com/code/metric/nvidia-nemotron-metric) that fixed a bug that was causing binary answers to match as floats instead of exactly as strings. This update led to a drop of about 0.3-0.4 points on new submissions. (The update is in the `verify()` function of the linked notebook.)

So that the leaderboard will only reflect submissions using the new metric, I will be initiating a rescore on Monday. This rescore will only evaluate those submissions that both are still present on the Public Leaderboard and that were made on or before March 28th, that is, prior to the metric update. This will affect a little less than 2/3 of the current LB. Other submissions made on or before March 28th, I will invalidate. Submissions made after that time will be preserved.

Why are we not rescoring every submission? Unfortunately, this would amount to over 9000 rescores and would strain our available capacity of the special GPUs used in this competition. We believe the above approach is the best compromise, that both preserves the existing leaderboard and maintains GPU availability for the community going forward.

Please let us know if you have any questions or concerns below.

Please [sign in](https://www.kaggle.com/account/login) to reply to this topic.

comment

## 14 Comments

Hotness

### Pinned comments

[push\_pin](https://www.kaggle.com/ryanholbrook)

[**Ryan Holbrook**](https://www.kaggle.com/ryanholbrook)

Kaggle Staff

Posted 10 days ago

arrow\_drop\_up3

more\_vert

The rescore is complete!

[push\_pin](https://www.kaggle.com/ryanholbrook)

[**Ryan Holbrook**](https://www.kaggle.com/ryanholbrook)

Kaggle Staff

Posted 12 days ago

arrow\_drop\_up4

more\_vert

I am about to begin the rescore process. The leaderboard may be in a strange state for a bit until the rescore completes. Here is what will happen:

1. **DONE** I will invalidate submissions _not_ still present on the leaderboard that were made on or before March 28th.
2. **DONE** I will initiate a rescore of those submissions still present on the leaderboard that were made on or before March 28th.

The rescore may take awhile to complete. I will keep you updated here.

### All other comments

[Sandy's profile](https://www.kaggle.com/sandeepgoyal1980)

[**Sandy**](https://www.kaggle.com/sandeepgoyal1980)

Posted 3 days ago

· 348th in this Competition

arrow\_drop\_up-1

more\_vert

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F32817761%2F80bd3e3539a0200f9da4658899dc3f1d%2FScreenshot%202026-04-16%20122210.png?generation=1776322383346556&alt=media)

has the submissions been rescored today too?
i am sure i had 5x0.84s and 2x0.85 before

[CPMP's profile](https://www.kaggle.com/cpmpml)

[**CPMP**](https://www.kaggle.com/cpmpml)

Competition Host

Posted 7 days ago

arrow\_drop\_up0

more\_vert

chatGPT, Gemini, Claude? Which one did you use?

[Tong Hui Kang's profile](https://www.kaggle.com/huikang)

[**Tong Hui Kang**](https://www.kaggle.com/huikang)

Posted 15 days ago

· 191st in this Competition

arrow\_drop\_up3

more\_vert

I want to confirm if the stored answer is `0234`, is `234` the considered correct?

I remember someone wrote a comment asking this, but I think it has since been deleted.

```python
import re
import math

def verify(stored_answer: str, predicted: str) -> bool:
    # Clean up strings
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # If the answer is a binary string, compare strictly as strings
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        # Try to convert the answers to floating point numbers
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        # Use a small absolute tolerance for numbers near zero
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception as e:
        print(e)
        # Fallback to case-insensitive string comparison
        return predicted.lower() == stored_answer.lower()

print(verify("0234", "234"))
content_copy
```

Currently this is True.

[Ryan Holbrook's profile](https://www.kaggle.com/ryanholbrook)

[**Ryan Holbrook**](https://www.kaggle.com/ryanholbrook)

Kaggle Staff

Posted 15 days ago

arrow\_drop\_up1

more\_vert

I will look into it. As this would require another update and rerun, perhaps it's better to accept the ambiguity if it doesn't meaningfully affect the scores. Will let you know.

[Sangram Patil's profile](https://www.kaggle.com/sangrampatil5150)

[**Sangram Patil**](https://www.kaggle.com/sangrampatil5150)

Posted 12 days ago

· 57th in this Competition

arrow\_drop\_up1

more\_vert

One of the bit manipulation questions is:

```1c
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
01100110 -> 00000001
01111111 -> 10000001
11111111 -> 10000011
00011110 -> 00000000
11000011 -> 10000011
00100101 -> 10000000
11100101 -> 10000011
11100100 -> 00000011
01011110 -> 00000001

Now, determine the output for: 01011000
content_copy
```

The given ground truth label is **1 only**, but the expected output is `00000001`. So I think the labels were created according to this format.

Also, some of the ground truth labels in the bit manipulation tasks are not 8-bit (e.g., `1`, `111101`, `1000100`, `10111111`, etc.). Even if we train the model to predict 8-bit outputs, we need the ground truth labels to be in the same format if we want to compute accurate metrics.

[Ravi Ramakrishnan's profile](https://www.kaggle.com/ravi20076)

[**Ravi Ramakrishnan**](https://www.kaggle.com/ravi20076)

Posted 15 days ago

· 9th in this Competition

arrow\_drop\_up-2

more\_vert

[@ryanholbrook](https://www.kaggle.com/ryanholbrook) I wish to purport that a lot of the recently launched competitions are having such operational issues including-

1. Post launch metric adjustments
2. Post launch data adjustments
3. Post launch data + metric adjustments
4. Ethical issues resulting in pausing the competition after launch

I think some additional operational enhancements with human-driven supervision could be improved to curb such post-launch sub-optimalities.

[CPMP's profile](https://www.kaggle.com/cpmpml)

[**CPMP**](https://www.kaggle.com/cpmpml)

Competition Host

Posted 15 days ago

arrow\_drop\_up0

more\_vert

Which ethical issues?
Which post launch data adjustment?

There has been one metric change so far.

[ImperfectKitto's profile](https://www.kaggle.com/defdet)

[**ImperfectKitto**](https://www.kaggle.com/defdet)

Posted 15 days ago

· 911th in this Competition

arrow\_drop\_up0

more\_vert

[@ravi20076](https://www.kaggle.com/ravi20076) is most likely talking about overall decline in quality of Kaggle competitions (those are issues other past and current competitions have)

[Ravi Ramakrishnan's profile](https://www.kaggle.com/ravi20076)

[**Ravi Ramakrishnan**](https://www.kaggle.com/ravi20076)

Posted 15 days ago

· 9th in this Competition

arrow\_drop\_up-1

more\_vert

[@defdet](https://www.kaggle.com/defdet) yes, a lot of the recently launched competitions in the last 4-5 months have had these issues. I am unsure if this is due to AI usage / supervision / governance issues.

[CPMP's profile](https://www.kaggle.com/cpmpml)

[**CPMP**](https://www.kaggle.com/cpmpml)

Competition Host

Posted 14 days ago

arrow\_drop\_up5

more\_vert

Why use this competition forum for a cross competition discussion?

[Ryan Holbrook's profile](https://www.kaggle.com/ryanholbrook)

[**Ryan Holbrook**](https://www.kaggle.com/ryanholbrook)

Kaggle Staff

Posted 14 days ago

arrow\_drop\_up4

more\_vert

Hi [@ravi20076](https://www.kaggle.com/ravi20076),

I will bring your concerns up with the team. You might wish to raise them in the [Product Feedback](https://www.kaggle.com/discussions/product-feedback) forum instead to keep this forum on topic.

[Chew Kok Wah's profile](https://www.kaggle.com/chewkokwahibrainai)

[**Chew Kok Wah**](https://www.kaggle.com/chewkokwahibrainai)

Posted 14 days ago

· 194th in this Competition

arrow\_drop\_up3

more\_vert

Host willingness to adjust Metric and data when error is found is showing a behavior of high integrity. It is unfair and discouraging to them to negatively turn a highly ethical behavior into something negative.

Beside that, there is no way to confirm if all the past Competitions before these 4, 5 months is better or worst in term of data or Metric incorrect issues. Most of them are private and never reveal to you.

[Muhammad's profile](https://www.kaggle.com/hark99)

[**Muhammad**](https://www.kaggle.com/hark99)

Posted 9 days ago

· 2070th in this Competition

arrow\_drop\_up0

more\_vert

[@ryanholbrook](https://www.kaggle.com/ryanholbrook) I just want to do inference on the model to check results but """PermissionError: \[Errno 13\] Permission denied: '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell'"""

[Yurnero's profile](https://www.kaggle.com/samson8)

[**Yurnero**](https://www.kaggle.com/samson8)

Posted 13 days ago

· 580th in this Competition

arrow\_drop\_up0

more\_vert

[@ryanholbrook](https://www.kaggle.com/ryanholbrook)

In the overview tab we can find this.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F13977249%2F2558cd409c1e98a690dfd190659dad08%2F1.jpg?generation=1775398654086016&alt=media)

but in the last version (14th) of the [NVIDIA Nemotron Metric notebook](https://www.kaggle.com/code/metric/nvidia-nemotron-metric) I see this

```php
def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    max_lora_rank: int = 32,
    max_tokens: int = 3584,
    top_p: float = 1.0,
    temperature: float = 1.0,
    max_num_seqs: int = 128,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    debug: bool = False,
)
content_copy
```

Is it a placeholder and actual scoring parameters are correct in the overview tab. I'm specifically interested in `max_tokens` and `max_model_length`

[Ryan Holbrook's profile](https://www.kaggle.com/ryanholbrook)

[**Ryan Holbrook**](https://www.kaggle.com/ryanholbrook)

Kaggle Staff

Posted 13 days ago

arrow\_drop\_up2

more\_vert

The defaults are only defaults. The correct parameters are in the Overview tab.