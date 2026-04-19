menu

[Skip to\\
\\
content](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo#site-content)

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
More](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo#)


menu

[Skip to\\
\\
content](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo#site-content)

[![Kaggle](https://www.kaggle.com/static/images/site-logo.svg)](https://www.kaggle.com/)

search​

[Sign In](https://www.kaggle.com/account/login?phase=startSignInTab&returnUrl=%2Fcode%2Fryanholbrook%2Fnvidia-nemotron-submission-demo)

[Register](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2Fcode%2Fryanholbrook%2Fnvidia-nemotron-submission-demo)

[Ryan Holbrook's profile](https://www.kaggle.com/ryanholbrook) Ryan Holbrook  · 23d ago · 21,797 views

arrow\_drop\_up1685

Copy & Edit
2167

file\_downloadDownload

![gold medal](https://www.kaggle.com/static/images/medals/notebooks/goldl@1x.png)

more\_vert

# NVIDIA Nemotron Submission Demo

## NVIDIA Nemotron Submission Demo

[Notebook](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo/notebook) [Input](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo/input) [Output](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo/output) [Logs](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo/log) [Comments (44)](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo/comments)

historyVersion 12 of 12chevron\_right

## Runtime

play\_arrow

16m 43s · GPU RTX Pro 6000

## Input

COMPETITIONS

![](https://www.kaggle.com/competitions/129716/images/thumbnail)

NVIDIA Nemotron Model Reasoning Challenge

MODELS

![](https://storage.googleapis.com/kaggle-avatars/images/13166197-kg.png)

Nemotron-3-Nano-30B-A3B-BF16 · ...b-a3b-bf16 · V1

## Tags

[GPU](https://www.kaggle.com/code?tagIds=16580-GPU)

## Language

Python

![Profile picture for undefined](https://storage.googleapis.com/kaggle-competitions/kaggle/129716/logos/thumb76_76.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1776460257&Signature=br4OwVCd43ceTFt7Gc9Roxcno2paFbjeefH6b%2FbXLksShOoOZuerrejWJq%2B0ZdcH90fSxxc%2F01ldZayg%2B22g5n1TVp%2BUyybjE6LKo557lLs1aY2drA42j47L5X3CENvDRo2F9PoosWlc4oTvx%2BeKxOUoWz0an3%2F%2FTHhLGJ9pOboeIxal80V%2F0KWk9s2hHk%2Bl%2F%2Fx7HFtL2B3l%2BJvW1pIqiUR6f8JMAsRYOYOkU1ArduA%2F4Uiy4vUuPyogAcUJM5WHndDmxyMWx28wViB8FE%2BRIEF%2BCuh2gITB%2F%2FlUvzaVuvgIBRL5XkRtUNMLKt17v1DBZnceT47FYJ29lceG7iTQBw%3D%3D&t=2026-02-05-16-10-30)

Competition Notebook

[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)

Public Score

0.49

Best Score

[0.50 V11](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo?scriptVersionId=303833043)

\_\_notebook\_\_

In \[1\]:

linkcode

```
import polars as pl

train = pl.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv')

train.head()
```

Out\[1\]:

shape: (5, 3)

| id | prompt | answer |
| --- | --- | --- |
| str | str | str |
| --- | --- | --- |
| "00066667" | "In Alice's Wonderland, a secre… | "10010111" |
| "000b53cf" | "In Alice's Wonderland, a secre… | "01000011" |
| "00189f6a" | "In Alice's Wonderland, secret … | "cat imagines book" |
| "001b24c4" | "In Alice's Wonderland, numbers… | "XXXVIII" |
| "001c63cb" | "In Alice's Wonderland, secret … | "wizard creates secret" |

In \[2\]:

linkcode

```
import site

cutlass_pkg_path = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
site.addsitedir(cutlass_pkg_path)

import kagglehub
import mamba_ssm
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working"
LORA_RANK = 32  # Can be set to a maximum of 32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16
)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Model loaded successfully.")

# Initialize LoRA Adapter
print(f"Initializing LoRA adapter with rank={LORA_RANK}...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# YOUR CODE HERE
# --------------
# model.train()
# --------------

# Save Adapter
print(f"Saving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
```

```
/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/torch/compiler/__init__.py:148: FutureWarning: torch._dynamo.allow_in_graph is deprecated and will be removed in a future version. Use torch._dynamo.nonstrict_trace instead.
  return torch._dynamo.allow_in_graph(fn)
```

```
Model loaded successfully.
Initializing LoRA adapter with rank=32...
```

```
/usr/local/lib/python3.12/dist-packages/torchao/float8/float8_tensor.py:122: FutureWarning: torch._dynamo.allow_in_graph is deprecated and will be removed in a future version. Use torch._dynamo.nonstrict_trace instead.
  @torch._dynamo.allow_in_graph
/usr/local/lib/python3.12/dist-packages/torchao/float8/float8_tensor.py:195: FutureWarning: torch._dynamo.allow_in_graph is deprecated and will be removed in a future version. Use torch._dynamo.nonstrict_trace instead.
  @torch._dynamo.allow_in_graph
/usr/local/lib/python3.12/dist-packages/torchao/float8/float8_scaling_utils.py:90: FutureWarning: torch._dynamo.allow_in_graph is deprecated and will be removed in a future version. Use torch._dynamo.nonstrict_trace instead.
  @torch._dynamo.allow_in_graph
/usr/local/lib/python3.12/dist-packages/torchao/float8/float8_linear.py:60: FutureWarning: torch._dynamo.allow_in_graph is deprecated and will be removed in a future version. Use torch._dynamo.nonstrict_trace instead.
  @torch._dynamo.allow_in_graph
/usr/local/lib/python3.12/dist-packages/torchao/dtypes/nf4tensor.py:1059: FutureWarning: torch._dynamo.allow_in_graph is deprecated and will be removed in a future version. Use torch._dynamo.nonstrict_trace instead.
  @torch._dynamo.allow_in_graph
/usr/local/lib/python3.12/dist-packages/torchao/quantization/subclass.py:206: FutureWarning: torch._dynamo.allow_in_graph is deprecated and will be removed in a future version. Use torch._dynamo.nonstrict_trace instead.
  @torch._dynamo.allow_in_graph
/usr/local/lib/python3.12/dist-packages/torchao/quantization/subclass.py:355: FutureWarning: torch._dynamo.allow_in_graph is deprecated and will be removed in a future version. Use torch._dynamo.nonstrict_trace instead.
  @torch._dynamo.allow_in_graph
/usr/local/lib/python3.12/dist-packages/torchao/quantization/subclass.py:392: FutureWarning: torch._dynamo.allow_in_graph is deprecated and will be removed in a future version. Use torch._dynamo.nonstrict_trace instead.
  @torch._dynamo.allow_in_graph
```

```
trainable params: 880,138,240 || all params: 32,458,075,584 || trainable%: 2.7116
Saving adapter to /kaggle/working...
```

In \[3\]:

linkcode

```
import subprocess

subprocess.run("zip -m submission.zip *", shell=True, check=True)
```

```
  adding: README.md (deflated 66%)
  adding: __notebook__.ipynb (deflated 75%)
  adding: adapter_config.json (deflated 55%)
  adding: adapter_model.safetensors (deflated 77%)
```

Out\[3\]:

```
CompletedProcess(args='zip -m submission.zip *', returncode=0)
```

In \[4\]:

linkcode

```
print('Done.')
```

```
Done.
```

## License

This Notebook has been released under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) open source license.

## Continue exploring

- ![](https://www.kaggle.com/static/images/kernel/viewer/input_light.svg)







Input

2 files




arrow\_right\_alt

- ![](https://www.kaggle.com/static/images/kernel/viewer/output_light.svg)







Output

1 file




arrow\_right\_alt

- ![](https://www.kaggle.com/static/images/kernel/viewer/logs_light.svg)







Logs

1002.9 second run - successful




arrow\_right\_alt

- ![](https://www.kaggle.com/static/images/kernel/viewer/comments_light.svg)







Comments

44 comments




arrow\_right\_alt