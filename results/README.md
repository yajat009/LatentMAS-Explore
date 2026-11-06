# Results

## The result in four sentences

The paper says that agents who share internal memory beat a single model.
We tested this on Qwen3-4B with coding questions (MBPP+).
When the agents really used the shared memory, accuracy fell from 75% to 48%.
The paper's 73.5% matches our score with the shared memory **not** used, so the reported gain may not come from the shared memory.

## The key numbers

Qwen3-4B, the same 84 MBPP+ questions, one question at a time:

| Setup | What it does | Correct |
|---|---|---|
| One model alone | No agents | 61 of 84 (73%) |
| Agents, 0 latent steps | The shared memory is thrown away, so the last agent answers alone | 63 of 84 (75%) |
| Agents, 20 latent steps | The last agent uses the shared memory | 40 of 84 (48%) |

The paper reports 73.5% for this method on all 378 questions.
A larger model loses much less: Qwen3-14B scored 42 of 45 without the memory and 41 of 45 with it.

## Only three settings matter

| Setting | Meaning | Use this value |
|---|---|---|
| Latent steps | How many silent thinking steps each agent takes. At 0, the shared memory is not used. | Report it for every run |
| Batch size | How many questions run at once. Above 1, a padding bug damages the runs. | 1 |
| `--think` | Adds the text `<think>` to the prompt. It does not turn reasoning on or off. | Test with and without it |

## What we know and what we do not know

| Question | Answer | How sure |
|---|---|---|
| Does the shared memory hurt the 4B model on MBPP+? | Yes, with `--think` | Sure |
| Does it hurt without `--think`? | Unclear: 10 steps were fine, 20 steps were not, 40 steps were fine | Not sure |
| Did the paper's 14B MBPP+ number use 0 latent steps? | Yes. The authors' log shows this. | Sure |
| Did the paper's 4B number use 0 latent steps? | Probably. The authors' own command does. | Fairly sure |
| Was that an error by the authors? | Unknown. They say 0 steps is part of their method. | Not proven |
| Does the shared memory ever help? | Unknown. 14B on GPQA scored 49% against 40%, but the padding bug likely lowered the 40%: the paper gives 48.5% for it. | Not sure |

## What is in this folder

| Item | What it is | Do you need it? |
|---|---|---|
| This page | The summary | Yes |
| `details/` | Full numbers for each question, and a list of corrected claims | Only to check a number |
| `data/` | Small CSV files behind the numbers | Only to recompute |
| `test_runs/` | The scripts that ran the tests | Only to run tests |
| `raw/` | Raw logs. Git ignores this folder. | No |
