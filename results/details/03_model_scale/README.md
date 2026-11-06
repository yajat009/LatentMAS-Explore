# 03 Model scale

**Status: SUGGESTIVE.** On MBPP+, the latent steps cost Qwen3-4B much more than Qwen3-14B. This rests on 45 questions and one seed.
The 8B result is INCONCLUSIVE. The 14B GPQA result is BUG-AFFECTED, but it is the only result here where LatentMAS beats the single agent.

## 4B against 14B on MBPP+

The settings match in both rows: first 45 questions, sequential layout, batch size 1, `--think`, L40S GPU, seed 42.

| Model | Steps 0 (method off) | Steps 10 (method on) | Paired result |
|---|---|---|---|
| Qwen3-4B | 37/45 (82.2%) | 25/45 (55.6%) | 1 win, 13 losses, p = 0.002 |
| Qwen3-14B | 42/45 (93.3%) | 41/45 (91.1%) | 1 win, 2 losses, p = 1 |

At 14B, steps 10 generated 7% fewer tokens (1,299 against 1,400) and took 4% less time (61.1 against 63.5 seconds per question).
There is no 14B TextMAS run on MBPP+, so the paper's speedup claim is not tested at 14B.

Limit: these rows use `--think`. Results in `02_latent_steps_and_think_tag/` show that the 4B cost depends on that flag, so the 14B comparison needs a no-`--think` run too.

## 8B on MBPP+

| Run | Correct | Note |
|---|---|---|
| Single agent, batch size 4 | 11/14 | Same 14 questions |
| LatentMAS, steps 20, batch size 1, `W_a` off | 13/14 | 2 wins, 0 losses, p = 0.5 |

Fourteen questions cannot separate the methods. The run with `W_a` on did not start.

## 14B on GPQA-Diamond (198 questions)

| Run | Correct | Against Single |
|---|---|---|
| Single agent | 79/198 (39.9%) | Reference |
| LatentMAS, steps 10 | 97/198 (49.0%) | 34 wins, 16 losses, p = 0.015 |
| TextMAS | 39/68 (57.4%), partial | 14 wins, 8 losses on 68, p = 0.29. LatentMAS also scored 39/68 on these 68. |

Earlier notes listed these runs as still running, so this result was never written up.

Limits: all three runs used batch size 4, so the padding bug affects them (`04_padding_bug/`).
In the single-agent run, only 51 of 198 outputs opened with `<think>`, and 50 of 198 never closed it. None of those 50 was correct.
LatentMAS had 40 outputs that never closed. Part of the gain can therefore come from the bug's damage to the single agent.
A batch-size-1 rerun would settle this.

## Evidence

- `scale_4b_vs_14b_45.json`: the 2 × 2 table, with jobs and caveats
- `../../data/runs.csv` and `../../data/question_outcomes.csv`: every run above
