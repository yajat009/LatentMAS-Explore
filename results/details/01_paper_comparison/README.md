# 01 Comparison with the paper on Qwen3-4B MBPP+

**Status: CONFIRMED for the first 84 questions. The paper's 73.5% does not reproduce with the method on and `--think`.**
No full 378-question run exists with the method on at batch size 1.

MBPP+ has 378 Python tasks. An answer is correct only if its code passes every hidden test.
The paper reports 63.5 (Single), 69.8 (TextMAS) and 73.5 (LatentMAS), in its sequential layout.

## Fair comparison: batch size 1

All rows use the same first 84 questions, with no padding. Each paired result is against LatentMAS with 0 steps.

| Run | Correct | Paired result |
|---|---|---|
| Single agent | 61/84 (72.6%) | 3 wins, 5 losses against steps 0, p = 0.73 |
| LatentMAS, steps 0 (method off), `--think` | 63/84 (75.0%) | Reference |
| LatentMAS, steps 20, `--think` | 40/84 (47.6%) | 3 wins, 26 losses, p = 0.00002 |
| LatentMAS, steps 20, no `--think` | 47/84 (56.0%) | See `../02_latent_steps_and_think_tag/` |

Against the single agent, LatentMAS with steps 20 and `--think` has 2 wins and 23 losses (p = 0.00002).
The paper reports LatentMAS 10 points above the single agent. Here it is 25 points below.

The single agent and LatentMAS with 0 steps are about equal, as expected: at 0 steps the Judger gets no cache.
The first 45 LatentMAS results in each row come from earlier runs that the later runs resumed from.

## Full test set, batch size above 1 (BUG-AFFECTED)

These are the only full 378-question runs. The padding bug affects all of them (`../04_padding_bug/`), so the LatentMAS rows are **not** a fair score for the method.

| Run | Batch size | Correct | Seconds per question | Generated tokens per question | Prompt tokens per question |
|---|---|---|---|---|---|
| Single agent | 15 | 261/378 (69.0%) | 29.0 | 1,430 | 243 |
| TextMAS | 8 | 273/378 (72.2%) | 93.1 | 2,154 | 5,119 |
| LatentMAS, steps 10 | 8 | 106/378 (28.0%) | 44.6 | 1,538 | 1,092 |
| LatentMAS, steps 10 | 15 | 86/378 (22.8%) | 44.5 | 1,578 | 1,092 |

- TextMAS and the single agent do not differ significantly: 36 wins, 24 losses, p = 0.16.
- At matched batch size 8, LatentMAS is 2.09× faster than TextMAS and generates 1.40× fewer tokens. The paper reports 3.7× and 3.3×.
- Our TextMAS generates 2,154 tokens per question, and the paper's generates 4,420. The cause is unknown.

Earlier versions of this repo led with "22.8%". That number includes the padding bug, so do not quote it as the method's score.

## How failures look

Counted on the Judger's output. "Collapsed" means that fewer than 15% of the last 400 words are distinct.

| Run | Accuracy | Collapsed | No `</think>` | No code extracted |
|---|---|---|---|---|
| Single agent, batch size 15 | 69.0% | 0.5% | 13.2% | 13.0% |
| TextMAS, batch size 8 | 72.2% | 0.5% | 29.6% | 2.6% |
| LatentMAS steps 10, batch size 15 | 22.8% | 23.5% | 48.7% | 31.5% |
| LatentMAS steps 10, batch size 1, `--think` | 55.6% | 13.3% | 8.9% | 15.6% |
| LatentMAS steps 10, batch size 1, no `--think` | 77.8% | 0.0% | 6.7% | 6.7% |
| Qwen3-14B LatentMAS steps 10, batch size 1 | 91.1% | 0.0% | 4.4% | 4.4% |

The collapsed outputs repeat fragments such as "thefunction and thefunction and". Some failing Judgers mention "the initial plan" although no agent wrote a plan in text, which shows that the cache reaches the Judger.

## Evidence

- `method_comparison_full_378.json`: output of `results/test_runs/analysis/compare_methods_from_checkpoints.py`
- `../../data/runs.csv`, `../../data/question_outcomes.csv`
