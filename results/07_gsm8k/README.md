# 07 GSM8K

**Status: INCONCLUSIVE.** All methods score about 95% on the first questions, so these runs cannot separate them. No full run exists.

All rows: Qwen3-4B, sequential layout, first questions of 1,319.

| Run | Correct | Correct with numeric scoring |
|---|---|---|
| Single agent, batch size 1 | 96/100 | 96/100 |
| TextMAS, batch size 1 | 38/40 | 38/40 |
| LatentMAS, steps 20, batch size 1 | 81/87 | 83/87 |

The paper reports 82.4 / 89.8 / 88.2 on the full set.

- **Paired test.** On the 87 shared questions the single agent scored 85. LatentMAS lost 4 and won 0 (p = 0.12). Two of those losses are scorer errors.
- **Scorer error.** Upstream compares strings, so `57.00` does not match gold `57`. This is upstream behavior and is not fixed.
- **Shared misses.** On the first 40 questions, all three methods miss the same 2 with the same answers. Question 12 expects "year 13" where 90 / 7.5 = 12. Question 37 says he sells "all his toys" but expects 2 sets left.
- **Open question.** Our single agent scores far above the paper's 82.4. We do not know why, because no full run exists. The first questions may be easier, but this is untested for GSM8K.

Latent steps at this size do not show the MBPP+ drop. The cause of that difference is unknown.

## Evidence

- `../data/runs.csv`, `../data/question_outcomes.csv` (the `correct_numeric` column)
