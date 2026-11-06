# 04 Padding bug

**Status: CONFIRMED as a bug. Its effect on accuracy is SUGGESTIVE.** Every result with batch size above 1 has this bug.

## What goes wrong

The tokenizer pads short prompts on the right, so a batch is rectangular.

- **LatentMAS.** `generate_latent_batch` starts the latent steps from the last position, which is a pad token for short prompts. It then builds an all-ones attention mask, so agents attend to the pads in the cache.
- **Single agent and TextMAS.** `generate()` continues from the last column, which is a pad token for short prompts. The first output token is then wrong.

## Measurements

| Measurement | Result |
|---|---|
| LatentMAS sequences that start from a pad, batch size 15 | 93.3% |
| Share of LatentMAS cache positions that are pads | 35.8% |
| Same two numbers in the authors' released HumanEval+ log | 94.4% and 32.7% |
| p(`<think>`) as first token, first 8 GSM8K prompts, no padding | 1.000 |
| Same, on padded rows | as low as 0.000 |
| Outputs that open normally, padded rows, GSM8K | 1 of 49 |
| Outputs that open normally, padded rows, MBPP+ | 9 of 352 |
| Outputs that open normally, batch size 1 | 50 of 50 |

## Effect on accuracy

| Comparison (Qwen3-4B MBPP+) | Result | Paired test |
|---|---|---|
| LatentMAS steps 10, batch size 1 against 15, first 45 questions | 25/45 against 17/45 | 4 wins, 12 losses, p = 0.08 |
| LatentMAS steps 10, batch size 8 against 15, all 378 | 106/378 against 86/378 | 41 wins, 61 losses, p = 0.06 |
| Single agent, batch size 1 against 15, first 84 questions | 61/84 against 59/84 | 3 wins, 5 losses, p = 0.73 |

Earlier notes stated "padding costs 18 points" as a measured fact. The 18-point gap comes from 45 questions and is not significant on its own.
The direction agrees in both LatentMAS comparisons. The single agent shows no measurable loss on MBPP+.

No completed run used `--pad_fix`, so the fix is untested at scale. The only run from that test is the unfixed half, with 24 questions.

## Evidence

- `padding_test_45.json`: the 45-question test and the contamination rates
- `../data/question_outcomes.csv`: the paired outcomes
