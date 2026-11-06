# 02 Latent steps and the `--think` flag

**Status: CONFIRMED that latent steps hurt Qwen3-4B on MBPP+ when `--think` is on. OPEN whether they hurt without it.**

This result corrects two earlier claims:

- "Every step of latent thinking makes the 4B model worse." This holds only with `--think`.
- "`--think` does not explain the MBPP+ drop." This holds at 20 steps, but not at 10 steps or 40 steps.

## What `--think` does

`--think` does **not** switch reasoning on or off. It adds the text `<think>` to the end of the Judger prompt.
Without the flag, the model writes `<think>` and a newline itself, and then reasons as usual. Both modes reason.
The only difference is that the flag starts the output straight after `<think>`, with no newline.

## Results

Qwen3-4B, MBPP+, sequential layout, batch size 1.

| Latent steps | With `--think`, first 45 | Without `--think`, first 45 |
|---|---|---|
| 0 (method off) | 37/45 (82.2%) | not run |
| 1 | 35/45 (77.8%) | not run |
| 2 | 35/45 (77.8%) | not run |
| 5 | 29/45 (64.4%) | not run |
| 10 | 25/45 (55.6%) | 35/45 (77.8%) |
| 20 | 24/45 (53.3%) | 25/45 (55.6%) |
| 40 | 24/45 (53.3%) | 33/45 (73.3%), with realign flag |

On the first 84 questions, steps 0 scores 63, steps 20 with `--think` scores 40, and steps 20 without it scores 47.

Paired tests:

| A | B | n | B wins | B losses | p |
|---|---|---|---|---|---|
| Steps 0, `--think` | Steps 10, `--think` | 45 | 1 | 13 | 0.002 |
| Steps 0, `--think` | Steps 20, `--think` | 84 | 3 | 26 | 0.00002 |
| Steps 10, `--think` | Steps 10, no `--think` | 45 | 10 | 0 | 0.002 |
| Steps 20, `--think` | Steps 20, no `--think` | 84 | 19 | 12 | 0.28 |
| Steps 40, `--think` | Steps 40, no `--think` | 45 | 11 | 2 | 0.02 |
| Steps 10, no `--think` | Steps 20, no `--think` | 45 | 3 | 13 | 0.02 |

## What this shows

- With `--think`, accuracy falls from 82.2% to about 53% as steps go from 0 to 20, and then stays flat. It falls smoothly, with no single cliff.
- Without `--think`, the three runs do not form a smooth line. Steps 10 and 40 stay near the method-off score, and steps 20 drops.

## What this does not show

The no-`--think` runs differ in more than the flag:

- Steps 10 and 40 ran on L40S GPUs in August. Steps 20 ran on an A30 in November.
- The steps-40 run also had `--latent_space_realign`. That flag does nothing on Qwen3-4B (`../05_realign_matrix/`).
- Each cell is one seed. Several comparisons are shown, so a p-value near 0.02 is weak evidence.

A clean test repeats steps 0, 10, 20 and 40, with and without `--think`, on the same GPU with 3 seeds.

An independent implementation (`results/test_runs/run_memory_controls.py`) was meant to test whether the drop is specific to the upstream code. Job 59001448 hit its time limit at 20 of 60 questions. The script saves only at the end, so this test has no result.

## Evidence

- `latent_steps_sweep_45.json`: the sweep with `--think`
- `../../data/question_outcomes.csv`: every paired test above. Run `results/test_runs/analysis/recompute_result_tables.py`.
