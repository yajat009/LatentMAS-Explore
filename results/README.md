# Results

This folder holds every result of this repo, with a label that says how far to trust it.
Each numbered folder has one README and its evidence files.
Numbers were recomputed on 2026-11-06 from the raw checkpoints on HPC3.

## Labels

| Label | Meaning |
|---|---|
| CONFIRMED | Proven by computation, or a paired test on enough questions |
| SUGGESTIVE | Points one way, but with few questions, one seed, or a confound |
| BUG-AFFECTED | Uses batch size above 1, so the padding bug applies |
| INCONCLUSIVE | The data cannot separate the options |
| PILOT | A first run of a new measurement, with known gaps |
| OPEN | Results disagree, and a named test is needed |

## Index

| Folder | Question | Answer | Label | n |
|---|---|---|---|---|
| [01_paper_comparison](01_paper_comparison/) | Does 73.5% on Qwen3-4B MBPP+ reproduce with the method on? | No. 63/84 with the method off, 40/84 with it on, 61/84 for the single agent. | CONFIRMED | 84 of 378 |
| [02_latent_steps_and_think_tag](02_latent_steps_and_think_tag/) | Do more latent steps hurt at 4B? | Yes with `--think`. Without it, results are not monotonic. | CONFIRMED / OPEN | 45 to 84 |
| [03_model_scale](03_model_scale/) | Does model size change this? | The cost is 12 questions of 45 at 4B and 1 of 45 at 14B. | SUGGESTIVE | 45 |
| [03_model_scale](03_model_scale/) | Does LatentMAS beat the single agent anywhere? | 14B GPQA: 49.0% against 39.9%, p = 0.015, with padding in both runs. | BUG-AFFECTED | 198 |
| [04_padding_bug](04_padding_bug/) | Does batching break the runs? | Yes. Its cost in accuracy is not measured well. | CONFIRMED / SUGGESTIVE | 45 to 378 |
| [05_realign_matrix](05_realign_matrix/) | Does `W_a` do anything? | Not on 4B (it is the identity). On 8B and 14B it is a large change, and our 14B runs did not use it. | CONFIRMED | weights |
| [06_released_logs](06_released_logs/) | Did the paper's MBPP+ cells use latent steps? | The 14B cell used 0. The 4B cell probably did too. Whether this was an error is not proven. | CONFIRMED / SUGGESTIVE | 378 |
| [07_gsm8k](07_gsm8k/) | Do the methods differ on GSM8K? | The first questions cannot separate them. | INCONCLUSIVE | 40 to 100 |
| [08_content_audit](08_content_audit/) | Do latent messages carry question-specific content? | No sign of it with the identity map. Not measured with `W_a` on. | PILOT | 200 |

## Corrections to earlier notes

Earlier versions of the README and report said these things. The numbers above replace them.

| Earlier claim | Correction |
|---|---|
| "I measured 22.8%" as the headline | That run has the padding bug. The fair number is 47.6% on the first 84 questions at batch size 1. |
| "Every latent step makes 4B worse" | This holds only with `--think`. |
| "`--think` does not explain the drop" | This rests on 20 steps only. At 10 steps, removing `--think` won 10 questions and lost 0. |
| "`--think` turned off" | The flag only adds the text `<think>` to the prompt. The model reasons either way. |
| "Padding costs 18 points" | That gap is from 45 questions, with p = 0.08. |
| "Baseline and TextMAS results are trustworthy" | They ran at batch size above 1. On MBPP+ the single agent shows no measurable loss from this. |
| GPQA "still running" | It finished. See `03_model_scale/`. |
| Question-specific content is zero on 8B | That test used the identity map, which is not the paper's method on 8B. |
| LatentMAS slower than Single "in 11 of 12 cells" | 29 of the 36 cells in the two main tables. |

## What to run next

1. Repeat steps 0, 10, 20 and 40, with and without `--think`, on one GPU type with 3 seeds. This settles the OPEN result.
2. Run Qwen3-14B MBPP+ without `--think`, and with `--latent_space_realign`.
3. Rerun 14B GPQA at batch size 1, or with `--pad_fix`.
4. Add a swapped-with-ridge condition to the content audit, and run it on 8B and on MBPP+.
5. Run one full test set at batch size 1.

## Files

- `data/runs.csv`: one row per run, with settings, n and accuracy
- `data/question_outcomes.csv`: one row per run and question
- `data/content_audit_scores.csv`: one row per audit question and condition
- `raw/`: raw logs and checkpoints. Git ignores this folder, and the copy of record is on HPC3.

To recompute every paired comparison with no GPU:

```bash
python results/test_runs/analysis/recompute_result_tables.py
```

Run names in the CSV files have the form `<model>_<task>_<method>_steps<m>_bs<batch>_think<0|1>`.
Runs with fewer questions than the full test set use the first questions of that set. These can be easier than the full set: on MBPP+ the single agent scores 77.8% on the first 45 and 69.0% on all 378.
