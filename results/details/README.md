# Details

Each numbered folder has the full numbers and evidence for one question. Start with [the results page](../README.md).

| Label | Meaning |
|---|---|
| CONFIRMED | Proven by computation, or a paired test on enough questions |
| SUGGESTIVE | Points one way, but with few questions, one seed, or a confound |
| BUG-AFFECTED | Uses batch size above 1, so the padding bug applies |
| INCONCLUSIVE | The data cannot separate the options |
| PILOT | A first run of a new measurement, with known gaps |
| OPEN | Results disagree, and a named test is needed |

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

