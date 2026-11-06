# LatentMAS on HPC3: inventory, reduced reproduction, audit pilot, cleanup

## Where things stand, 2026-11-06 12:30 (read this first; it replaces the summaries below)

**Why our results differed from the paper, in four points**

1. We ran LatentMAS with 20 latent steps. The paper's Qwen3-4B and Qwen3-8B numbers behave like runs with 0 latent steps, which is the default of the README commands.
2. At 0 steps the released Hugging Face code gives the Judger no latent memory (`methods/latent_mas.py`: `past_for_decoding = past_kv if self.latent_steps > 0 else None`). The run is then one agent with the Judger prompt.
3. At 0 steps we reproduce the paper's LatentMAS numbers, and we agree with the authors' own released logs on 94-95% of questions. With 20 steps, 4B loses 20-25 points on MBPP+, for us and for other users (issues #10, #22, #50).
4. The paper's single-agent baseline does not reproduce. The released code scores 8-9 points higher, for us and for at least five other users (#39, #41, #29, #22, #8). So on the cells we tested, LatentMAS and the baseline tie.

### Test 1: the README commands on GSM8K, 100 fixed random questions

Design: the README's own 4B and 8B example commands (`--max_new_tokens 2048`, sequential, no `--latent_steps`, so 0, default batch size 20), and the README baseline command. All four runs use the same 100 test questions, drawn with `--subset_seed 0` (a new opt-in flag; the first questions of GSM8K are easier than the full set, so a prefix would bias the comparison). Sampling seed 42, one run each. Logs and checkpoints: `results/readme_repro/`. Comparison: `hpc3/compare_readme_repro.py`, output in `results/readme_repro/compare.md`.

| Model | Method | n | Ours % | SE | Paper % | Gap | Gap in SE | Authors' log, same questions |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | Single (baseline) | 100 | 90.0 | 3.0 | 82.4 | +7.6 | +2.5 | - |
| Qwen3-4B | LatentMAS, README command (0 steps) | 100 | 90.0 | 3.0 | 88.2 | +1.8 | +0.6 | 90/100, agree on 94 |
| Qwen3-8B | Single (baseline) | 80 | 90.0 | 3.4 | 81.1 | +8.9 | +2.7 | - |
| Qwen3-8B | LatentMAS, README command (0 steps) | 100 | 90.0 | 3.0 | 93.8 | -3.8 | -1.3 | 93/100, agree on 95 |

Paired on the same questions: 4B LatentMAS against baseline 2 wins, 2 losses; 8B (n=80) 1 win, 1 loss. Both ties. Numeric rescoring (the paper's stated rule; the code compares strings) changes only 4B LatentMAS, to 91.0.

Reading: the LatentMAS column reproduces within noise (4B +0.6 SE, 8B -1.3 SE). The baseline column does not (both +2.5 SE or more). The paper's claimed gain of LatentMAS over Single (+5.8 on 4B, +12.7 on 8B) does not appear.

### Test 2: the authors' released logs, no GPU

The README links one log per cell (Google Drive). I downloaded the sequential GSM8K and MBPP+ logs for 4B and 8B. Each has a per-question result line, so the authors' runs can be scored on our exact questions.

- Their final accuracies equal the paper: GSM8K 4B 88.25, MBPP+ 4B 73.54, MBPP+ 8B 74.60.
- **GSM8K 8B:** the log has 1198 of 1319 correct, 90.8%. Its summary line says accuracy 0.938 (the paper's 93.8) next to `"correct": 1198`. The two numbers in that line do not agree. Our 90.0 matches the per-question count, not the paper.
- The logs come from an older codebase (`/data_storage/zyf/zjr/MUSCLE/`, method name `muscle`). They do not record the latent-step count. The GSM8K 8B log ran at 0.46 s per question, which the Hugging Face path cannot do.
- The sequential GSM8K logs use prompts that are not in the released `prompts.py` or its git history ("planner agent in a sequential latent-sharing system", `<latent_think_start>` tags). The 4B Judger answers in a median of 468 characters with no thinking block.
- The MBPP+ 4B log ran at batch size 50 with `<think>` appended and no system prompt. The released defaults are batch size 20 and a system prompt.
- **MBPP+ 4B, same 84 questions:** our 0-step run agrees with the authors' log on 92.9% (75.0 against 77.4). Our 20-step run agrees on 63.1% (47.6). Our full-set baseline (bs 15) agrees on 83.9% (69.0 against 73.5).

So the paper's exact numbers cannot be regenerated from the released code: the prompts and settings differ. The released code at 0 steps lands within noise of them.

### What the upstream issue tracker shows

The upstream repo has 56 issues and pull requests. I read the 17 that concern reproduction. They report the same three things we measured.

| Issue | Reporter's finding | Authors' reply |
|---|---|---|
| #10 (closed) | 4B MBPP+ LatentMAS scores 26.2% with 10 latent steps, 32.8% with 1 step. The released example log uses 0 steps. | Use the command without `--latent_steps`. "Setting latent_steps = 0 simply means that the model performs one forward pass of thoughts generation." |
| #13 (closed) | At 0 steps the Judger discards the KV cache, so the run is a single agent. | The cache "can still" be passed, "as shown in our vLLM backend". The Hugging Face code was not changed. |
| #8 (closed) | 8B AIME24: baseline 83.3, TextMAS 73.3, LatentMAS 23.3 at 10 steps. Paper: 50.0 / 53.3 / 56.7. At 0 steps LatentMAS gives 50. | "Smaller models (e.g., Qwen3-4B / 8B) can sometimes struggle with stability during latent reasoning." They recommend Qwen3-14B. |
| #22 (open) | 4B, 20 steps, batch size 1: LatentMAS far below baseline on MBPP+ and HumanEval+. With the authors' command (no steps flag, batch size 30): MBPP+ 71.6, HumanEval+ 79.2, close to the paper. Baseline and TextMAS stay above the paper. | Gave the commands without `--latent_steps`. |
| #25 (open), #26 (open PR) | Batch size above 1 with steps above 0 lowers accuracy. Cause: right padding. | No fix merged. |
| #39 (closed) | Baselines are above Table 1 on many cells: GSM8K 4B 87.6 against 82.4, GSM8K 14B 90.1 against 83.7, MedQA 8B 59.0 against 53.0. With the authors' commands LatentMAS matches the paper, and the baseline then matches or beats it. A second user gets 65.4% from the authors' 4B GSM8K command. | Gave commands. No explanation of the baseline gap. |
| #41 (open), #29 (open) | 4B: baseline beats LatentMAS on GSM8K and MedQA. MedQA baseline is 66-70% against the paper's 47.7. | Gave commands without `--latent_steps`. |
| #50 (open) | 8B GSM8K: baseline 76, LatentMAS 41 at 10 steps, batch size 20. | No reply. |
| #27 (closed) | 8B ARC-Challenge with the authors' command: baseline 93.3, LatentMAS 92.8. | Gave the command. |
| #53, #55 (open) | Which latent-step value did each cell of Tables 1 and 2 use? | No reply. |

The pattern across reporters: with latent steps above 0, 4B and 8B lose accuracy. With the authors' own commands the steps are 0, the LatentMAS number comes out near the paper, and the baseline comes out as high or higher.

### Final numbers from this morning's audit jobs (MBPP+ 4B, first 84 questions, batch size 1 unless noted)

| Run | Correct | % | Wins vs baseline bs 1 | Losses | p | No code block | Syntax error | Repetition loop |
|---|---|---|---|---|---|---|---|---|
| Baseline, bs 1 | 61/84 | 72.6 | - | - | - | 9 | 0 | 0 |
| Baseline, bs 15 (old run) | 59/84 | 70.2 | 3 | 5 | 0.73 | 9 | 0 | 0 |
| TextMAS, bs 8 (old run) | 60/84 | 71.4 | 5 | 6 | 1.00 | 4 | 0 | 0 |
| LatentMAS, 0 steps, `--think` | 63/84 | 75.0 | 5 | 3 | 0.73 | 9 | 0 | 0 |
| LatentMAS, 20 steps, `--think` | 40/84 | 47.6 | 2 | 23 | <0.0001 | 11 | 17 | 8 |
| LatentMAS, 20 steps, no `--think` | 47/84 | 56.0 | 3 | 17 | 0.003 | 14 | 9 | 3 |

GSM8K 4B, first 100 questions, baseline at batch size 1: 96/100. Batch size 8: 54/56 on the same prefix. The padding bug (earlier audit) is real but costs little accuracy here.

### What can and cannot be reproduced from the released code

| Paper claim, 4B and 8B sequential | Status | Evidence |
|---|---|---|
| LatentMAS accuracy | Reproduces, at 0 latent steps | GSM8K 4B 90.0 vs 88.2; 8B 90.0 vs 93.8 (vs 90.8 in their log); MBPP+ 4B 75.0 vs 77.4 on the same 84 |
| TextMAS accuracy | Reproduces on the one full-set run we have | MBPP+ 4B 72.2 vs 69.8, n=378 |
| Single-agent accuracy | Does not reproduce; ours is higher | GSM8K 4B 90.0 vs 82.4; 8B 90.0 vs 81.1; MBPP+ 4B 69.0 vs 63.5 (n=378) |
| LatentMAS beats Single | Does not reproduce on tested cells | paired ties on GSM8K; MBPP+ 4B 0-step 63 vs 61 of 84 |
| Latent steps add accuracy at 4B | Opposite | 20 steps: -25 points on MBPP+, 23 losses to 2 wins |

### Limits of Test 1

- 100 questions, one seed. SE is 3 points, so differences under about 6 points are not resolved. The paper reports the mean of three runs on 1319 questions.
- The 4B LatentMAS run resumed its last 20 questions from a checkpoint, which restarts the sampling seed. The 8B baseline has 80 questions: its batch job hit the time limit and the in-session resume did not start before this session's deadline.
- GSM8K only. MBPP+ 4B rests on this morning's 84-question runs.
- Batched runs have the right-padding artifact (earlier audit). The README commands use batch size 20, so they share it.

### GPU cost of this round

New batch jobs: 59007402 (13.1 min), 59007403 (15.1 min), 59007404 (15.1 min): 43 GPU-minutes on A100s. On top of that, about 28 minutes on this interactive session's A30, which was already allocated. Jobs stopped at their limits because model loading took about 4 minutes each, which I had not planned for. Cancelled: 59001448 (the independent-implementation check; it would not have finished).

### Corrections to earlier sections of this report

- "Reproduction fails" (Phase 3 and the audit): that was true for 20 latent steps. With the README commands the LatentMAS column reproduces. The failure is specific to turning the latent steps on.
- "The paper's 4B LatentMAS number is consistent with a 0-step run, not proof": the authors' own log now gives direct evidence: 92.9% per-question agreement with our 0-step run.
- "Why is our GSM8K baseline so far above the paper?": the random sample gives 90.0, lower than the 96.0 prefix, so the prefix was easier. The remaining gap to 82.4 is shared by other users' full-set runs (87.6 in issue #39).

### What to run next

1. Nothing more on 4B/8B GSM8K at small n. A full-set cell costs about 2 GPU-hours on an A100 with the Hugging Face path.
2. Score the authors' other logs (MedQA, ARC, HumanEval+) against ours question by question. It is free. `hpc3/compare_readme_repro.py` shows how.
3. If the question is whether the latent channel carries information, test at Qwen3-14B. The authors say the channel works there, and their 14B commands use 10-20 steps. The 4B/8B cells cannot answer it, because the paper's numbers for them come from 0 steps.

## Audit, 2026-11-06 morning (superseded by the section above)

You challenged the three-way GSM8K tie and asked for a code audit. This section corrects the summary below. The test numbers are interim: jobs 59001443-59001448 were still running when this was written. Their limit is 1 h 30 min, and they save to `results/audit_tests/`.

**Corrections to what I reported earlier**

1. **"Baseline and TextMAS results are trustworthy" was wrong.** They ran at batch size above 1, which has a real bug (finding 1). The word "trustworthy" below does not apply to them until bs=1 runs replace them.
2. **"Early questions are easier" for GSM8K was an untested guess.** It holds for MBPP+ (77.8 on the first 45, 69.0 on all 378). I have no full GSM8K run, so I do not know why our baseline is far above the paper's 82.4.
3. **"Test `--think` first" was the wrong priority.** The new run shows `--think` does not explain the MBPP+ drop (finding 3).
4. **"Too easy to separate the methods" skipped the check you asked for.** The tie is now explained (finding 4).

**Finding 1: right padding breaks batched baseline and TextMAS runs. Confirmed bug, small effect on scores so far.**
The tokenizer pads on the right, and `generate()` starts each row from its last column. For a shorter row that column is a pad token. On the first 8 GSM8K prompts, the model alone gives p(`<think>`) = 1.000. Padded rows gave as low as 0.000, and the measured first token matched the actual output on every row. Across the saved runs, unpadded rows open normally 7/7 (GSM8K) and 26/26 (MBPP+). Padded rows do so 1/49 and 9/352. In the bs=1 rerun, 50/50 open with `<think>`. The prompt's statement that batching these methods is safe is not correct. `--pad_fix` exists but was not used.

Effect on accuracy so far is small: GSM8K baseline 49/50 at bs=1 against 48/50 at bs=8, MBPP+ baseline 22/27 at both. So this bug does not explain the reproduction failure.

**Finding 2: the GSM8K scorer compares strings.** `57.00` against gold `57` is marked wrong. Numeric rescoring moves LatentMAS from 81/87 to 83/87, and changes neither baseline nor TextMAS. This is upstream behaviour (`methods/*`, `normalize_answer`), not yet fixed because a fix changes results.

**Finding 3: the latent steps cause the 4B MBPP+ drop.** Same first 27 questions:

| Run | Correct | Against baseline bs=1 |
|---|---|---|
| Baseline, bs=1 | 22/27 (81.5) | reference |
| Baseline, bs=15, old | 22/27 (81.5) | 2 wins, 2 losses |
| TextMAS, bs=8, old | 21/27 (77.8) | 1 win, 2 losses |
| LatentMAS steps 0, `--think` | 23/27 (85.2) | 2 wins, 1 loss |
| LatentMAS steps 20, `--think` | 14/27 (51.9) | 1 win, 9 losses |
| LatentMAS steps 20, no `--think` | 13/27 (48.1) | 0 wins, 9 losses |

Steps 0 gives the Judger no cache. Steps 0 has also reached 52/68 (76.5). With or without `--think`, steps 20 falls to about half. n=27 is small, but the 9-to-1 split agrees with the 23-to-4 split on 84 questions. This fits your earlier finding that the released 14B MBPP+ log used steps 0: the paper's 4B LatentMAS number (73.5) is close to what steps 0 scores here. That is consistent with the paper's cell being a steps-0 run. It is not proof.

**Finding 4: the tie is not a bug.** The runs are independent: different prompts, agents and outputs. All three miss the same 2 of the first 40 questions with the same answers. Q12 expects "year 13" where 90 / 7.5 = 12. Q37 says he sells "all his toys" but expects 2 sets left. Past question 40, LatentMAS does differ: 2 real misses plus the 2 scorer errors.

**Checked and found correct** (real 4B model, greedy): generating from a cache matches one concatenated pass; feeding an embedding matches feeding the same token ID (difference 0.0); `hidden_states[-1]` is post-norm; repo changes from upstream `9a9e4d3` other than the checkpoint code, ablation flags and vLLM guard are behind `--pad_fix`, which was off; `utils.py` and `prompts.py` are identical to upstream.

**Not explained yet:** why latent steps hurt MBPP+ but not GSM8K (LatentMAS 83/87 after rescoring), and why our GSM8K baseline is so far above the paper's. Job 59001448 (`experiments.exp0_controls`, an independent implementation) tests whether the MBPP+ drop is specific to upstream's code. It writes results only at the end, so it may produce nothing if the limit ends it first.

**What these runs do not settle:** all are small prefixes, one seed. No full test set was run.

## Summary (written before the audit; see corrections above)

**GPU time:** about 4.3 GPU-hours of actual work, slightly over your 4-hour target: batch job 2.67 h, audit pilot 1.28 h, smoke test at most 0.37 h. Your interactive session also held an A30 for its full 4 hours, including idle time. All of this was on the class account.

**What already existed:** 3 of the existing configurations are trustworthy: 4B MBPP+ baseline (69.0), 4B MBPP+ TextMAS (72.2), and 14B GPQA baseline (39.9). No LatentMAS result is, because none has the method on, batch size 1 and the full test set.

**Reproduction (partial, on subsets):**

| Model, task | n | Baseline | TextMAS | LatentMAS (m20) | Paper (B / T / L, full set) |
|---|---|---|---|---|---|
| 4B MBPP+ | 84 | 70.2 | 71.4 | **47.6** | 63.5 / 69.8 / 73.5 |
| 4B GSM8K | 40 | 95.0 | 95.0 | 95.0 | 82.4 / 89.8 / 88.2 |
| 8B MBPP+ | 14 | 78.6 | not run | 92.9 | 64.8 / 69.5 / 74.6 |

4B MBPP+ LatentMAS does not reproduce with these settings: 23 losses and 4 wins against the baseline, p = 0.0003. 4B GSM8K cannot separate the methods at this size. 8B has too few questions for a verdict.

**Audit pilot (GSM8K, 200 questions, log p(gold)):**
- Question-specific content: 4B +0.397 [-0.757, 1.495], 8B -0.572 [-1.480, 0.393]. Both intervals include zero.
- Any real message: 4B +1.874 [0.736, 3.030], 8B +3.010 [2.050, 3.943].
- Agent prompts: 4B -12.279, 8B -9.115. Both intervals exclude zero.
- Mean cos(ridge, identity): 4B 1.0000, 8B 0.0274. Your prediction holds.
- Patching: every swap interval includes zero. Every pause interval excludes zero.
- Logit lens: gold's median rank reaches about 5-10 by step 8 on 4B. It stays in the thousands on 8B.

**Cleanup:** branch `cleanup-ste`, 6 commits, not merged or pushed. Tests pass and the toy numbers are identical before and after. Upstream code is unchanged apart from comments, docstrings and unused imports. Details in Phase 5.

**Problems hit:**

| Problem | Log | Fix or status |
|---|---|---|
| `hpc3_bundle.zip` missing at first | - | You uploaded it |
| `repro-fixes.patch` does not apply to the `architecture/` layout | - | Applied the missing parts by hand, `87c80de` |
| `inventory_results.py` mislabeled old logs as steps 0, bs 20 | `results/inventory.csv` (wrong), `results/inventory_corrected.csv` (use this) | Fixed. `inventory_corrected.csv` predates the sample-count fix, so resumed `.log` rows can show too small an n. Checkpoint rows are right. |
| Inventory script skipped everything when given an absolute path, because `/data` is on the skip list | - | Fixed |
| `reproduce.sh` had no checkpoints and overwrote logs on rerun | - | Fixed |
| Your plan's 12 jobs would take about 70 GPU-hours | - | Replaced by the 4-hour plan you asked for |
| 8B `exp5` out of memory at batch 8 on A30 | `results/audit_pilot/exp5_content_audit_Qwen3-8B.log` | One retry at batch 2 succeeded |
| Runs overran their `timeout` limits by 6-10 minutes, so 8B got little time and realign never ran | `logs/latentmas-4h.58983470.out` | Open. Cause not checked. Exit codes in that log are also wrong: `$(date)` resets `$?`. |
| A30s ran MBPP+ at about 60 s per question, twice my estimate | same | Smaller samples than planned |
| Earlier notes said the `gpu` partition rejects this account | - | Outdated for the class account. Notes corrected. |

**Jobs still running:** none from this work. 58983470 ended at its time limit. Only your interactive session 58981325 remains, and it ends at 05:27.

**What to run next, in order:**
1. Decide about `--think`. A 4B MBPP+ run with steps 20 and no `--think` would test the largest open question, about 40-60 minutes for 84 questions.
2. Finish 8B MBPP+: the three runs resume from their checkpoints with `sbatch hpc3/run_4h_plan.sh`. Raise the `-t` limit and the per-run limits first, because finished work is kept but the limits are short. Note the script would also resume the 4B runs, so comment those out if you only want 8B.
3. Review and merge `cleanup-ste`: `git checkout cleanup-ste`, then `git merge main` to bring in this report and `compare_subset.py`. Nothing has been pushed.

## Setup before Phase 1

- `~/hpc3_bundle.zip` unzipped to `/tmp/hpc3_bundle`. `diff -r` showed no shared files, so no repo change needed merging. The repo was clean at `9fc054e`. The bundle went in as `25ec502`, unchanged.
- The GPU budget changed during the session: you asked for about 4 GPU-hours in place of about 20, with as much reproduced as that allows. Phase 3 therefore runs on subsets. See Phase 3.

## Phase 1: existing results

Source: `results/inventory_corrected.csv`. `results/inventory.csv` is the first pass and is wrong for most old logs. Old logs have no argument header, so the script read them as `latent_steps=0, bs=20`. I fixed `hpc3/inventory_results.py` to read settings from the file name and the `[Latent Steps]` marker.

Duplicates are merged, one row per configuration. SE = sqrt(p(1-p)/n), using our accuracy. "Trust" means: method on (steps above 0), batch size 1 for latent runs, finished, full test set.

The Gap columns on partial rows are not a fair comparison. The paper reports the full test set, and the first questions are easier: baseline scores 77.8 on the first 45 MBPP+ questions and 69.0 on all 378.

| Model | Task | Method | Latent steps | Realign | Batch | Think | n | Ours % | Paper % | Gap | Gap in SE | Trust |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | gpqa | baseline | 0 | F | 4 | 1 | 198 | 39.9 |  |  |  | yes |
| Qwen3-14B | gpqa | latent_mas | 10 | F | 4 | 1 | 198 | 49.0 |  |  |  | no: padding leak (batch>1) |
| Qwen3-14B | gpqa | text_mas | 0 | F | 4 | 1 | 68 | 57.4 |  |  |  | no: partial 68/198 |
| Qwen3-14B | humanevalplus | muscle | 10 | F | 20 | 1 | 164 | 84.8 |  |  |  | no: padding leak (batch>1), old code |
| Qwen3-14B | mbppplus | latent_mas | 0 | F | 1 | 1 | 45 | 93.3 | 75.7 | +17.6 | +4.7 | no: method off (steps 0), partial 45/378 |
| Qwen3-14B | mbppplus | latent_mas | 0 | F | 15 | 1 | 378 | 76.2 | 75.7 | +0.5 | +0.2 | no: method off (steps 0), padding leak (batch>1) |
| Qwen3-14B | mbppplus | latent_mas | 10 | F | 1 | 1 | 45 | 91.1 | 75.7 | +15.4 | +3.6 | no: partial 45/378 |
| Qwen3-4B | gsm8k | baseline | 0 | F | 15 | 1 | 60 | 96.7 | 82.4 | +14.3 | +6.2 | no: partial 60/1319 |
| Qwen3-4B | gsm8k | latent_mas | 10 | F | 1 | 1 | 30 | 100.0 | 88.2 | +11.8 |  | no: partial 30/1319 |
| Qwen3-4B | gsm8k | latent_mas | 10 | F | 4 | 1 | 8 | 100.0 | 88.2 | +11.8 |  | no: padding leak (batch>1), partial 8/1319 |
| Qwen3-4B | gsm8k | text_mas | 0 | F | 15 | 1 | 30 | 93.3 | 89.8 | +3.5 | +0.8 | no: partial 30/1319 |
| Qwen3-4B | mbppplus | baseline | 0 | F | 15 | 1 | 378 | 69.0 | 63.5 | +5.5 | +2.3 | yes |
| Qwen3-4B | mbppplus | latent_mas | 0 | F | 1 | 1 | 45 | 82.2 | 73.5 | +8.7 | +1.5 | no: method off (steps 0), partial 45/378 |
| Qwen3-4B | mbppplus | latent_mas | 0 | F | 15 | 1 | 120 | 70.0 | 73.5 | -3.5 | -0.8 | no: method off (steps 0), padding leak (batch>1), partial 120/378 |
| Qwen3-4B | mbppplus | latent_mas | 1 | F | 1 | 1 | 45 | 77.8 | 73.5 | +4.3 | +0.7 | no: partial 45/378 |
| Qwen3-4B | mbppplus | latent_mas | 2 | F | 1 | 1 | 45 | 77.8 | 73.5 | +4.3 | +0.7 | no: partial 45/378 |
| Qwen3-4B | mbppplus | latent_mas | 5 | F | 1 | 1 | 45 | 64.4 | 73.5 | -9.1 | -1.3 | no: partial 45/378 |
| Qwen3-4B | mbppplus | latent_mas | 10 | F | 1 | 0 | 45 | 77.8 | 73.5 | +4.3 | +0.7 | no: partial 45/378 |
| Qwen3-4B | mbppplus | latent_mas | 10 | F | 1 | 1 | 45 | 55.6 | 73.5 | -17.9 | -2.4 | no: partial 45/378 |
| Qwen3-4B | mbppplus | latent_mas | 10 | T | 1 | 1 | 45 | 51.1 | 73.5 | -22.4 | -3.0 | no: partial 45/378 |
| Qwen3-4B | mbppplus | latent_mas | 10 | F | 8 | 1 | 378 | 28.0 | 73.5 | -45.5 | -19.7 | no: padding leak (batch>1) |
| Qwen3-4B | mbppplus | latent_mas | 10 | F | 15 | 1 | 378 | 22.8 | 73.5 | -50.7 | -23.5 | no: padding leak (batch>1) |
| Qwen3-4B | mbppplus | latent_mas | 20 | F | 1 | 1 | 45 | 53.3 | 73.5 | -20.2 | -2.7 | no: partial 45/378 |
| Qwen3-4B | mbppplus | latent_mas | 40 | T | 1 | 0 | 45 | 73.3 | 73.5 | -0.2 | -0.0 | no: partial 45/378 |
| Qwen3-4B | mbppplus | latent_mas | 40 | F | 1 | 1 | 45 | 53.3 | 73.5 | -20.2 | -2.7 | no: partial 45/378 |
| Qwen3-4B | mbppplus | text_mas | 0 | F | 8 | 1 | 378 | 72.2 | 69.8 | +2.4 | +1.0 | yes |
| Qwen3-4B | mbppplus | text_mas | 0 | F | 15 | 1 | 15 | 80.0 | 69.8 | +10.2 | +1.0 | no: partial 15/378 |

The two rows from `example_logs/` are the authors' released runs: 14B MBPP+ (steps 0) and 14B HumanEval+ (old code, `muscle`).

**Trustworthy:** only three rows meet all the criteria.
- 4B MBPP+ TextMAS: 72.2 against the paper's 69.8, gap +1.0 SE, so it matches.
- 4B MBPP+ baseline: 69.0 against the paper's 63.5, gap +2.3 SE, so it does not match. Ours is higher.
- 14B GPQA baseline: 39.9. The inventory table has no paper number for GPQA.

**Not trustworthy:** every LatentMAS row. No run has the method on, batch size 1 and the full test set. The closest are the 45-question sweep rows.

**Missing:** a full batch-size-1 LatentMAS run on any task; every full GSM8K run; all Qwen3-8B results; the hierarchical layout on 4B and 8B; HumanEval+, MedQA and ARC on 4B and 8B.

**Worth a look:** at 4B, steps 10, batch 1, the same 45 questions score 55.6 with `--think` and 77.8 without it. Steps 40 with realign and no `--think` scores 73.3. n=45 is small (SE about 7), and I did not test this further. `reproduce.sh` always passes `--think`.

**Jobs at the start:** `squeue` showed only your interactive session, 58981325.

## Phase 2: environment

| Check | Result |
|---|---|
| Env | `/pub/ynagaraj/envs/latentmas`: torch 2.11.0+cu128, transformers 4.57.1, CUDA available. A second env in miniconda has torch 2.13.0. Run with `PYTHONNOUSERSITE=1`. |
| GPU | NVIDIA A30 24 GB, node hpc3-gpu-k54-05 |
| `hpc3/repro-fixes.patch` | Does not apply: it targets the old flat layout, and the root files are now symlinks into `architecture/`. The vLLM guard and the Qwen3-8B choice were already in. I applied the rest by hand in `87c80de`: removed `import pdb` and the unused matplotlib import, pinned transformers. |
| `python -m experiments.test_experiments` | All 8 checks passed |
| Smoke test, 4 GSM8K questions, 512 tokens, 4B | baseline, TextMAS, LatentMAS m20, LatentMAS m20 realign all finished, 3/4 correct each. Logs in `results/smoke/`. |
| `HF_HOME` | The models were already in `/pub/ynagaraj/hf`, so I made `/pub/ynagaraj/hf_cache` a symlink to it. Nothing was downloaded twice. |

## Phase 3: reduced reproduction

Batch job 58983470, one A30, 02:30 to 05:10, ended by its 2 h 40 min time limit (`logs/latentmas-4h.58983470.out`). Script: `hpc3/run_4h_plan.sh`. Logs and checkpoints: `results/repro_4h/`. Comparison: `hpc3/compare_subset.py`, saved output `results/repro_4h/compare_subset.txt`.

All rows: sequential layout. LatentMAS ran with `--think`, batch size 1, max 4096 new tokens. No run reached its target size, so none has a final JSON summary. The numbers come from the checkpoints.

**Each table row uses the first n questions, and the paper number is for the full test set, so "gap" is only a rough check.** Early questions are easier than the full set (GSM8K baseline here is 95.0 against 82.4), so positive gaps do not count as matches. Method-against-method on the same questions is the fair comparison.

| Model | Task | Layout | Method | Latent steps | Realign | n | Ours % | Paper % | Gap | Gap in SE |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | MBPP+ | sequential | baseline (old run, cut to n) | - | - | 84 | 70.2 | 63.5 | +6.7 | +1.4 |
| Qwen3-4B | MBPP+ | sequential | TextMAS (old run, cut to n) | - | - | 84 | 71.4 | 69.8 | +1.6 | +0.3 |
| Qwen3-4B | MBPP+ | sequential | LatentMAS | 20 | off | 84 | 47.6 | 73.5 | -25.9 | -4.7 |
| Qwen3-4B | GSM8K | sequential | baseline | - | - | 40 | 95.0 | 82.4 | +12.6 | +3.7 |
| Qwen3-4B | GSM8K | sequential | TextMAS | - | - | 40 | 95.0 | 89.8 | +5.2 | +1.5 |
| Qwen3-4B | GSM8K | sequential | LatentMAS | 20 | off | 40 | 95.0 | 88.2 | +6.8 | +2.0 |
| Qwen3-8B | MBPP+ | sequential | baseline | - | - | 14 | 78.6 | 64.8 | +13.8 | +1.3 |
| Qwen3-8B | MBPP+ | sequential | LatentMAS | 20 | off | 14 | 92.9 | 74.6 | +18.3 | +2.7 |

Full-set references for 4B MBPP+ from Phase 1: baseline 69.0 (n=378), TextMAS 72.2 (n=378). Runs with more questions than the shared n: 4B GSM8K baseline 54/56, LatentMAS 81/87 (54/56 on the baseline's 56). 8B MBPP+ baseline 13/16.

**Paired results.** 4B MBPP+, LatentMAS against baseline: 4 wins, 23 losses, exact McNemar p = 0.0003. 4B GSM8K: all three methods got the same questions right on the first 40. 8B MBPP+: LatentMAS 2 wins, 0 losses on 14.

**Causes checked for the 4B MBPP+ miss (over 3 SE):**
- Latent steps: 20, so the method was on. Batch size: 1. transformers: 4.57.1.
- Truncated thinking: 11.9% of LatentMAS Judger outputs have no `</think>`, against 13.1% for the baseline, so this does not explain the gap.
- Not ruled out: `--think`. The Phase 1 sweep at 45 questions scored higher without it (see "Worth a look"). I did not test this here.
- The first 45 questions came from your earlier run with the same settings (24/45). The 39 new ones scored 16/39, so the two parts agree.

**Verdict, Qwen3-4B (partial).** MBPP+ LatentMAS does not reproduce with these settings. It scored 47.6 on the first 84 questions, 22.6 points below baseline on the same questions, while the paper reports LatentMAS 10 points above baseline. Baseline and TextMAS are close to the paper's numbers on this task (Phase 1). On GSM8K the first 40 questions are too easy to separate the methods: all three score 95.0. That result neither confirms nor contradicts the paper.

**Verdict, Qwen3-8B (no verdict).** n=14 is too small. LatentMAS scored 13/14 and baseline 11/14 on the same questions, which is not evidence either way. The realign run did not start.

**Not run:** the rest of every test set, 8B realign, 8B GSM8K, 8B TextMAS, latent steps 40, the hierarchical layout, and HumanEval+, MedQA and ARC.

## Phase 4: content audit pilot

GSM8K, first 200 questions, 20 latent steps, teacher-forced, so no generation and no accuracy (the accuracy columns are `nan`). All numbers are log p(gold). Intervals are 95%. Raw output is in `results/audit_pilot/`.

The 8B `exp5` run ran out of memory at `--batch_size 8` on the 24 GB A30 (`exp5_content_audit_Qwen3-8B.log`). I reran it once at `--batch_size 2` (`exp5_content_audit_Qwen3-8B_bs2.log`). All other runs used batch size 8. `exp6` used its default identity mapping.

**Decomposition, X minus Y, paired on the same questions**

| Part | 4B | 8B | Interval includes zero? |
|---|---|---|---|
| Agent prompts | -12.279 [-13.196, -11.366] | -9.115 [-10.033, -8.136] | no, no |
| Extra positions / compute | +0.701 [0.404, 1.008] | -1.558 [-1.832, -1.279] | no, no |
| Any real message | +1.874 [0.736, 3.030] | +3.010 [2.050, 3.943] | no, no |
| Question-specific content | +0.397 [-0.757, 1.495] | -0.572 [-1.480, 0.393] | **yes, yes** |
| Total | -9.308 [-10.326, -8.278] | -8.236 [-9.087, -7.334] | no, no |

Question-specific content is not distinguishable from zero on either model. The extra-positions term has opposite signs on the two models.

**Mean log p(gold) and gold top-1 rate by condition**

| Condition | 4B log p | 4B top-1 | 8B log p | 8B top-1 |
|---|---|---|---|---|
| judger_alone | -8.940 | 0.420 | -11.190 | 0.480 |
| prompts_only | -21.219 | 0.520 | -20.305 | 0.525 |
| pause | -20.518 | 0.505 | -21.863 | 0.535 |
| swapped | -18.644 | 0.470 | -18.853 | 0.510 |
| latent_identity | -18.247 | 0.495 | -19.426 | 0.540 |
| latent_ridge | -18.197 | 0.515 | -17.858 | 0.515 |
| latent_random | -19.785 | 0.520 | -20.175 | 0.535 |

**Mean cos(ridge, identity):** 4B 1.0000, 8B 0.0274. Your prediction holds: about 1.0 for 4B, and clearly lower for 8B.

**Per-agent patching.** Delta = log p(gold | real) - log p(gold | patched). Above zero means the patch hurt.

| Patch | 4B | 8B |
|---|---|---|
| no_memory | -9.308 [-10.326, -8.278] | -8.203 [-9.052, -7.287] |
| swap_planner | 0.076 [-0.644, 0.773] (includes 0) | 0.001 [-0.053, 0.060] (includes 0) |
| pause_planner | 1.057 [0.465, 1.637] | 1.024 [0.852, 1.189] |
| swap_critic | 0.288 [-0.257, 0.805] (includes 0) | 0.013 [-0.039, 0.067] (includes 0) |
| pause_critic | 0.539 [0.032, 1.062] | 0.679 [0.558, 0.796] |
| swap_refiner | 0.255 [-0.265, 0.780] (includes 0) | -0.016 [-0.064, 0.030] (includes 0) |
| pause_refiner | 1.108 [0.688, 1.592] | 0.996 [0.889, 1.104] |
| swap_all | -0.164 [-0.858, 0.529] (includes 0) | -0.008 [-0.063, 0.048] (includes 0) |

Every swap interval includes zero on both models. Every pause interval excludes zero on both models. The 4B pause_critic lower bound (0.032) is close to zero.

**Logit lens: median rank of the gold answer's first token (1 = top choice)**

| Agent | Model | Step 0 | Step 4 | Step 8 | Step 12 | Step 16 | Share of steps in top 10 |
|---|---|---|---|---|---|---|---|
| Planner | 4B | 151329 | 29 | 9 | 5 | 5 | 0.494 |
| Critic | 4B | 151544 | 7 | 8 | 9 | 9 | 0.490 |
| Refiner | 4B | 151586 | 8 | 9 | 10 | 10 | 0.441 |
| Planner | 8B | 24898 | 8133 | 5562 | 20682 | 4589 | 0.000 |
| Critic | 8B | 92956 | 13281 | 14772 | 13058 | 14010 | 0.000 |
| Refiner | 8B | 93491 | 13122 | 21156 | 14327 | 13421 | 0.000 |

These are one seed on one task, with 200 questions. I did not run `analyze_audit.py`, because it combines several seeds.

## Phase 5: code cleanup

Branch `cleanup-ste`, 6 commits on top of main, not pushed and not merged. 46 files changed.

I did this work in a temporary worktree, because the GPU jobs were running from the main checkout. A subagent made the edits, and I reran the key checks myself.

**Checks, all at the branch head**

| Check | Result |
|---|---|
| `python -m experiments.test_experiments` | All 8 checks passed, before and after |
| exp5 toy run, before and after | Identical. Two "before" runs also matched each other, so the comparison can detect a change. |
| Upstream code with docstrings removed, compared by syntax tree | 8 of 10 files identical. `models.py` lost unused `import os` and `import csv`. `data.py` lost a duplicate mid-file import. |
| Upstream modules import, `run.py --help` runs | Pass |
| Agent's wider check: every experiments script on the tiny model, compared with main | Agent reports 30 outputs byte-identical. I did not rerun this one. |

Limits of these checks: the tiny model scores 0 accuracy, so accuracy arithmetic is tested only on zeros. No `run.py` path ran end to end after the cleanup, because `run.py` accepts only the real Qwen models. For upstream files the evidence is the syntax-tree comparison. One visible change: `--help` descriptions come from `__doc__`, so that text changed. Flags did not.

**What changed, by file**

- `experiments/latent_mapping.py`, `latent_pipeline.py`, `eval_utils.py`, `audit_utils.py`: docstrings, type hints, shape comments, verb function names (for example `get_input_embedding_matrix`, `build_judger_prompts`, `score_gold_logprob`, `compute_mcnemar_exact_p`).
- New shared helpers replace duplicated code: `iterate_batches`, `select_donor_questions`, `collect_correct_by_condition`, `add_adaptive_step_arguments`, `get_model_tag`, `generate_with_args`, `repeat_for_each_agent`, `compute_mean_token_norm`, `compute_bootstrap_ci`. `build_memory_per_agent` moved to `latent_pipeline.py`.
- `experiments/exp0`-`exp6`, `analyze_audit.py`, `check_padding.py`, `test_experiments.py`, `make_tiny_model.py`: rewritten docs, type hints, clearer names.
- `experiments/README.md`: one line per script with what it compares and its command. The claim about `research-directions.md` is gone, because that file does not exist.
- `hpc3/inventory_results.py`: docs, hints, renames. `reproduce.sh`: renames. `run_4h_plan.sh`: one comment. `run_l40s.sh`: comments, corrected usage path.
- `yajat/`: all 7 `.py` files got docstrings, comments and type hints, with no logic edits. All 9 `.sh`/`.sbatch` files got comment edits only.
- `architecture/` (`run.py`, `models.py`, `data.py`, `prompts.py`, `utils.py`, `methods/*`, `__init__.py`): docstrings and comments, commented-out debug lines removed, open items marked `# TODO:`. No renames and no logic changes. Prompt strings are unchanged.

**Stale after cleanup:** line numbers such as `models.py:241` in `yajat/README.md` and `hpc3/CLAUDE_CODE_PROMPT.md` no longer point to the same lines.

**Possible bugs found while reading, left alone.** These come from the subagent's reading. Most were not run, and fixing them could change results, so they need your decision.

*Your code*
- exp0-exp4 and exp6 leave the seed out of the output file name, and exp0-exp4 also leave out the model, so reruns overwrite each other. This is a problem for the 3-seed runs the README recommends.
- `analyze_audit.py` does not guard against mixing models or tasks.
- `load_items` with `--toy --max_samples -1` drops the last item.
- A fresh clone fails: the scripts need the root symlinks, which are gitignored and described as temporary.

*Upstream*
- `load_checkpoint` stops at a truncated line, but that line stays in the file, so a second resume can lose records. A resume also restarts the seed, so sampling differs from an uninterrupted run.
- AIME scoring catches only `ValueError`, so `pred=None` stops the run.
- `--text_mas_context_length` defaults to -1, which drops the last character of the context.
- GSM8K scoring is strict: `\boxed{1,000}` becomes `1`, and `18.00` does not equal `18`.
- `_truncate_past` leaves RoPE keys in place while new tokens reuse earlier positions (`--sequential_info_only`, `--latent_only`).
- `--use_vllm --pad_fix` is not fixed, and `run_batch_vllm` has further problems. Your rules already forbid `--use_vllm`.


