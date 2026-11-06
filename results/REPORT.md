# LatentMAS on HPC3: inventory, reduced reproduction, audit pilot, cleanup

## Summary

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


