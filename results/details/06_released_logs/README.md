# 06 The authors' released logs

**Status: CONFIRMED that the 14B MBPP+ log used 0 latent steps, and that the current code discards the shared cache at 0 steps. SUGGESTIVE for the 4B cell. NOT PROVEN that this was an error.** The authors state that 0 steps is part of the method.

Upstream ships two logs in `architecture/example_logs/`. Each one matches one cell of the paper's tables.

| Log | Latent steps | Accuracy, log (paper) | Generated tokens per question, log (paper) | Paper cell |
|---|---|---|---|---|
| `qwen3_14b_mbppplus_sequential.txt` | 0 in all 1,134 entries | 76.2 (75.7) | 1,621.2 (1,621) | Table 1, 14B, MBPP+, sequential |
| `qwen3_14b_humanevalplus_hierarchical.txt` | 10 in all 492 entries | 84.8 (86.6) | 1,512.3 (1,512) | Table 2, 14B, HumanEval+, hierarchical |

Facts that follow:

1. **The 14B MBPP+ LatentMAS cell ran with 0 latent steps.** The log also shows `use_vllm: False`. In the released code, `latent_mas.py` passes no cache to the Judger at 0 steps. `kv_cache_trace_latent_steps_0.log` shows the Judger inheriting 0 positions, and `kv_cache_trace_latent_steps_10.log` shows 566.
2. **The token column counts generated tokens only.** Prompt tokens add 785 (MBPP+) and 1,183 (HumanEval+) per question, and LatentMAS reads four prompts.
3. **The logs come from older code.** They print `method: muscle` and have no `latent_steps` argument. That older code is not public, so nobody outside the authors can confirm that it also discarded the cache.

## Was the 4B MBPP+ cell also run with the method off?

Probably, but this is an inference. The authors did not release that log.

| Evidence | Method off (ours) | Method on (ours) | Paper's cell |
|---|---|---|---|
| Accuracy, first 60 questions | 45/60 (75.0%), same as the single agent | 19/60 (31.7%), batch size 15 | 73.5% on 378 |
| Accuracy, batch size 1 | 63/84 (75.0%) | 40/84 (47.6%), steps 20 | 73.5% on 378 |
| Generated tokens per question | 1,337 (60 questions) | 1,527 to 1,578 | 1,339 |
| Time relative to the single agent, first 60 questions, batch size 15 | 1.12× (1,809 against 1,612 s) | 1.58× (2,551 against 1,612 s) | 1.10× (577 against 523) |

The paper also gives 1,339 tokens for this cell in both the sequential and the hierarchical table. Other LatentMAS cells differ between the two tables.

Limit: the no-`--think` runs at 10 and 40 steps also score near 75% (`../02_latent_steps_and_think_tag/`), so accuracy alone cannot separate "method off" from "method on without `--think`". Tokens and time still point to "method off".

The HumanEval+ log shows that the authors did use the method for some cells.

## What the paper says against what the log shows

| Paper ([v3](https://arxiv.org/html/2511.20639v3)) | Released material |
|---|---|
| Latent steps m ∈ {0, 10, 20, 40, 80}, with tuning. Tables 1 and 2 do not give m per cell. | 0 is therefore allowed, but the reader cannot see that a Table 1 cell used it. |
| Section 5: accuracy peaks at 40 to 80 steps, and the authors adopt a budget in this range. | The Table 1 14B MBPP+ log uses 0 steps. |
| `W_a` is computed once per run and reused at all inference steps. | `--latent_space_realign` is off by default. |
| Means over three independent runs. | The log's token count matches the table to the digit, while its accuracy (76.2) differs from the table (75.7). |

## What the authors said

| Source | Statement |
|---|---|
| [Issue 10](https://github.com/Gen-Verse/LatentMAS/issues/10) | Their command for Qwen3-4B MBPP+ has no `--latent_steps`, so it runs at the default of 0. This is the strongest evidence for the 4B cell. |
| [Issue 10](https://github.com/Gen-Verse/LatentMAS/issues/10), [issue 12](https://github.com/Gen-Verse/LatentMAS/issues/12) | 0 steps is "fully consistent with our method". They say the cache can still pass through the vLLM backend, so it does not degenerate to a single model. The 14B log says `use_vllm: False`. |
| [Issue 12](https://github.com/Gen-Verse/LatentMAS/issues/12) | Non-zero steps were used elsewhere: 10 on HumanEval+ and MedQA, 20 or 40 on 14B GSM8K. |
| [Issue 12](https://github.com/Gen-Verse/LatentMAS/issues/12) | Qwen3-4B and 8B "can sometimes struggle with stability" in latent reasoning. The tables do not show this. |
| [Issue 39](https://github.com/Gen-Verse/LatentMAS/issues/39), [issue 41](https://github.com/Gen-Verse/LatentMAS/issues/41) | Their 2026 commands for Qwen3-4B GSM8K and MedQA also have no `--latent_steps`. |

## What other people measured

| Source | Setting | Result |
|---|---|---|
| [Issue 22](https://github.com/Gen-Verse/LatentMAS/issues/22) | Qwen3-4B MBPP+, batch size 1 | Steps 0, 10, 20: 71.4, 62.1, 61.3. Single agent 72.7. This agrees with our direction. |
| [Issue 10](https://github.com/Gen-Verse/LatentMAS/issues/10) | Cache against concatenated prompts at 0 steps | Identical outputs, reported by the user. One latent step dropped accuracy to 32.8. |
| [Issue 8](https://github.com/Gen-Verse/LatentMAS/issues/8) | Code reading | At 0 steps the Hugging Face path discards the cache. |
| [Issue 49](https://github.com/Gen-Verse/LatentMAS/issues/49) | Llama 3B and 8B, four tasks | Steps 0 is best on every task, and accuracy falls with more steps. |
| [Issue 20](https://github.com/Gen-Verse/LatentMAS/issues/20) | Qwen3-14B GSM8K, 30 steps, batch size 5, realign | LatentMAS 76.4, single agent 93.2. Batch size above 1 has the padding bug. |
| [Issue 25](https://github.com/Gen-Verse/LatentMAS/issues/25) | Padding | Batch size 1 against 20 on MBPP+: 62.1 against 21.9 |
| [Issue 39](https://github.com/Gen-Verse/LatentMAS/issues/39) | Authors' commands on 14B | **Counter-evidence.** The user reports LatentMAS numbers close to Table 1. Their concern is that baselines come out higher than the paper's. |
| [Issues 29](https://github.com/Gen-Verse/LatentMAS/issues/29), [41](https://github.com/Gen-Verse/LatentMAS/issues/41) | Qwen3-4B MedQA single agent | About 66 to 70, against 47.7 in the paper |
| [Deshpande and Lu](https://icml.cc/virtual/2026/69292), ICML 2026 workshop | Ablations on three benchmarks | Identity in place of the trained `W_a` caused no significant drop. Removing cross-agent transfer was comparable or better on math and science. Latent beat single agent on code, and text beat latent. |

## What would settle it

1. Ask the authors for the value of m in every Table 1 and Table 2 cell ([issue 53](https://github.com/Gen-Verse/LatentMAS/issues/53) has asked since July 2026, and [issue 55](https://github.com/Gen-Verse/LatentMAS/issues/55) since September), and for the code that made the released logs.
2. Run Qwen3-14B MBPP+ on all 378 questions at batch size 1: single agent, and LatentMAS at 0, 10 and 40 steps. The paper gives 68.5 and 75.7. If the single agent scores near 76, the 14B gain in that cell comes from a low baseline.

## Reading the paper's speed column

The paper's headline speedup is against TextMAS. Against the single agent, LatentMAS is slower in 29 of the 36 cells of `architecture/assets/main_table1.png` and `main_table2.png`.

## Evidence

- `released_logs_token_counts.json` and `.log`: output of `results/test_runs/analysis/count_tokens_from_logs.py`
- `kv_cache_trace_latent_steps_0.log`, `kv_cache_trace_latent_steps_10.log`: output of `results/test_runs/analysis/trace_kv_cache.py`
