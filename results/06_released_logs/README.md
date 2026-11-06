# 06 The authors' released logs

**Status: CONFIRMED for the 14B MBPP+ log. SUGGESTIVE for the 4B MBPP+ cell, which has no released log.**

Upstream ships two logs in `architecture/example_logs/`. Each one matches one cell of the paper's tables.

| Log | Latent steps | Accuracy, log (paper) | Generated tokens per question, log (paper) | Paper cell |
|---|---|---|---|---|
| `qwen3_14b_mbppplus_sequential.txt` | 0 in all 1,134 entries | 76.2 (75.7) | 1,621.2 (1,621) | Table 1, 14B, MBPP+, sequential |
| `qwen3_14b_humanevalplus_hierarchical.txt` | 10 in all 492 entries | 84.8 (86.6) | 1,512.3 (1,512) | Table 2, 14B, HumanEval+, hierarchical |

Facts that follow:

1. **The 14B MBPP+ LatentMAS cell ran with the method off.** At 0 steps, `latent_mas.py` passes no cache to the Judger. `kv_cache_trace_latent_steps_0.log` shows the Judger inheriting 0 positions, and `kv_cache_trace_latent_steps_10.log` shows 566.
2. **The token column counts generated tokens only.** Prompt tokens add 785 (MBPP+) and 1,183 (HumanEval+) per question, and LatentMAS reads four prompts.
3. **The logs come from older code.** They print `method: muscle` and have no `latent_steps` argument.

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

## Reading the paper's speed column

The paper's headline speedup is against TextMAS. Against the single agent, LatentMAS is slower in 29 of the 36 cells of `architecture/assets/main_table1.png` and `main_table2.png`.

## Evidence

- `released_logs_token_counts.json` and `.log`: output of `results/test_runs/analysis/count_tokens_from_logs.py`
- `kv_cache_trace_latent_steps_0.log`, `kv_cache_trace_latent_steps_10.log`: output of `results/test_runs/analysis/trace_kv_cache.py`
