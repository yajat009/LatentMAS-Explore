# Test runs

For what each file and setting means, see [the guide](../../architecture/GUIDE.md).

This folder holds every script that produced a number in `results/`. Run all of them from the repo root.

- `results/test_runs/*.py` are the audit and follow-up experiments. They use their own pipeline, `latent_pipeline.py`, which fixes the batching bugs.
- `analysis/` reads finished runs. It needs no GPU, except `trace_kv_cache.py`.
- `cluster/` submits runs on HPC3. See [cluster/README.md](cluster/README.md).

Raw output goes to `results/raw/`, which git ignores. To update the evidence in the repo, run:

```bash
python results/test_runs/analysis/export_compact_results.py
python results/test_runs/analysis/recompute_result_tables.py
```

## Experiment scripts

| Script | What it compares | Status |
|---|---|---|
| `check_batch_padding.py` | Batched results against one-at-a-time results, which must match | Passes on the tiny model |
| `run_memory_controls.py` | Judger alone, latent memory (identity and ridge `W_a`), and memory from a different question | One MBPP+ run, not finished |
| `run_adaptive_latent_steps.py` | Fixed step counts against stopping when the latent state converges | Not run on a real model |
| `run_content_audit.py` | Judger alone, prompts only, pause tokens, swapped memory, and real memory, by log p(gold) | Pilot done: `results/details/08_content_audit/` |
| `run_message_decoding.py` | Real latent vectors against swapped or paused ones, and a logit lens on every step | Pilot done: `results/details/08_content_audit/` |
| `summarize_content_audit.py` | Content-audit runs across seeds: means, intervals and Holm-corrected p-values | Needs several seeds |

Example commands:

```bash
python -m results.test_runs.test_experiments          # CPU checks on a tiny model, about 30 s
python -m results.test_runs.run_content_audit --task gsm8k --max_samples 200 --latent_steps 20 --batch_size 8 --no_generate
python -m results.test_runs.run_message_decoding --task gsm8k --max_samples 200 --latent_steps 20 --batch_size 8
```

Common flags: `--model_name` (default `Qwen/Qwen3-4B`), `--batch_size`, `--latent_steps`, `--max_new_tokens`,
`--temperature`, `--seed`, `--toy` (three built-in questions, no download). Run each script with 3 seeds before you trust a difference.
Only the content-audit and message-decoding output names include the model, so other scripts overwrite an earlier run of the same task.

Shared modules:

| Module | What it holds |
|---|---|
| `latent_pipeline.py` | Model loading, `SharedMemory` (the KV cache with its mask and segments), the latent agents, text generation |
| `latent_mapping.py` | Maps a hidden state to an input embedding: `identity`, `ridge`, `soft`, `mix` |
| `audit_utils.py` | Pause, random-rotation and replay mappings, forced step counts, teacher-forced scoring, bootstrap intervals |
| `eval_utils.py` | Common flags, data loading, batching, scoring, and the paired McNemar report |
| `make_tiny_model.py` | Builds a random Qwen3 model of about 200k parameters for the CPU checks |

`latent_pipeline.py` differs from `architecture/` in these ways:

- It pads on the left, so latent steps start from a real token.
- It carries the real attention mask across agents.
- Rotary position embeddings count only real tokens, so batch size does not change answers.
- At `latent_steps=0` the Judger still gets the prompt cache. The explicit no-memory control is `judger_alone`.
- It has the sequential layout only. It has no TextMAS baseline.

## Analysis scripts

| Script | What it does |
|---|---|
| `export_compact_results.py` | Writes the CSV files in `results/data/` from the raw checkpoints |
| `recompute_result_tables.py` | Recomputes every paired comparison in `results/` from those CSV files |
| `compare_methods_from_checkpoints.py` | Compares runs on their shared questions, with tokens and time, while the runs are still going |
| `count_tokens_from_logs.py` | Counts tokens per agent from a finished log, because `run.py` does not report them |
| `measure_realign_matrix.py` | Computes `W_a` from the model weights and measures its distance from the identity |
| `trace_kv_cache.py` | Prints the KV cache size after each agent, to show the mechanism |
| `test_truncate_past.py` | Tests `_truncate_past` on every cache format |

## Old names

Earlier notes use these names, and the folder `experiments/` is now `results/test_runs/`. The commit history keeps the old files.

| Old name | New name |
|---|---|
| `experiments/exp0_controls.py` | `run_memory_controls.py` |
| `experiments/exp2_adaptive_steps.py` | `run_adaptive_latent_steps.py` |
| `experiments/exp5_content_audit.py` | `run_content_audit.py` |
| `experiments/exp6_decode_messages.py` | `run_message_decoding.py` |
| `experiments/analyze_audit.py` | `summarize_content_audit.py` |
| `experiments/check_padding.py` | `check_batch_padding.py` |
| `yajat/analyze.py` | `analysis/count_tokens_from_logs.py` |
| `yajat/check_realign.py` | `analysis/measure_realign_matrix.py` |
| `yajat/compare_arms.py` | `analysis/compare_methods_from_checkpoints.py` |
| `yajat/trace_pipeline.py` | `analysis/trace_kv_cache.py` |
| `yajat/run_latentmas.sbatch` | `cluster/run_one_configuration.sbatch` |
| `yajat/submit_ls_sweep.sh` | `cluster/submit_latent_steps_sweep.sh` |
| `hpc3/reproduce.sh` | `cluster/reproduce_paper_tables.sh` |
| `hpc3/run_l40s.sh` | `cluster/run_on_free_l40s.sh` |
