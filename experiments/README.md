# Experiments

This folder holds one small script per research question. The scripts share a minimal LatentMAS
pipeline (`latent_pipeline.py`). The pipeline reuses the repo's prompts, datasets and scorers and
fixes the batching bugs. Every condition therefore runs on the same questions, and the scripts
compare conditions with a paired test.

| Script | What it compares | Command |
|---|---|---|
| `check_padding.py` | Batched results against one-at-a-time results, which must match | `python -m experiments.check_padding --task gsm8k --max_samples 8 --batch_size 8` |
| `exp0_controls.py` | Judger alone, latent memory (identity), latent memory (ridge W_a), and memory from a different question | `python -m experiments.exp0_controls --task gsm8k --max_samples 300 --latent_steps 20 --think` |
| `exp1_soft_embeddings.py` | Identity, ridge, soft embeddings at several temperatures, and a ridge/soft mix, with drift diagnostics | `python -m experiments.exp1_soft_embeddings --task gsm8k --max_samples 200 --latent_steps 20 --think` |
| `exp2_adaptive_steps.py` | Fixed step counts against stopping per row when the latent state converges or the model is confident | `python -m experiments.exp2_adaptive_steps --task gsm8k --max_samples 100 --think` |
| `exp3_trimmed_memory.py` | Full memory, Refiner only, Refiner latent steps only, and all latent steps only | `python -m experiments.exp3_trimmed_memory --task mbppplus --max_samples 150 --latent_steps 20 --think` |
| `exp4_execution_check.py` | Judger with and without one retry after a visible-test failure, with and without latent memory | `python -m experiments.exp4_execution_check --task mbppplus --max_samples 378 --latent_steps 20 --think` |
| `exp5_content_audit.py` | Judger alone, prompts only, pause tokens, swapped memory, and real memory (identity, ridge, random rotation), by accuracy and teacher-forced log p(gold) | `python -m experiments.exp5_content_audit --task gsm8k --max_samples -1 --latent_steps 20 --think --seed 1` |
| `exp6_decode_messages.py` | Real latent vectors against one agent's vectors swapped or paused (activation patching), with a logit lens on every latent step | `python -m experiments.exp6_decode_messages --task gsm8k --max_samples 300 --latent_steps 20 --mapping identity` |
| `analyze_audit.py` | exp5 conditions across seeds: mean and seed range, paired bootstrap intervals, and Holm-corrected permutation p-values | `python -m experiments.analyze_audit experiments/results/exp5_Qwen3-4B_gsm8k_m20_seed*.jsonl` |

Shared modules:

- `latent_pipeline.py` loads the model and holds `SharedMemory` (KV cache, which means key/value cache, with its mask and named segments), the latent agents and text generation.
- `audit_utils.py` holds the pause, random-rotation, replay and recording mappings, forced step counts, teacher-forced gold scoring and bootstrap intervals.
- `latent_mapping.py` maps a hidden state to an input embedding (`identity`, `ridge`, `soft`, `mix`).
- `eval_utils.py` holds the common flags, data loading, batching, scoring (the rules of `methods/latent_mas.py`) and the paired McNemar report.
- `make_tiny_model.py` builds a random 200k-parameter Qwen3, and `test_experiments.py` runs the CPU checks on it.

## What the pipeline fixes

- The tokenizer pads prompts on the left, so latent steps start from a real token and not from a pad.
- The pipeline carries the real attention mask across agents. The repo uses all-ones masks.
- RoPE (rotary position embedding) positions count only real tokens, so the batch size does not change answers.
- At `latent_steps=0` the Judger still gets the agents' prompt cache. `judger_alone` is the explicit no-memory control.
- Trimming the memory keeps the original positions, and new tokens continue after the untrimmed length.

## Setup

The scripts need `torch`, `transformers` (4.56 or newer), `datasets` and `tokenizers`.
They do not import `models.py` or `methods/`, so they do not need vLLM or matplotlib.

Run everything from the repo root with `python -m`:

```bash
python -m experiments.test_experiments          # CPU, about 30 s
python -m experiments.check_padding --task gsm8k --max_samples 8 --batch_size 8
```

## Running without a GPU

On a laptop, use a small real model and short answers. The scripts pick the CPU, or `mps` on an
Apple-silicon Mac, automatically. Qwen3-0.6B and Qwen3-1.7B have tied embeddings like Qwen3-4B,
so `exp1_soft_embeddings.py` behaves the same way:

```bash
python -m experiments.exp0_controls --model_name Qwen/Qwen3-0.6B --task gsm8k \
    --max_samples 20 --latent_steps 10 --max_new_tokens 256
```

Leave out `--think` here, because thinking traces run to thousands of tokens and dominate the
run time. Accuracies at this size are low and noisy. Use these runs to debug and compare
conditions, not to quote numbers.

For real numbers on a free GPU, use Google Colab (T4, 16 GB) or Kaggle (T4 or P100, about
30 GPU-hours a week). Qwen3-4B fits in 16 GB, and the T4 runs in fp16 automatically:

```bash
git clone https://github.com/Gen-Verse/LatentMAS && cd LatentMAS
# upload experiments.zip, then:
unzip experiments.zip && pip install -q transformers datasets accelerate
python -m experiments.exp0_controls --task gsm8k --max_samples 100 --latent_steps 20
```

## Flags and output

The commands in the table above suit Qwen3-4B on one 24 GB GPU.

Common flags: `--model_name` (default `Qwen/Qwen3-4B`), `--batch_size`, `--latent_steps`,
`--max_new_tokens`, `--temperature`, `--seed`, `--toy` (three built-in questions, no download).
Each run prints the accuracy per condition, with wins and losses against the baseline and an
exact McNemar p-value. Each run also writes per-question rows to `experiments/results/*.jsonl`.
Run each script with 3 seeds before you trust a difference.

## Not included yet

- The compute-matched TextMAS baseline.
- The hierarchical prompt layout. The pipeline uses the sequential layout.
