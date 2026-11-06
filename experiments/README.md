# Experiments

One small script per research direction from `research-directions.md`. They share a
minimal LatentMAS pipeline (`latent_pipeline.py`) that reuses the repo's prompts,
datasets and scorers but fixes the batching bugs, so every condition is compared
on the same questions with a paired test.

| File | Direction | What it compares |
|---|---|---|
| `check_padding.py` | 0. padding fix | Batched vs one-at-a-time results (should match) |
| `exp0_controls.py` | 0. controls | Judger alone, latent (identity), latent (ridge W_a), memory from a different question |
| `exp1_soft_embeddings.py` | 1. soft thoughts | identity, ridge, soft embeddings at several temperatures, ridge/soft mix, plus drift diagnostics |
| `exp2_adaptive_steps.py` | 2. adaptive steps | Fixed m vs stopping per row when the latent state converges or the model is confident |
| `exp3_trimmed_memory.py` | 3. trim memory | Full memory vs Refiner only vs latent steps only |
| `exp4_execution_check.py` | 4. code repair | Judger with and without one retry on visible-test failure, with and without latent memory |
| `exp5_content_audit.py` | audit | Judger alone, prompts only, pause tokens, swapped memory, real memory (identity / ridge / random rotation); optional adaptive steps with per-row compute-matched controls; accuracy and teacher-forced log p(gold) |
| `exp6_decode_messages.py` | decoding | Logit lens on every latent step, and per-agent activation patching (swap or pause one agent's latent vectors) |
| `analyze_audit.py` | audit | Combines exp5 seeds: mean and seed range, paired bootstrap intervals, Holm-corrected permutation p-values |

Shared pieces:

- `latent_pipeline.py`: loading, the `SharedMemory` (KV cache + mask + named segments), latent agents, text generation.
- `audit_utils.py`: pause, random-rotation and replay mappings, per-agent forced step counts, teacher-forced gold scoring.
- `latent_mapping.py`: hidden state to input embedding (`identity`, `ridge`, `soft`, `mix`).
- `eval_utils.py`: data loading, scoring (same rules as `methods/latent_mas.py`), paired McNemar report.
- `make_tiny_model.py` and `test_experiments.py`: a random 200k-parameter Qwen3 and CPU checks.

## What the pipeline fixes

- Prompts are padded on the left, so latent steps start from a real token, not a pad.
- The real attention mask is carried across agents (the repo uses all-ones masks).
- RoPE positions count only real tokens, so batch size no longer changes answers.
- At `latent_steps=0` the Judger still gets the agents' prompt cache. `judger_alone` is the explicit no-memory control.
- Trimming memory keeps original positions, and new tokens continue after the untrimmed length.

## Setup

Needs `torch`, `transformers` (4.56 or newer), `datasets` and `tokenizers`.
It does not import `models.py` or `methods/`, so vLLM and matplotlib are not needed.

Run everything from the repo root with `python -m`:

```bash
python -m experiments.test_experiments          # CPU, about 30 s
python -m experiments.check_padding --task gsm8k --max_samples 8 --batch_size 8
```

## Running without a GPU

On a laptop (CPU, or `mps` on an Apple-silicon Mac, picked automatically), use a small real
model and short answers. Qwen3-0.6B and Qwen3-1.7B have tied embeddings like Qwen3-4B, so
direction 1 behaves the same way:

```bash
python -m experiments.exp0_controls --model_name Qwen/Qwen3-0.6B --task gsm8k \
    --max_samples 20 --latent_steps 10 --max_new_tokens 256
```

Leave out `--think` here: thinking traces run to thousands of tokens and dominate the run time.
Expect accuracies to be low and noisy at this size. Use these runs to debug and compare
conditions, not to quote numbers.

For real numbers on a free GPU, use Google Colab (T4, 16 GB) or Kaggle (T4 or P100, about
30 GPU-hours a week). Qwen3-4B fits in 16 GB, and the T4 runs in fp16 automatically:

```bash
git clone https://github.com/Gen-Verse/LatentMAS && cd LatentMAS
# upload experiments.zip, then:
unzip experiments.zip && pip install -q transformers datasets accelerate
python -m experiments.exp0_controls --task gsm8k --max_samples 100 --latent_steps 20
```

## Suggested first runs (Qwen3-4B, one 24 GB GPU)

```bash
python -m experiments.exp0_controls        --task gsm8k    --max_samples 300 --latent_steps 20 --think
python -m experiments.exp1_soft_embeddings --task gsm8k    --max_samples 200 --latent_steps 20 --think
python -m experiments.exp2_adaptive_steps  --task gsm8k    --max_samples 100 --think
python -m experiments.exp3_trimmed_memory  --task mbppplus --max_samples 150 --latent_steps 20 --think
python -m experiments.exp4_execution_check --task mbppplus --max_samples 378 --latent_steps 20 --think
```

Common flags: `--model_name` (default `Qwen/Qwen3-4B`), `--batch_size`, `--latent_steps`,
`--max_new_tokens`, `--temperature`, `--seed`, `--toy` (three built-in questions, no download).
Each run prints accuracy per condition with wins/losses against the baseline and an exact
McNemar p-value, and writes per-question rows to `experiments/results/*.jsonl`.
Run each with 3 seeds before trusting a difference.

## Not included yet

- The compute-matched TextMAS baseline (control e in direction 0).
- The hierarchical prompt layout; the pipeline uses the sequential one.
