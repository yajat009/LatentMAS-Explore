# Architecture

This folder holds the LatentMAS method code from
[Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS) (commit `9a9e4d3`), and this document.
The paper is [Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639).
This document says what the method does, where each step is in the code, and how this copy differs from upstream.
[GUIDE.md](GUIDE.md) explains every word, setting, benchmark and file in plain language.

## What the method does

LatentMAS is a multi-agent system. Agents pass internal model state to each other and do not write text.
The paper claims that this keeps accuracy, generates fewer tokens, and runs faster than agents that write text.

Four agents run in order. All of them are the same model with different prompts:

```
Planner  -> reads its prompt, takes m latent steps, writes no text   \
Critic   -> reads its prompt, takes m latent steps, writes no text    > all add to one shared KV cache
Refiner  -> reads its prompt, takes m latent steps, writes no text   /
Judger   -> reads its prompt on top of the shared KV cache, then writes the answer
```

Three terms carry the method:

- **Hidden state (h).** The vector that a transformer holds for one position. The last-layer hidden state
  is the vector that the model uses to score the next token. It has 2,560 numbers in Qwen3-4B and 5,120 in Qwen3-14B.
- **KV cache (key/value cache).** The stored attention keys and values of every position that the model
  has read. It is the only channel between agents in LatentMAS.
- **Latent step.** One forward pass in which the input is the previous hidden state, not a token embedding.
  Each step adds one position to the KV cache. The flag `--latent_steps` sets m.

One latent step works in this order:

1. The model takes the last-layer hidden state `h` of the last position.
2. `h` is mapped to an input vector `e = rescale(h · W_a)`. `W_a` maps output space to input space.
3. `e` is rescaled to the average length of the token embeddings.
4. The model runs one forward pass on `e`, which adds a position to the cache and gives the next `h`.

`W_a` is a ridge-regression solution from the model's output and input embedding matrices.
It is used only with `--latent_space_realign`. Without that flag, `W_a` is the identity.

The repo compares three methods. All of them use the same questions and scoring:

| Method | File | What it does |
|---|---|---|
| Single agent (`baseline`) | `methods/baseline.py` | One prompt and one generation. |
| TextMAS (`text_mas`) | `methods/text_mas.py` | The same four agents. Each agent writes text, and the next agent reads it. |
| LatentMAS (`latent_mas`) | `methods/latent_mas.py` | The four agents above. Only the Judger calls `generate()`. |

## Code map

| File | What it holds |
|---|---|
| `run.py` | Command-line entry point: parses flags, loads data, runs batches, prints the final JSON line |
| `models.py` | `ModelWrapper`: loading, `generate_latent_batch` (the latent loop), `generate_text_batch`, the `W_a` computation |
| `methods/latent_mas.py` | `LatentMASMethod`: runs the agents in order and hands the cache to the Judger |
| `methods/__init__.py` | `default_agents`: the four agent roles |
| `prompts.py` | Agent prompts for the sequential and hierarchical layouts |
| `data.py` | Dataset loaders. Only MedQA is local (`../data/medqa.json`). The others download from Hugging Face. |
| `utils.py` | Answer extraction, answer normalization, and the code-execution timeout |
| `assets/` | The paper's figures, result tables and slides, unchanged from upstream |
| `example_logs/` | The authors' two released run logs, unchanged from upstream |

## How to run it

Run from the repo root, because `data.py` reads `./data/medqa.json` by a relative path.

```bash
pip install -r requirements.txt
python architecture/run.py --method latent_mas --model_name Qwen/Qwen3-4B --task mbppplus \
    --prompt sequential --latent_steps 20 --generate_bs 1 --max_samples 20
```

Check these settings before you trust a run:

| Setting | Why it matters |
|---|---|
| `--latent_steps` | The default is 0. At 0, `latent_mas.py` passes no cache to the Judger, so the method is off. |
| `--latent_space_realign` | Off by default, so `W_a` is the identity. Qwen3-4B has tied embeddings, so `W_a` is the identity there anyway. Qwen3-14B does not. |
| `--generate_bs` | Above 1, the tokenizer pads on the right, and this breaks all three methods. See `results/details/04_padding_bug/`. Use 1, or pass `--pad_fix`. |
| `--think` | This flag does not turn reasoning on or off. It only adds `<think>` to the end of the Judger prompt. Without it, the model writes `<think>` itself. It has a measured effect. See `results/details/02_latent_steps_and_think_tag/`. |
| `--use_vllm` | Do not use it. It mis-scores code tasks. |
| `transformers` | Keep version 4.57.1. Version 5.x breaks `models.py:_past_length`. |

## How this copy differs from upstream

To see every difference, run:

```bash
git fetch upstream
git diff upstream/main:models.py HEAD:architecture/models.py
```

Files moved from the repo root into `architecture/`, and `assets/` and `example_logs/` moved with them.
`data/` stayed at the root, because `data.py` depends on that path. `utils.py` and `prompts.py` have comment changes only.

| Change | File | Effect on results |
|---|---|---|
| Docstrings and comments, unused imports and commented-out debug lines removed | all | None. A syntax-tree comparison confirmed this. |
| vLLM and matplotlib imports made optional, `import pdb` removed | `methods/latent_mas.py`, `models.py` | None. Upstream crashes without them. |
| `Qwen3-8B` added to the `--model_name` choices (upstream listed 4B twice) | `run.py` | None |
| Dataset name `gsm8k` changed to `openai/gsm8k` | `data.py` | None. It is the same dataset. |
| `_truncate_past` ported to the current cache API | `methods/latent_mas.py` | None by default. It runs only with the next flags. |
| New `--latent_only` and `--sequential_info_only` flags (the method code read them, but `run.py` never defined them) | `run.py` | Off by default |
| New `--checkpoint`: saves each batch and resumes after a preempted job | `run.py` | None. A resumed run restarts the random seed. |
| New `--pad_fix`: left padding, real attention masks, corrected output slice | `run.py`, `models.py`, `methods/latent_mas.py` | Off by default. No completed run used it. |
| `transformers==4.57.1` pinned | `requirements.txt` | Keeps the code working |

## Known upstream problems, left unchanged

A fix to any of these changes results, so the code keeps the upstream behavior.

- GSM8K scoring compares strings, so `57.00` does not match gold `57`. See `results/details/07_gsm8k/`.
- The MedQA loader can reuse the previous item's answer letter (`data.py`).
- AIME scoring catches only `ValueError`, so a missing prediction stops the run.
- `--text_mas_context_length` defaults to -1, which drops the last character of the context.
- `_truncate_past` keeps rotary position information in the kept keys while new tokens reuse earlier positions. This affects only the two ablation flags.
- `load_checkpoint` stops at a cut-off line but leaves that line in the file, so a second resume can lose records.
