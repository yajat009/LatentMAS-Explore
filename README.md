# LatentMAS-Explore

A reproduction study of **LatentMAS** — *Latent Collaboration in Multi-Agent Systems*
([arXiv:2511.20639](https://arxiv.org/abs/2511.20639),
[Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS)).

LatentMAS claims that multi-agent collaboration can move out of token space and into
the model's KV cache, buying large token savings and a **×3.7 wall-clock speedup at
no accuracy cost**. This fork runs that claim on real hardware and reports what
survives.

**Headline: at Qwen3-4B on MBPP+, it does not reproduce.** The speedup is ×2.1 in
wall clock and ×1.4 in generated tokens, not ×3.7, and accuracy collapses by ~50
points. Full numbers below.

---

## How the method works

Four agents — Planner → Critic → Refiner → Judger. Under `latent_mas` the first three
**never emit a token**. Each one's prompt is forward-passed on top of the running
`past_key_values`, then `latent_steps` autoregressive steps run in embedding space:
the last layer's final hidden state is fed straight back in as `inputs_embeds`, with
no sampling and no detokenization. The KV cache *is* the entire inter-agent channel.
Only the judger calls `generate()`, inheriting all of it.

The token savings fall directly out of this — three of four agents produce zero
output tokens. `models.py:generate_latent_batch` is ~40 lines and is the whole paper.

```
latent_steps=10                        latent_steps=0
  Planner  KV   0 ->  166                Planner  KV   0 ->  156
  Critic   KV 166 ->  373                Critic   KV 156 ->  353
  Refiner  KV 373 ->  566                Refiner  KV 353 ->  536
  Judger   inherits 566  <- channel      Judger   inherits   0  <- DISCARDED
```

---

## Changes made to this repo

Everything in `repro/` is new. Upstream files were changed only where noted.

### 1. Fixes required to make the repo run at all

| file | change |
|---|---|
| `methods/latent_mas.py` | Made the vLLM import conditional. It was unconditional, but `requirements.txt` has no vllm — import-time crash. |
| `run.py` | `--model_name` choices listed `Qwen3-4B` twice and omitted `Qwen3-8B`. |
| `data.py` | Bare dataset id `gsm8k` → `openai/gsm8k`; modern `huggingface_hub` rejects the legacy form. |
| `methods/latent_mas.py` | Ported `_truncate_past` off the legacy `Cache` API, removed in transformers 5.x. |

Applied by `repro/02_patch_blockers.sh`. Original `run.py` kept at `repro/run.py.orig`.

### 2. New capability in `run.py`

- **`--checkpoint PATH`** — JSONL appended after every batch, replayed on startup as
  a prefix skip, carrying elapsed time forward and reporting `resumed_from`. Every
  GPU partition available here is preemptible and preemption is *cancel*, not
  requeue; four jobs died mid-run before this existed, two past halfway.
- **`--latent_only` / `--sequential_info_only`** — read by `methods/latent_mas.py`
  but never defined in `run.py`, so they were permanently `False`. Now reachable,
  making the paper's own ablations runnable.
- **`--pad_fix`** — opt-in, four changes that are only correct together (see
  finding 5). Defaults **off**: right padding is what the authors ran, so the
  unfixed path stays the reproduction path and both are reachable from one binary.

### 3. Measurement the repo lacks

`run.py` reports accuracy and wall-clock and **no token counts** — but the central
claim is a token claim.

| script | what it does |
|---|---|
| `repro/analyze.py` | Re-tokenizes every agent's `[Output]` block from a stdout log |
| `repro/compare_arms.py` | Reads the checkpoint JSONL instead, so it works **mid-run**; compares arms on the prefix they have all finished |
| `repro/analyze_sweep.py` | Builds the `latent_steps` sweep table and classifies its shape |
| `repro/check_realign.py` | Rebuilds the realign least-squares solve from safetensors — no GPU, never materializes a 14B model |
| `repro/compare_padfix.py` | A/B for `--pad_fix` |
| `repro/trace_pipeline.py` | Narrates the KV cache growing across agents — fastest way to see the mechanism |

### 4. Cluster harness

`repro/run_latentmas.sbatch` (self-resubmitting on SIGTERM, guarded by "did the
checkpoint grow this attempt?" so a real crash fails once, not twenty times), plus
`00_setup_env.sh`, `01_fetch_assets.sh`, and the `submit_*.sh` launchers.

---

## Results

Qwen3-4B · MBPP+ · sequential · `max_new_tokens=4096` · `think=1` · one L40S · seed 42.

### Main comparison — 336 problems

The three arms that ran to full length (`repro/results/arm_comparison.json`):

| arm | accuracy | sec/problem | gen tok/prob | prompt tok/prob |
|---|---|---|---|---|
| `baseline` (single agent, bs=15) | 0.694 | 29.1 | 1416 | 245 |
| `latent_mas` ls=10, bs=15 | **0.226** | 44.7 | 1551 | 1102 |
| `text_mas` bs=8 | **0.723** | 95.4 | 2158 | 5148 |

**TextMAS beats LatentMAS by ~50 accuracy points.** LatentMAS is cheaper — ×2.13 in
wall clock, ×1.39 in generated tokens — but neither figure is the claimed ×3.7.

**LatentMAS also loses to the plain single-agent baseline on every axis at once:**
lower accuracy, 1.5× slower, 4.5× the prefill. There is no dimension on which it
wins here. TextMAS beats the baseline by only 2.9 points for 3.3× the wall clock.

> **The wall-clock ratio is confounded.** LatentMAS ran 15 problems at a time,
> TextMAS only 8 (TextMAS OOMs at 15). A bigger batch lowers time-per-problem on its
> own, because fixed GPU overhead is shared across more work — so part of the ×2.13
> is batch size, not method. The **token** ratio (×1.39) is batch-independent and is
> the trustworthy one. A matched `latent_mas` bs=8 run is queued to decontaminate this.

### `latent_steps` sweep — 45 problems, bs=1

| ls | 0 | 1 | 2 | 5 | 10 | 20 | 40 | 10 +realign |
|---|---|---|---|---|---|---|---|---|
| accuracy | 0.822 | 0.778 | 0.778 | 0.644 | 0.556 | 0.533 | 0.533 | 0.511 |

**Monotonic, no cliff.** The 0.289 decline is spread evenly across steps, which
points at the latent channel being genuinely harmful at this scale rather than
broken by a single bug. More latent thinking is strictly worse, every step of the way.

### Five-arm comparison — 45 problems

| arm | accuracy | sec/problem | gen tok/prob | prompt tok/prob |
|---|---|---|---|---|
| `baseline` | 0.778 | 25.7 | 1237 | 227 |
| `text_mas` bs=8 | **0.800** | 75.8 | 2203 | 4757 |
| `latent_mas` ls=0 | 0.756 | 30.5 | 1335 | 1029 |
| `latent_mas` ls=10, bs=15 | 0.378 | 41.6 | 926 | 1029 |
| `latent_mas` ls=10, bs=1 | 0.556 | 30.3 | 1527 | 1029 |

The last two rows isolate the padding bug: it costs ~18 points (0.556 → 0.378). But
bs=1 is still 22 points under baseline, so **the latent channel itself costs more
than the padding bug does**, and fixing padding will not rescue the arm.

---

## Findings, explained

**1. The two released logs are two paper cells, confirmed to the token.**
Re-tokenizing them reproduces the paper's Token column exactly — 1621.2 vs 1621
(Table 1, 14B/MBPP+) and 1512.3 vs 1512 (Table 2, 14B/HumanEval+). That is not
coincidence; these are the runs behind those cells, which pins down two things the
paper does not state.

**2. The Token metric counts generated text only.** Prefill is omitted: 785
tokens/problem for the MBPP+ run, 1183 for HumanEval+. LatentMAS forward-passes
*four* agent prompts to build its cache where a single agent passes one, and none of
that ~4× prefill appears in the Token column. State this carefully rather than as a
refutation — prefill is parallel and far cheaper per token than sequential decode,
so `generated + prompt` is not apples-to-apples either. The paper measures decode
tokens, the dominant cost, which is legitimate. But the omitted prefill is exactly
what resurfaces in the two places the savings don't materialize: wall clock and peak
memory.

**3. The published MBPP+/14B cell was produced with the latent channel switched off.**
That log carries `latent_steps: 0` in all 1134 trace entries, and in today's code
`past_for_decoding = past_kv if self.latent_steps > 0 else None` — the judger gets
`None` and the accumulated cache is discarded, degenerating to a single-agent call
with three wasted forward passes. **`--latent_steps` defaults to 0; always pass it
explicitly.** Caveat: both released logs print `method: muscle` and carry no
`latent_steps` field at all, so they came from an earlier code revision, and what
that revision did at ls=0 is unknown.

**4. LatentMAS costs peak memory, and the efficiency framing omits it.** Same GPU,
same batch, same data, Qwen3-4B on one 24 GB A30 at bs=15: `latent_mas` OOMs where
`baseline` and `text_mas` both run. The judger decodes against the concatenated KV
of all three prior agents for every one of `max_new_tokens` steps. This is intrinsic,
not an implementation wart — it is the same property the token savings come from.

**5. The latent thought is seeded from a PAD token at any `--generate_bs > 1`.**
`generate_latent_batch` reads `hidden_states[-1][:, -1, :]` — position −1 of a
*right-padded* batch — then builds `latent_mask = torch.ones(...)`, un-masking every
pad already in the cache. Measured: 93.3% of sequences seed from a pad, 35.8% of KV
positions are pad. Re-tokenizing the *released* HumanEval+ log gives 94.4% and
32.7%, so the published numbers were produced in this state too. Invisible at bs=1,
which is why smoke tests miss it.

**6. `--latent_space_realign` is a mathematical no-op on Qwen3-4B.** The flag
least-squares-solves `W_out @ M ≈ W_in` so a hidden state (output space) can be fed
back as an input embedding; without it the code replaces `M` with the identity.
Qwen3-4B has **tied** embeddings, so `W_out` *is* `W_in` and the solve returns the
identity anyway — measured `M_vs_I_relative_fro = 0.0`, cosine 1.0. The flag cannot
be tested at 4B at all. On Qwen3-14B embeddings are untied and `M` is far from
identity (relative Frobenius 1.05, mean cosine 0.004 — near-orthogonal), so every
14B run *without* the flag feeds output-space vectors into the input-embedding slot
with only a norm rescale.

**7. The ×3.7 speedup is measured against TextMAS, not a single agent.** The paper's
own Table 1 agrees in direction that LatentMAS is slower than a single agent
(4B/MBPP+: 577 vs 523). Any summary saying "LatentMAS is faster" without naming the
comparison arm is wrong.

---

## Why these numbers may differ from the paper

Ranked by how much of the gap each could plausibly explain:

1. **Model scale.** This is Qwen3-4B; the paper's cells are 14B (and larger). The
   latent channel asks the model to consume its own hidden states as input
   embeddings — plausibly a capability that only appears above some scale. The
   monotonic sweep is consistent with "4B cannot use this channel at all."
2. **Tied embeddings at 4B.** Per finding 6, `--latent_space_realign` is inert at
   4B, so there is no way to map output-space vectors into the input slot. At 14B the
   flag is a real transform. 4B may be structurally unable to run the method as intended.
3. **Code revision drift.** Both released logs print `method: muscle` and have no
   `latent_steps` field, so they predate the released `run.py`. Matching flags does
   not guarantee matching behavior.
4. **The padding bug** (~18 points) — real, but present in the authors' runs too, so
   it explains our absolute numbers, not the gap to theirs.
5. **Sampling noise** — temperature 0.6, seed 42; ±1–2 points run to run. Far too
   small to matter here.

---

## Environment notes

Built and run on UCI HPC3. Two constraints shaped every script:

- `~/.local/lib/python3.10/site-packages` leaks onto `sys.path` of every python3.10
  env on this account, shadowing torch's CUDA libs — and a bare `pip install` into a
  fresh env *uninstalls out of it*. Every script exports `PYTHONNOUSERSITE=1` and
  `PIP_USER=0`; `00_setup_env.sh` asserts no `.local` path survives.
- The billed `gpu`/`gpu32` partitions reject this account at the job_submit plugin,
  so every run is preemptible. Hence checkpoint/resume and the self-chaining sbatch.

`models.py` hardcodes `torch.bfloat16` — never schedule on V100 (sm_70) or
RTX6000 (sm_75).

Detailed findings and full order of operations: **[`repro/README.md`](repro/README.md)**.

## License

Upstream code is under the original project's license; see [`LICENSE`](LICENSE).
