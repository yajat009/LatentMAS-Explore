# LatentMAS-Explore

A reproduction study of **LatentMAS** — *Latent Collaboration in Multi-Agent Systems*
([arXiv:2511.20639](https://arxiv.org/abs/2511.20639),
[Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS)).

LatentMAS claims that multi-agent collaboration can move out of token space and into
the model's KV cache, buying large token savings and a **×3.7 wall-clock speedup at
no accuracy cost**. This fork runs that claim on real hardware and reports what
survives.

**Scope — read this first.** The paper's speedup table covers **AIME24, AIME25 and
GPQA-Diamond at Qwen3-8B and 14B**. Most of the work here is MBPP+ at **Qwen3-4B**,
which is *outside* that range on both axes. So these results do not refute the paper's
headline; they map its edges. Two findings follow:

1. **The floor is between 4B and 14B.** At 14B the latent channel is free (0.911 with
   it on vs 0.933 off — one problem in 45). At 4B, identical code/task/prompt/GPU, it
   costs **27 accuracy points** and the model's output collapses into degenerate
   repetition in 25.8% of problems (0.0% at 14B). The paper never claims 4B; this
   locates where the method stops working, which was not previously documented.
2. **The ×3.7 cannot be reproduced on MBPP+, for a structural reason.** The saving
   comes from silencing three agents, so it scales with how verbose those agents would
   have been. TextMAS spends ~38,600 tokens on an AIME24 problem and ~2,200 on an
   MBPP+ problem — **17× less to save**. Measured on MBPP+/4B: ×2.32 wall clock, ×1.39
   tokens. That is the right answer for this task, not a failed reproduction.

A GPQA-Diamond / 14B / three-arm run — the configuration the paper actually claims —
is in flight. Until it lands, nothing here tests the headline claim directly.

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

Everything in `yajat/` is new. Upstream files were changed only where noted.

### 1. Fixes required to make the repo run at all

| file | change |
|---|---|
| `methods/latent_mas.py` | Made the vLLM import conditional. It was unconditional, but `requirements.txt` has no vllm — import-time crash. |
| `run.py` | `--model_name` choices listed `Qwen3-4B` twice and omitted `Qwen3-8B`. |
| `data.py` | Bare dataset id `gsm8k` → `openai/gsm8k`; modern `huggingface_hub` rejects the legacy form. |
| `methods/latent_mas.py` | Ported `_truncate_past` off the legacy `Cache` API, removed in transformers 5.x. |

Applied by `yajat/02_patch_blockers.sh`. Original `run.py` kept at `yajat/run.py.orig`.

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
| `yajat/analyze.py` | Re-tokenizes every agent's `[Output]` block from a stdout log |
| `yajat/compare_arms.py` | Reads the checkpoint JSONL instead, so it works **mid-run**; compares arms on the prefix they have all finished |
| `yajat/analyze_sweep.py` | Builds the `latent_steps` sweep table and classifies its shape |
| `yajat/check_realign.py` | Rebuilds the realign least-squares solve from safetensors — no GPU, never materializes a 14B model |
| `yajat/compare_padfix.py` | A/B for `--pad_fix` |
| `yajat/trace_pipeline.py` | Narrates the KV cache growing across agents — fastest way to see the mechanism |

### 4. Cluster harness

`yajat/run_latentmas.sbatch` (self-resubmitting on SIGTERM, guarded by "did the
checkpoint grow this attempt?" so a real crash fails once, not twenty times), plus
`00_setup_env.sh`, `01_fetch_assets.sh`, and the `submit_*.sh` launchers.

---

## Results

MBPP+ · sequential · `max_new_tokens=4096` · `think=1` · one L40S · seed 42.
Qwen3-4B unless stated; the scale table below adds Qwen3-14B.

### Main comparison — all 378 problems

All three arms complete (`yajat/results/arm_comparison.json`):

| arm | accuracy | sec/problem | gen tok/prob | prompt tok/prob |
|---|---|---|---|---|
| `baseline` (single agent, bs=15) | 0.691 | 29.0 | 1430 | 243 |
| `latent_mas` ls=10, bs=15 | **0.228** | 44.5 | 1578 | 1092 |
| `text_mas` bs=8 | **0.722** | 93.1 | 2154 | 5119 |

**TextMAS beats LatentMAS by ~49 accuracy points.** LatentMAS is cheaper — ×2.09 in
wall clock, ×1.37 in generated tokens — but neither figure is the claimed ×3.7.

**LatentMAS also loses to the plain single-agent baseline on every axis at once:**
lower accuracy, 1.5× slower, 4.5× the prefill. There is no dimension on which it
wins here. TextMAS beats the baseline by only 2.9 points for 3.3× the wall clock.

> **The wall-clock ratio was confounded — and correcting it makes the speedup
> larger, not smaller.** The table above compares LatentMAS at bs=15 against TextMAS
> at bs=8 (TextMAS OOMs at 15), and a bigger batch lowers time-per-problem on its own.
> Re-running LatentMAS at a matched bs=8 gives, on the 216 problems both have finished:
>
> | arm (both bs=8) | accuracy | sec/prob | gen tok |
> |---|---|---|---|
> | `latent_mas` ls=10 | 0.324 | **43.0** | 1575.5 |
> | `text_mas` | 0.713 | **99.7** | 2191.5 |
>
> **×2.32 wall clock, ×1.39 generated tokens** — against the paper's ×3.7. So the
> confound was real and worth removing, but it was hiding a *higher* speedup, and the
> claim still does not reproduce on either axis. The token ratio is batch-independent
> and barely moved (×1.37 → ×1.39), which is the consistency check.
>
> (LatentMAS is marginally *faster* at bs=8 than bs=15 — 43.0 vs 44.5 s/problem —
> backwards from normal batching behaviour, and most likely the padding: bs=15 packs
> more pad per sequence, so more compute is wasted on non-tokens. Accuracy rises too,
> 0.228 → 0.324, for the same reason. Figures are on 216/378 problems; the arm is
> still running.)

### Scale — the decisive experiment

Same code, same task, same prompt, same L40S, `bs=1` (no padding artifact), 45
problems. The only variable is model size:

| | **ls=0** (channel off) | **ls=10** (channel on) | cost of the channel |
|---|---|---|---|
| **Qwen3-4B** | 0.822 | **0.556** | **−26.6 pts** |
| **Qwen3-14B** | 0.933 | **0.911** | **−2.2 pts** (one problem) |

Data: `yajat/results/scale_2x2.json`.

**At 14B the latent channel is free; at 4B it is catastrophic.** This is the central
result of the reproduction. It also matches the authors' own released 14B log run at
ls=10 (84.8% on HumanEval+) — the method does work, at 14B.

The paper's tables start at 8B, so it never claims 4B works — this is not a
contradiction of the paper but a measurement of where the method's floor sits, which
the paper does not report.

**And at 14B the channel does pay — modestly.** Turning it on cuts generated tokens
7.2% (1299.4 → from 1400.3) and wall clock 3.8% (61.1 vs 63.5 s/problem) at
statistically equal accuracy. So the mechanism delivers, just an order of magnitude
below the headline. Note the comparison arm: this is ls=10 vs ls=0, **not** vs
TextMAS, which is what the paper's ×3.7 measures against and which was not run at
14B here.

Caveat: 45 problems carries roughly ±7 points of sampling error, so the 4B→14B gap
(27 points) is solid but the within-14B gap (2.2 points, one problem) is not
distinguishable from zero — which is the point: the channel costs nothing measurable
at 14B.

### `latent_steps` sweep — 45 problems, bs=1

| ls | 0 | 1 | 2 | 5 | 10 | 20 | 40 | 10 +realign |
|---|---|---|---|---|---|---|---|---|
| accuracy | 0.822 | 0.778 | 0.778 | 0.644 | 0.556 | 0.533 | 0.533 | 0.511 |

**Monotonic, no cliff.** The 0.289 decline is spread evenly across steps, which
points at the latent channel being genuinely harmful at this scale rather than
broken by a single bug. More latent thinking is strictly worse, every step of the way.

### How 4B fails: degenerate repetition

The 4B collapse is not an extraction or plumbing bug — it is the model's output
falling into a repetition loop:

```
...and_and and__and and and_and and__and andand andandandand and and_and and_ and_re
and and and and_and and__and and_ and_and and__and and__and and_ and__and...
```

Measured over every finished problem in each arm ("degenerate tail" = fewer than 15%
unique words in the last 400):

| arm | accuracy | degenerate tail | unclosed `<think>` | empty prediction |
|---|---|---|---|---|
| 4B `latent_mas` ls=10 | 0.282 | **25.8%** | 48.9% | 31.8% |
| 4B `text_mas` | 0.722 | 0.9% | 29.6% | 2.6% |
| 4B `baseline` | 0.690 | 0.6% | 11.9% | 13.0% |
| **14B `latent_mas` ls=10** | **0.911** | **0.0%** | 4.4% | 4.4% |

**This rules out a harness bug and evidences the mechanism.** Same code, same
extractor, same prompts: TextMAS degenerates 0.9% of the time and 14B LatentMAS 0.0%.
Only the 4B latent arm collapses. Degenerate repetition is the standard signature of a
residual stream drifting off-distribution — the model loses varied continuations and
falls into a fixed point — which is exactly what feeding `hᴸ` back with no projection
onto the embedding manifold predicts.

One genuine but minor defect surfaced in the same audit: 6 of 280 problems (2.1%) have
code after `</think>` that the extractor misses. Real, worth fixing, far too small to
explain anything.

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
explicitly.** The two logs differ here, which is easy to miss: neither arg namespace
has a `latent_steps` field, but each log's per-agent `[Latent Steps]` marker carries
the value — **0** in all 1134 MBPP+ entries, **10** in all 492 HumanEval+ entries. So
only the HumanEval+ cell exercised the channel. It also prints `method: muscle` where
the MBPP+ log prints `latent_mas`, so at least one predates the released `run.py`.

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

## Why these numbers differ from the paper

**First: we tested outside the paper's claimed range.** Its speedup table covers
AIME24/AIME25/GPQA-Diamond at 8B and 14B. Most work here is MBPP+ at 4B. Both axes
differ, so a gap was expected and most of it is explained below without any appeal to
error on either side.

**Task verbosity explains the speedup gap.** The saving comes from silencing three
agents, so its size is set by how much those agents would otherwise say:

| task / model | TextMAS tok | LatentMAS tok | reduction |
|---|---|---|---|
| AIME24 / 8B (paper) | 38,596 | 8,953 | 76.8% |
| AIME25 / 8B (paper) | 45,088 | 8,699 | 80.7% |
| GPQA-D / 14B (paper) | 20,931 | 3,606 | 82.8% |
| **MBPP+ / 4B (here)** | **2,191** | **1,575** | **28%** |

TextMAS is ~17× more verbose on AIME24 than on MBPP+. There is simply far less to save
on an easy code task, so ×3.7 is unreachable there by construction.

**Confirmed, not contradicted: LatentMAS is slower than a single agent.** Reading the
paper's Speed column as time (2808/688 = ×4.08 = its ×4.1), Single → LatentMAS goes
421 → 688, 450 → 820, 1018 → 1149, 1040 → 1473: **slower in 11 of 12 cells.** Our
MBPP+ measurement (44.5 vs 29.0 s/problem) agrees with the paper's own data.

**Measured: model scale.** The 2×2 above is a controlled comparison — 27 points at 4B,
nothing at 14B, every other variable fixed. Since the paper's tables start at 8B, this
maps the method's floor rather than disputing a claim.

**Why scale should matter (mechanism — inferred, not proven).** The method replaces
the sampled token with the raw last-layer hidden state. The token is not only a
bottleneck; it is a *projection back onto the training distribution* — sampling lands
you exactly on one of the ~150k rows of `W_in`, discarding accumulated drift at every
step. Feed `hᴸ` back directly and that snap never happens, so the loop has no error
correction anywhere in it and drift compounds. The sweep's shape is the fingerprint:
a smooth decline that **plateaus at 0.533** once the state has fully decohered, rather
than a cliff (which would indicate a bug) or unbounded decay. A larger model plausibly
has a wider basin — 14B carries 5120 dimensions against 4B's 2560 for a similarly sized
vocabulary, so features sit closer to orthogonal and a perturbed vector stays nearer
what it started as. **This last step is the least certain claim here**: the data show
*that* scale rescues the method, not *why*. See "open" below for the test.

**Not the explanation — `--latent_space_realign`.** Tempting, but wrong. The flag is
inert at 4B (tied embeddings make the least-squares solve return the identity exactly;
`M_vs_I_relative_fro = 0.0`) and a real near-orthogonal transform at 14B. But it
**defaults to off on both**, and `_build_latent_realign_matrix` then overwrites `M`
with `torch.eye(...)`. So both models fed back raw `hᴸ` with only a norm rescale.
Realignment does not distinguish our runs from theirs.

**Secondary, unresolved: task and prompt.** The authors' one released log with the
channel genuinely on is HumanEval+/*hierarchical*; we ran MBPP+/*sequential*. Our 4B↔14B
comparison is internally clean, so this does not threaten the scale result — but it is
not ruled out as an additional factor.

**Code revision drift.** The HumanEval+ log prints `method: muscle` and the MBPP+ log
prints `latent_mas`; neither arg namespace has a `latent_steps` field (the value comes
from each log's per-agent `[Latent Steps]` marker). So at least one predates the
released `run.py`, and matching flags does not guarantee matching behavior.

**Not a factor: the padding bug.** Real (~18 points) but present in the authors' runs
too — 94.4% pad-seeded in their released HumanEval+ log. It moves our absolute numbers,
not the gap to theirs.

**Not a factor: sampling noise.** Temperature 0.6, seed 42; ±1–2 points run to run.

### Open

The drift account above is measurable and currently untested. Instrument the latent
loop to record, at each step, the **maximum cosine similarity between the latent vector
and any row of `W_in`** — i.e. how close the state stays to any real token embedding.
If it decays fast at 4B and holds at 14B, drift is confirmed and the mechanism is
established rather than inferred. One forward pass per step, no generation, minutes per
model; it belongs in `yajat/trace_pipeline.py`, which already walks the loop.

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

Detailed findings and full order of operations: **[`yajat/README.md`](yajat/README.md)**.

## License

Upstream code is under the original project's license; see [`LICENSE`](LICENSE).
