# LatentMAS-Explore

A reproduction study of **LatentMAS** — *Latent Collaboration in Multi-Agent Systems*
([arXiv:2511.20639](https://arxiv.org/abs/2511.20639),
[Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS)).

This is a fork. The upstream method code is essentially unmodified; what is mine is
everything in `repro/`, the resume support in `run.py`, and the findings below.
The original project README is preserved verbatim at
[`README_upstream.md`](README_upstream.md).

## Purpose

LatentMAS makes a specific, checkable claim: multi-agent collaboration can move out
of token space and into the model's KV cache, buying large token savings and a
**×3.7 wall-clock speedup** at no accuracy cost.

I wanted to know what that claim actually rests on — so I ran it. Not to debunk it;
to find out which parts survive contact with a GPU, and to learn how a latent-space
agent handoff works by taking one apart. The headline result is that the mechanism
is real and elegant, and that several of the numbers around it mean something
narrower than they read.

## Repository layout

| directory | purpose |
|---|---|
| [`repro/`](repro/) | **Mine.** Reproduction harness, cluster scripts, and the full findings log |
| [`repro/results/`](repro/results/) | Run outputs — distilled summaries tracked, raw logs gitignored |
| [`repro/logs/`](repro/logs/) | Slurm and tooling logs; three tracked as evidence |
| [`methods/`](methods/) | Upstream. The three arms: `baseline`, `text_mas`, `latent_mas` |
| [`example_logs/`](example_logs/) | Upstream. The authors' two released logs — each is exactly one paper cell |
| [`data/`](data/) | `medqa.json`; the other nine tasks stream from HF at runtime |
| [`assets/`](assets/) | Upstream paper figures and slides |

Root-level `run.py`, `models.py`, `methods/`, `prompts.py`, `data.py`, `utils.py`
are upstream, with the exceptions noted under Contributions.

## How the method works

Four agents — Planner → Critic → Refiner → Judger. In `latent_mas`, the first three
**never emit a token**. Each one's prompt is forward-passed *on top of the running
`past_key_values`*, then `latent_steps` autoregressive steps run in embedding space:
the last layer's final hidden state is fed straight back in as `inputs_embeds`, with
no sampling and no detokenization. The KV cache *is* the entire inter-agent channel.
Only the judger calls `generate()`, inheriting all of it.

The token savings fall directly out of this: three of four agents produce zero
output tokens. `models.py:generate_latent_batch` is about forty lines and is the
whole paper.

## Contributions

**Made the repo run.** Two blockers stop it at import or argparse: `methods/latent_mas.py`
imports vLLM unconditionally though `requirements.txt` has no vllm, and `run.py`'s
`--model_name` choices list `Qwen3-4B` twice while omitting `Qwen3-8B`. Also fixed
dependency drift — `data.py` used the legacy bare dataset id `gsm8k`, which modern
`huggingface_hub` rejects, and transformers 5.x has removed the legacy `Cache` API
that `_truncate_past` calls. See `repro/02_patch_blockers.sh`, `repro/00_setup_env.sh`.

**Made long runs survivable.** Every GPU partition available to me is preemptible,
and preemption here is *cancel*, not requeue — four jobs died mid-run, two of them
past the halfway mark. `run.py` now takes `--checkpoint PATH`: a JSONL appended after
every batch, replayed on startup as a prefix skip, carrying elapsed time forward.
`repro/run_latentmas.sbatch` resubmits itself on SIGTERM, guarded by "did the
checkpoint grow this attempt?" so a real crash fails once rather than twenty times.

**Built the measurement the repo lacks.** `run.py` reports accuracy and wall-clock
and *no token counts* — but the paper's central claim is a token claim.
- `repro/analyze.py` re-tokenizes every agent's `[Output]` block from a stdout log
- `repro/compare_arms.py` reads the checkpoint JSONL instead, so it works **mid-run**
  and compares every arm on the common prefix they have all finished
- `repro/trace_pipeline.py` narrates the KV cache growing across agents — the
  fastest way to see the mechanism

### What I found

**1. The two released logs are two paper cells, confirmed to the token.**
Re-tokenizing them reproduces the paper's Token column *exactly*: 1621.2 vs 1621
(Table 1, 14B/MBPP+), 1512.3 vs 1512 (Table 2, 14B/HumanEval+). That is not
coincidence — these are the runs behind those cells, which pins down two things the
paper does not state.

**2. The Token metric counts generated text only.** Prefill is omitted: 785
tokens/problem for the MBPP+ run, 1183 for HumanEval+. LatentMAS forward-passes
*four* agent prompts to build its cache where a single agent passes one, and none of
that ~4× prefill appears in the Token column. This should be stated carefully rather
than as a refutation: prefill is parallel and far cheaper per token than sequential
decode, so `generated + prompt` is not an apples-to-apples cost either. The paper
measures decode tokens, which is the dominant cost and a legitimate metric. But the
omitted prefill is exactly what resurfaces in the two places the savings don't
materialize — wall-clock and peak memory, below.

**3. The published MBPP+/14B cell was produced with the latent channel switched off.**
That log has `latent_steps: 0` in all 1134 trace entries, and in today's code
`past_for_decoding = past_kv if self.latent_steps > 0 else None` — so the judger
receives `None` and the accumulated cache is discarded. Verified live with
`trace_pipeline.py`:

```
latent_steps=10                        latent_steps=0
  Planner  KV   0 ->  166                Planner  KV   0 ->  156
  Critic   KV 166 ->  373                Critic   KV 156 ->  353
  Refiner  KV 373 ->  566                Refiner  KV 353 ->  536
  Judger   inherits 566  <- channel      Judger   inherits   0  <- DISCARDED
```

`--latent_steps` **defaults to 0**. Always pass it explicitly. (Caveat: both released
logs print `method: muscle` and carry no `latent_steps` field at all, so they came
from an earlier code revision; what that revision did at ls=0 is unknown.)

**4. LatentMAS costs peak memory, and the efficiency framing omits it.** Same GPU,
same batch size, same data, Qwen3-4B on one 24 GB A30 at bs=15: `latent_mas` OOMs
where `baseline` and `text_mas` both run. The judger decodes against the
concatenated KV of all three prior agents, for every one of `max_new_tokens` steps.
This is intrinsic, not an implementation wart — it is the same property the token
savings come from.

**5. Wall-clock: slower than the single-agent baseline here.** The paper's own
Table 1 agrees in direction (4B/MBPP+: 577 vs 523). **Its ×3.7 speedup is against
TextMAS, not against a single agent** — any summary saying "LatentMAS is faster"
without naming the comparison arm is wrong.

**6. The latent thought is seeded from a PAD token at any `--generate_bs > 1`.**
`generate_latent_batch` reads `hidden_states[-1][:, -1, :]` — position −1 of a
*right-padded* batch — then builds `latent_mask = torch.ones(...)`, un-masking every
pad already in the cache. Measured: 93.3% of sequences seed from a pad and 35.8% of
KV positions are pad. Re-tokenizing the *released* HumanEval+ log gives 94.4% and
32.7%, so the published numbers were produced in this state too. It is invisible at
bs=1, which is why smoke tests miss it.

### Measured results

Qwen3-4B · MBPP+ · sequential · `max_new_tokens=4096` · `think=1` · one L40S · seed 42.
All arms on the common prefix of 45 problems they have all finished
(`repro/results/arm_comparison.json`):

| arm | accuracy | sec/problem | gen tok/prob | prompt tok/prob |
|---|---|---|---|---|
| `baseline` (single agent) | 0.778 | 25.7 | 1237 | 227 |
| `text_mas` (bs=8) | **0.800** | 75.8 | 2203 | 4757 |
| `latent_mas` ls=0 | 0.756 | 30.5 | 1335 | 1029 |
| `latent_mas` ls=10, bs=15 | 0.378 | 41.6 | 926 | 1029 |
| `latent_mas` ls=10, bs=1 | 0.556 | 30.3 | 1527 | 1029 |

The token claim reproduces in direction: LatentMAS decodes 42% of TextMAS's
generated tokens at ls=10, 61% at ls=0. The speed claim reproduces in direction but
not magnitude — 75.8 → 41.6 s/problem is ×1.8 against TextMAS, not ×3.7 — and
inverts against the single-agent baseline. Read the ls=10 token and speed figures
with care: that arm is also the one whose accuracy has collapsed, and part of what
makes it cheap is that it stops generating sooner. The accuracy does not reproduce
at 4B — and the last two rows isolate why. Right-padding costs ~18 points (0.556 → 0.378), but bs=1
is still 22 points under baseline, so **the latent channel itself costs more than
the padding bug does.** Over the full 378 problems the gap widens further:
`baseline` 0.690, `latent_mas` ls=10 0.228.

Caveats worth stating: this is 4B, not the paper's 14B; `text_mas` runs at bs=8
because it OOMs at 15; and generation is sampled at temperature 0.6, so expect
±1–2 points run to run.

## Learning steps

The order I would take, cheapest insight first. Each assumes `repro/00_setup_env.sh`,
`01_fetch_assets.sh`, `02_patch_blockers.sh` have been run.

**1. Read the mechanism before spending a GPU-hour.** `python repro/trace_pipeline.py`
narrates the cache growing across agents. Twenty minutes with its output next to
`models.py:generate_latent_batch` is worth more than any run.

**2. Sweep `latent_steps` — the biggest open gap.** Only ls=0 and ls=10 have been
run, and accuracy collapses between them. The sweep says whether that is monotonic
(the channel is actively harmful at this scale) or a cliff (a bug at some length).
`ls ∈ {1,2,5,10,20,40}`, `--max_samples 45 --generate_bs 1`, ~25 min each.
The single most informative experiment available.

**3. Test `--latent_space_realign`.** Completely unrun. It least-squares-solves
`W_out @ M ≈ W_in` to map a hidden state from output space back into input-embedding
space; without it `M` is identity, so output-space vectors are fed back as input
embeddings. A plausible cause of the ls>0 collapse. `REALIGN=1`, paired with step 2.

**4. Then decide whether to fix the padding bug.** Its size is known now (~18 pts).
A correct fix needs all three of: seed at `attention_mask.sum(-1)-1`; propagate the
real mask into `latent_mask`; and if switching to left padding, also fix the
`sequences[idx, prompt_len:]` slice in `generate_text_batch`, which is only correct
under right padding. Do it on a branch — right padding is what the authors ran, so
the unfixed path is the reproduction path.

**5. Finish `text_mas` on the full 378.** It is the arm the ×3.7 speedup is measured
against, so the headline claim stays partly untested until it does. It must run at
bs=8, so re-run one comparison arm at bs=8 for a fair wall-clock pairing.

**6. Wire up the two dead ablation flags.** `latent_only` and `sequential_info_only`
are read by `methods/latent_mas.py` but never defined in `run.py`, so they are
permanently False. Two `add_argument` lines make the paper's own ablations reachable.

Detailed findings, cluster constraints, and the full order of operations:
**[`repro/README.md`](repro/README.md)**.

## Environment notes

Built and run on UCI HPC3. Two constraints shaped every script here:

- `~/.local/lib/python3.10/site-packages` leaks onto `sys.path` of every python3.10
  env on this account, shadowing torch's CUDA libs — and a bare `pip install` into a
  fresh env *uninstalls out of it*. Every script exports `PYTHONNOUSERSITE=1` and
  `PIP_USER=0`; `00_setup_env.sh` asserts no `.local` path survives.
- The billed `gpu`/`gpu32` partitions reject this account at the job_submit plugin,
  so every run is preemptible. Hence the checkpoint/resume and self-chaining sbatch.

`models.py` hardcodes `torch.bfloat16` — never schedule on V100 (sm_70) or
RTX6000 (sm_75).

## License

Upstream code is under the original project's license; see [`LICENSE`](LICENSE).
