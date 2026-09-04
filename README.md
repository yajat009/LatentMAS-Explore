# LatentMAS-Explore

This is my attempt to reproduce the results of the paper **LatentMAS — Latent
Collaboration in Multi-Agent Systems** ([arXiv:2511.20639](https://arxiv.org/abs/2511.20639),
code at [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS)).

I ran the authors' released code on a university cluster, compared what I got to
what the paper reports, and then dug into *why* the numbers differ. This README is
written as a lab notebook: what the paper claims, what I did, what I saw, and what I
think it means. Every technical term is explained the first time it shows up, in a
box like this:

> **Term.** Plain-language explanation.

The short version is at the top. The full story follows.

---

## The short version

The paper says that if you let several copies of a language model "talk" to each
other using their internal numbers instead of words, you get answers that are just
as good or better, while using far fewer words and running about **3.7× faster**.

I tested this on the exact setup the paper reports in its Table 1: the Qwen3-4B
model, on a coding benchmark called MBPP+. The paper reports **73.5% accuracy**
there. I measured **22.8%**.

After a lot of digging, here is what I believe:

1. **The released code is faithful to the paper.** I checked every stage against the
   paper's description and against the library it runs on. It does what the paper
   says.
2. **When the "talking with internal numbers" feature is actually switched on, the
   4B model falls apart.** Its answers dissolve into repeated nonsense words about a
   quarter of the time. A bigger model (14B) does not have this problem at all.
3. **The paper's own MBPP+ results were very likely produced with that feature
   switched off.** Three separate measurements line up with this. If true, the
   paper's 73.5% is really a single model answering alone, and the "improvement"
   is not what it appears to be.
4. **Even where the method works (14B), the speedup is much smaller than 3.7×**
   on this task — about 2.3× against the comparison the paper uses, and it is
   actually *slower* than a single model working alone.

Everything below is the evidence for those four sentences.

---

## Part 1 — What the paper is claiming

### The idea

Normally, when you build a system with several language-model "agents" that help
each other (a planner, a critic, and so on), each agent writes out its thoughts in
words, and the next agent reads those words. That is a lot of words, and every
word costs time to generate.

LatentMAS proposes skipping the words. Instead of an agent writing text, it hands
the next agent its **internal state** directly.

> **Hidden state.** Inside a language model, every word position is represented by
> one long list of numbers (2,560 numbers for Qwen3-4B; 5,120 for 14B). The model
> updates this list layer by layer as it processes text. The version at the very
> last layer is what the model uses to decide the next word. This list of numbers
> is called the hidden state. It is not words — it is the model's raw internal
> representation.

> **KV cache.** As the model reads text, it stores a compressed memory of every
> position it has seen so far, so it doesn't have to re-read from scratch for each
> new word. This memory is called the KV cache ("K" for key, "V" for value — names
> from the attention mechanism). It is the model's working memory for the current
> conversation.

The method works like this, for four agents in a row (Planner → Critic → Refiner →
Judger):

1. The Planner reads its prompt. Instead of writing a plan in words, it takes its
   last-layer hidden state and feeds that number-list straight back into itself as
   if it were the next word. It repeats this a chosen number of times (the paper
   calls these **latent steps**). Each step adds to the KV cache. No words are
   produced.
2. The Critic starts with the Planner's KV cache already loaded, so it "remembers"
   what the Planner was thinking. It does the same silent thinking.
3. The Refiner does the same.
4. Only the Judger writes actual words — the final answer — with the whole
   accumulated memory of the previous three agents behind it.

Because three of the four agents never write a word, the system produces far fewer
words in total. That is where the claimed savings come from.

### The numbers the paper reports (Table 1, the cell I tested)

Qwen3-4B, MBPP+ benchmark, "sequential" arrangement of agents:

| | Single model alone | TextMAS (agents talk in words) | **LatentMAS** (this method) |
|---|---|---|---|
| Accuracy | 63.5% | 69.8% | **73.5%** |
| Words generated per problem | 1,634 | 4,420 | **1,339** |
| Time per run (seconds) | 523 | 2,148 | **577** |

The speedup the paper highlights is 2,148 ÷ 577 ≈ **3.7×**, comparing LatentMAS
against TextMAS.

> **MBPP+.** A benchmark of 378 small Python programming tasks ("write a function
> that…"). An answer counts as correct only if the generated code passes all the
> hidden test cases. It is a pass/fail check, not a judgment call.

> **TextMAS.** The same four agents, but each one writes its thoughts in words and
> the next one reads them. This is the "normal" way to do multi-agent systems, and
> it is what the paper's speedup is measured against.

> **Single / baseline.** One model, one prompt, one answer. No agents at all.

---

## Part 2 — Getting the code to run

The released code did not run out of the box. Four things had to be fixed before I
could test anything. These are in `architecture/` (where I moved the authors' code —
see the layout note at the end).

| file | problem | fix |
|---|---|---|
| `methods/latent_mas.py` | Tried to import a library called vLLM that isn't in the requirements, so it crashed on startup. | Made the import optional. |
| `run.py` | The list of allowed model names had `Qwen3-4B` twice and was missing `Qwen3-8B`. | Fixed the list. |
| `data.py` | Used an old-style dataset name (`gsm8k`) that the current download library rejects. | Updated to `openai/gsm8k`. |
| `methods/latent_mas.py` | A function for trimming the KV cache used an interface the current library removed. | Rewrote it for the current interface. |

I also added three things the code didn't have:

- **Checkpointing** (`--checkpoint`). The cluster I use can kill a job at any time
  to make room for paying users, and it does — four of my runs were killed partway
  through, two of them past the halfway point. Now every batch of results is
  saved to a file as it finishes, and a restarted job picks up where it left off.
- **Two ablation switches** (`--latent_only`, `--sequential_info_only`) that the
  method code reads but that were never actually wired into the command line, so
  they could never be turned on.
- **A padding fix** (`--pad_fix`), off by default. Explained in Finding 5.

And I built the measurement tools the repo lacks. The paper's central claim is
about how many words are generated, but `run.py` only prints accuracy and time. So:

| script | what it does |
|---|---|
| `yajat/analyze.py` | Counts the words each agent generated, from a finished log. |
| `yajat/compare_arms.py` | Does the same from the checkpoint files, so it works on runs that are still going. Compares different setups only on problems they've all finished. |
| `yajat/analyze_sweep.py` | Builds the latent-steps sweep table. |
| `yajat/check_realign.py` | Computes the paper's alignment matrix from the model weights, without needing a GPU (Finding 6). |
| `yajat/trace_pipeline.py` | Prints the KV cache growing step by step across the four agents — the quickest way to *see* the mechanism. |

---

## Part 3 — What I measured

All runs: MBPP+, sequential agents, up to 4,096 words of output per answer,
sampling temperature 0.6, top-p 0.95, random seed 42, one NVIDIA L40S GPU. These
match the settings stated in the paper. Qwen3-4B unless I say otherwise.

> **Temperature / top-p.** Knobs that control how random the model's word choices
> are. 0.6 and 0.95 are the paper's values. Because there is randomness, expect
> results to wobble by a point or two between runs.

> **bs (batch size).** How many problems the GPU works on at the same time. Bigger
> batches are faster per problem. This turns out to matter a lot — see Finding 5.

> **ls (latent steps).** How many silent thinking steps each of the three silent
> agents takes. `ls=0` means zero steps, which — as I discovered — means the whole
> mechanism is off (Finding 3).

### 3.1 — The main comparison, all 378 problems

| setup | accuracy | seconds per problem | words generated | words in the prompt |
|---|---|---|---|---|
| Single model (bs=15) | 69.1% | 29.0 | 1,430 | 243 |
| **LatentMAS**, ls=10 (bs=15) | **22.8%** | 44.5 | 1,578 | 1,092 |
| TextMAS (bs=8) | **72.2%** | 93.1 | 2,154 | 5,119 |

Reading this row by row:

- **TextMAS beats LatentMAS by about 49 accuracy points.** LatentMAS is cheaper —
  2.1× faster, 1.4× fewer words — but not the 3.7× the paper claims, and at 22.8%
  accuracy the savings don't mean much.
- **LatentMAS loses to the plain single model on every single column.** Lower
  accuracy, slower, and 4.5× more prompt words (because it has to feed four
  agents' prompts through the model instead of one). There is no measurement here
  on which it wins.
- TextMAS beats the single model by only about 3 points, for 3.2× the time.

TextMAS ran with bs=8 instead of 15 because it runs out of GPU memory at 15. That
makes the time comparison slightly unfair, which I fixed next.

### 3.2 — The same comparison with matched batch sizes

I re-ran LatentMAS at bs=8 so both are on equal footing. On the 216 problems both
finished:

| setup (both bs=8) | accuracy | seconds per problem | words generated |
|---|---|---|---|
| LatentMAS, ls=10 | 32.4% | **43.0** | 1,576 |
| TextMAS | 71.3% | **99.7** | 2,192 |

Speedup: **2.32×** in time, **1.39×** in words. Fixing the batch mismatch made
the speedup a bit *bigger*, not smaller — but it's still far from 3.7×.

(Interesting side note: LatentMAS was slightly *faster* at the smaller batch —
43.0 vs 44.5 seconds — which is backwards from how batching usually works. I think
this is the padding problem in Finding 5: at bs=15 the model wastes effort on
filler.)

### 3.3 — Does more silent thinking help? (45 problems, bs=1)

The paper says accuracy goes *up* with more latent steps, peaking at 40–80. I
swept the number of steps:

| latent steps | 0 | 1 | 2 | 5 | 10 | 20 | 40 | 10 (with alignment on) |
|---|---|---|---|---|---|---|---|---|
| accuracy | **82.2%** | 77.8% | 77.8% | 64.4% | 55.6% | 53.3% | 53.3% | 51.1% |

**Every step of silent thinking makes the 4B model worse.** The decline is smooth —
no single point where it suddenly breaks. Then it flattens around 53%. I'll explain
in Finding 7 why the shape of this curve matters.

I ran this at bs=1 to remove the padding problem (Finding 5), so this is the
cleanest view of the method on its own.

### 3.4 — Is it the model size? (the decisive test)

Same code, same task, same prompts, same GPU, same 45 problems, bs=1. The only
thing I changed was the model:

| | ls=0 (mechanism off) | ls=10 (mechanism on) | what turning it on costs |
|---|---|---|---|
| **Qwen3-4B** | 82.2% | **55.6%** | **−26.6 points** |
| **Qwen3-14B** | 93.3% | **91.1%** | −2.2 points (one problem out of 45) |

**At 14B the mechanism costs essentially nothing. At 4B it is catastrophic.**

And at 14B it does deliver a small saving: with the mechanism on, the model
generated 7% fewer words (1,299 vs 1,400) and ran 4% faster (61.1 vs 63.5 seconds)
at the same accuracy. So the idea *works* — just at a much smaller scale than
advertised, and only for the bigger model.

This agrees with the one released log from the authors where the mechanism was
genuinely on (14B, ls=10, a different benchmark): 84.8%.

### 3.5 — What does a failure actually look like?

I pulled apart the 4B failures. Here is how one starts (the model is supposed to
be writing a Python function):

> *"The task is to write a Python function calledsimilar_elements that takes two
> tuples as input... The initial plan is to use set operations... The initial plan
> may not be correct because the testcases use tuples as input... The initial plan
> may not be correct because."*

Notice two things. First, it is already glitching in the first sentence —
"called**similar_elements**" with the space dropped, "testcases", a sentence that
just stops. Second, it says "the initial plan" — **it is reading the Planner's
silent thoughts out of the cache.** It never saw a plan in words. So the mechanism
is genuinely transferring information; the information is just damaged.

By 3,000 characters in:

> *"thefunction and thefunction and thefunction and thefunction and thefunction..."*

It has collapsed completely. I measured how often this happens:

| setup | accuracy | collapsed into repetition | never finished its reasoning | produced no code at all |
|---|---|---|---|---|
| 4B LatentMAS, ls=10 | 28.2% | **25.8%** | 48.9% | 31.8% |
| 4B TextMAS | 72.2% | 0.9% | 29.6% | 2.6% |
| 4B Single | 69.0% | 0.6% | 11.9% | 13.0% |
| **14B LatentMAS, ls=10** | 91.1% | **0.0%** | 4.4% | 4.4% |

> **"Collapsed into repetition."** I counted an answer as collapsed if fewer than
> 15% of its last 400 words were different from each other — i.e., it's stuck in a
> loop.

> **"Never finished its reasoning."** Qwen3 models "think out loud" inside
> `<think>...</think>` tags before answering. If the closing tag never appears, the
> model spent its entire word budget thinking and never got to the answer.

Same code, same answer-extraction, same prompts across all four rows. Only the 4B
LatentMAS row collapses. **This rules out a bug in my harness** — if the
extraction or plumbing were broken, TextMAS and Single would break too.

---

## Part 4 — The findings, one at a time

### Finding 1 — The authors' two released logs are two exact cells of the paper

The repo ships two example output logs from Qwen3-14B. I re-counted the words in
them and got the paper's Table numbers **to the digit**: 1,621.2 vs 1,621 (MBPP+),
1,512.3 vs 1,512 (HumanEval+). These aren't similar runs; these are the runs.

That lets me learn two things the paper doesn't say (Findings 2 and 3).

### Finding 2 — The paper counts only generated words, not prompt words

The "Token" column counts words the model *wrote*. It does not count words the
model *read* (the prompts): 785 per problem for the MBPP+ log, 1,183 for
HumanEval+.

That matters because LatentMAS has to push **four** agents' prompts through the
model to build its memory, where a single model pushes one. None of that appears
in the paper's count.

To be fair to the paper: reading a prompt is much cheaper per word than writing
an answer (reading happens all at once; writing is one word at a time). So
"generated words" is a reasonable thing to measure. But the hidden reading cost is
exactly what shows up in the two places the savings vanish — time and memory.

### Finding 3 — The paper's MBPP+ 14B result was run with the mechanism OFF

In the released MBPP+ log, every one of the 1,134 agent entries records
`[Latent Steps] 0`. And in the code:

```python
past_for_decoding = past_kv if self.latent_steps > 0 else None
```

At zero latent steps, the shared memory is **thrown away** before the Judger
answers. The three silent agents run, build up a memory, and then it's discarded.
The Judger answers alone. It's a single model with extra wasted work.

The command-line default for `--latent_steps` is **0**. If you run the code the
obvious way, the mechanism is off.

The other released log (HumanEval+) has latent steps = 10 in all 492 entries, so
that one genuinely used the mechanism.

### Finding 4 — This is very likely true of the paper's *4B* MBPP+ result too

This is the finding I consider most important, so here is all the evidence.

**(a)** On the same first 60 problems, my single model scores **75.0%** and my
LatentMAS with ls=0 scores **75.0%** — identical, as they should be, because ls=0
*is* a single model. With ls=10 it scores 31.7%.

**(b)** My ls=0 run generates **1,335** words per problem. The paper's 4B LatentMAS
cell reports **1,339**. (My ls=10 runs generate 926–1,578 depending on batch size.)

**(c)** The paper says LatentMAS is only 1.10× slower than a single model (577 vs
523 seconds). My ls=0 is 1.19× slower than my single model. My ls=10 is **1.62×**
slower. The ls=0 number fits; the ls=10 number doesn't.

**(d)** The 14B MBPP+ log (Finding 3) is proven ls=0 and matches the paper exactly.

So three independent measurements — accuracy, word count, and timing — all fit
"the paper's MBPP+ LatentMAS numbers are single-model runs." If that's right, the
paper's 73.5% is a single model with a slightly different prompt, and the "↑" over
Single is prompt variation plus averaging over three runs.

I want to be honest about the limit of this: I can't prove what the authors ran at
4B, because they didn't release that log. And the HumanEval+ log shows they *did*
use the mechanism for some cells. This is a strong inference, not a proof.

### Finding 5 — A bug in batched runs: the silent thinking starts from a filler token

When the GPU processes several problems at once, shorter prompts are padded with
filler tokens at the *end* so they're all the same length.

> **Right padding.** Adding filler at the end of shorter sequences so a batch is
> rectangular. The filler is supposed to be ignored.

The code grabs the hidden state at the *last position* of each row to start the
silent thinking:

```python
last_hidden = outputs.hidden_states[-1][:, -1, :]
```

With right padding, the last position is **filler** for every prompt shorter than
the longest one in the batch — which I measured at 93.3% of prompts. So the silent
thinking starts from the model's reaction to a filler token, not to the actual
prompt. And the code then tells the model to pay attention to *all* positions
including the filler (35.8% of the cache).

This is invisible at bs=1, which is why quick tests miss it. It also affects the
authors' own runs: the released HumanEval+ log is 94.4% filler-seeded.

I measured its cost directly: same 45 problems, ls=10, **bs=1 gives 55.6% and
bs=15 gives 37.8%** — about 18 points. Over the full 378 problems the bs=15 run
lands at 22.8%.

The fix needs four coordinated changes (`--pad_fix`, off by default because the
padded version is what the authors ran). But note: even fixed, 4B at ls=10 is
still 22 points below the single model. **Fixing padding doesn't rescue the
method at 4B.** The mechanism itself costs more than the bug does.

### Finding 6 — The paper's "alignment" step is off by default in the code

The paper anticipates a problem with feeding hidden states back in as input:

> **Out-of-distribution input.** The model was trained to receive word embeddings
> at its input — a fixed set of ~150,000 specific number-lists, one per word. A
> hidden state is a *different kind* of number-list from the model's output side.
> Feeding one in as input is something the model never saw during training.

Their solution is a matrix `Wa` that maps output-side vectors into input-side
space, computed once from the model's weights. The paper says it's used at every
latent step and is worth +2.3% to +5.3% accuracy (their Figure 7).

In the code:

```python
parser.add_argument("--latent_space_realign", action="store_true")   # default: off
...
if self.args.latent_space_realign:  pass
else:                               realign_matrix = torch.eye(...)   # replace with "do nothing"
```

**Off by default, and the computed matrix is thrown away for an identity matrix.**
Running the code as released does not run the paper's method.

For Qwen3-4B this turns out not to matter, for a mathematical reason:

> **Tied embeddings.** Some models use the *same* matrix to turn words into
> vectors (input) and vectors into word scores (output). Qwen3-4B does. When the
> two are the same, the alignment matrix `Wa` works out to be exactly "do nothing"
> anyway (my `check_realign.py` confirms: distance from identity = 0.000).

So at 4B, the alignment can't help even when turned on — which is what I measured
(51.1% with it vs 55.6% without; within noise). At 14B the embeddings are *not*
tied, `Wa` is a real transformation (it rotates vectors almost 90°), and the code
skips it unless you pass the flag.

There's a second, smaller difference: the paper scales the fed-back vector by one
global constant; the code rescales *every* vector to exactly the average embedding
length, erasing any information in their relative sizes. I haven't tested whether
this matters.

### Finding 7 — Why the 4B model collapses (my explanation)

This is my interpretation. The measurements above support it, but I haven't
proven the mechanism directly.

Normally, a language model's loop goes: hidden state → scores for every word →
pick one word → look up that word's embedding → feed it in as the next input.

LatentMAS deletes the middle: hidden state → feed it straight in as the next input.

The paper's argument is that picking a word throws away information (a hidden
state carries thousands of numbers; a word carries about 17 bits' worth). That's
true. But **picking a word also does something else: it snaps the state back onto
the set of inputs the model was trained on.** Whatever noise or drift built up in
the hidden state gets discarded, because the next input is a clean, real word
embedding. Every single step.

Remove that, and you have a loop with no correction in it. Step 1's output is a
little off. It becomes step 2's input, so step 2's output is more off. Nothing pulls
it back. Errors compound.

The sweep in 3.3 is what that looks like: one or two steps barely hurt (the drift
is small), then it falls, then it *flattens* at 53% — because once the state has
fully drifted into noise, more steps can't make noise noisier. A bug would look
like a sudden cliff. Drift looks like this.

Why would 14B survive it? My best guess: a 14B model represents things with 5,120
numbers instead of 2,560. With more room, a slightly-off vector is more likely to
still be "near" what it was, rather than landing on top of something else. That
last step is the least certain part of this whole document.

The direct test would be: at each latent step, measure how close the fed-back
vector is to *any* real word embedding, for 4B vs 14B. If 4B's drifts away fast
and 14B's holds, that's the mechanism. It's a cheap experiment and it's next on
my list.

### Finding 8 — LatentMAS uses more GPU memory, and the paper doesn't mention it

Same GPU (a 24 GB A30), same batch size (15), same data: **LatentMAS runs out of
memory** where Single and TextMAS both fit. The Judger has to attend over the
combined memory of all three previous agents for every one of its up-to-4,096
output words. This is built into the method — it's the same property the word
savings come from — but "uses fewer words" and "uses more memory" are both true,
and the paper only reports one.

### Finding 9 — By the paper's own numbers, LatentMAS is slower than a single model

The paper's Table 2 has a "Speed" column (seconds per run; lower is better).
Comparing the Single column to the LatentMAS column: 421 → 688, 450 → 820,
1018 → 1149, 1040 → 1473... **LatentMAS is slower than a single model in 11 of the
12 cells.** The 3.7× speedup is only against TextMAS, which is the slow option.

Any summary that says "LatentMAS is faster" without saying "than TextMAS" is
wrong, by the paper's own data. My measurement (44.5 vs 29.0 seconds per problem)
agrees.

Also from that table: the accuracy gains over TextMAS are ↑3.4, ↑0.0, ↑2.1, ↑3.3,
↑0.0, ↑3.9, ↑0.5, ↑1.0 — four of twelve are exactly zero.

### Finding 10 — Things I checked and cleared (not bugs)

I want to be clear about what I *didn't* find wrong, because "the code is broken"
was my first suspicion too.

- **The hidden state being fed back is the right one.** It's the version after the
  model's final normalization — the exact vector the model uses to score words.
  I checked this specifically for the library version installed (transformers
  4.57.1), because that version changed how hidden states are collected and could
  have silently broken this. It didn't.
- **The memory handoff to the Judger is correct.** I traced the library's code
  path for "start generating with a pre-filled memory" and it does the right
  thing. And it's confirmed by behavior: the failing 4B Judger talks about "the
  initial plan," which it could only know from the cache.
- **The prompts match the paper's Appendix K** (a couple of sentences are in a
  different order).
- **Word budget (4,096), temperature (0.6), top-p (0.95)** all match Table 1's
  stated settings.
- **Answer checking matches the paper's protocol.** Take the last Python code
  block, attach the hidden tests, run with a 10-second timeout.
- **The answer extractor has one small gap:** 6 of 280 answers (2.1%) had code the
  extractor missed. Real, but far too small to matter.

### Finding 11 — A contradiction I can't explain yet

The paper's Figure 8 shows accuracy *rising* with more latent steps, peaking at
40–80. My sweep (3.3) shows it *falling* the whole way. Figure 8 is measured on
14B and mine is 4B, so they're not directly comparable — but the directions are
opposite, and the paper doesn't say what the best setting is at 4B.

### Finding 12 — Another loose end: my TextMAS is half as talkative as theirs

My TextMAS generates 2,191 words per problem. The paper's generates 4,420. Same
model, same task, same word budget. I don't know why. It matters because the
whole saving comes from silencing agents — so if their agents were twice as
verbose, they had twice as much to save.

---

## Part 5 — What's still running

- **GPQA-Diamond at 14B, all three setups** (jobs 55761333–5). This is a
  hard science benchmark from the paper's Table 2, where the paper reports its
  biggest speedups (6–7×). It's the first run here that tests the headline claim in
  the configuration the paper makes it.
- **Two runs with `<think>` turned off** (jobs 55764319–20). The code injects a
  literal `<think>` tag to force reasoning mode; the paper's printed prompts don't
  show one; and 48.9% of my 4B LatentMAS answers never finish reasoning. If turning
  it off recovers a lot of accuracy, part of the gap was my setup.

---

## Part 6 — Notes on the cluster, for anyone repeating this

- Every GPU partition I can use is **preemptible** — a paying job can cancel mine
  at any moment, and it's a cancel, not a pause. Hence the checkpointing. The
  job script resubmits itself when killed.
- On this account, a hidden folder of Python packages leaks into every
  environment and breaks GPU libraries. Every script sets `PYTHONNOUSERSITE=1`.
- The model code hardcodes 16-bit "bfloat16" math. Do not run it on older GPUs
  (V100, RTX6000) that don't support it.

## Layout

| folder | what's in it |
|---|---|
| `architecture/` | The authors' method code (`run.py`, `models.py`, `prompts.py`, `data.py`, `utils.py`, `methods/`), with the four fixes from Part 2. Run as `python architecture/run.py ...`. |
| `yajat/` | Everything I wrote: the job scripts, the measurement tools, and a longer findings log (`yajat/README.md`). |
| `yajat/results/` | Distilled result files (JSON). Raw logs are not tracked — they're multi-megabyte. |
| `data/`, `assets/`, `example_logs/` | Upstream data, figures, and the authors' two released logs. |

The root of the repo has only this README, the license, the requirements, and
`.gitignore`.

## License

Upstream code is under the original project's license; see `LICENSE`.
