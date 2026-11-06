# Prompt for Claude Code on HPC3: inventory, reproduce LatentMAS on Qwen3-4B and 8B, pilot the audit, clean up the code

Paste everything below the line into Claude Code, started from the repo folder on a GPU node.

---

You are working in my LatentMAS research repo on UCI's HPC3 cluster. I am Yajat, a student. Work through the phases below in order. Each phase has a goal, steps and a "done when" check. At the end of each phase, append a short section to `results/REPORT.md` (create it if needed). Then go on to the next phase. Stop and ask me only when a rule below says to.

## Context

- Paper: *Latent Collaboration in Multi-Agent Systems* (LatentMAS), arXiv 2511.20639. Upstream code: `github.com/Gen-Verse/LatentMAS` at commit `9a9e4d3`.
- The method: three agents (Planner, Critic, Refiner) never write text. Each one reads its prompt on top of a shared KV cache, then takes `latent_steps` steps. Each step feeds the last-layer hidden state back in as the next input, after `e = rescale(h · W_a)`. The Judger then writes the answer on top of the whole cache. `W_a` is a ridge-regression map from output to input embeddings, turned on by `--latent_space_realign`. Qwen3-4B has tied embeddings, so `W_a` is about the identity there. Qwen3-8B is untied.
- Known pitfalls in upstream `run.py`/`models.py` (already confirmed, do not re-investigate):
  1. `--latent_steps` defaults to 0. At 0 the Judger gets no cache at all (`methods/latent_mas.py:158`), so the run is a single Judger prompt.
  2. `W_a` is off unless `--latent_space_realign` is passed.
  3. Batched latent runs (`--generate_bs` > 1) build all-ones attention masks over the past cache (`models.py:241, 296, 334`), so padding leaks between agents. Use `--generate_bs 1` for `latent_mas`. Baseline and TextMAS do not pass a cache, so batching them is safe.
  4. The vLLM path mis-scores code tasks. Never use `--use_vllm`.
  5. Unconditional imports of vllm and matplotlib, `import pdb`, and `Qwen3-8B` missing from the `--model_name` choices. `hpc3/repro-fixes.patch` fixes these and pins `transformers==4.57.1` (5.x breaks `_past_length`).
  6. MedQA gold labels can silently reuse the previous item's letter (`data.py:210-216`).
- The `experiments/` package is my own fixed pipeline (left padding, real masks carried across agents, correct RoPE positions). Scripts: `exp0`–`exp4`, `exp5_content_audit.py` (pause tokens, swapped memory, identity/ridge/random W_a, adaptive steps), `exp6_decode_messages.py` (logit lens and activation patching), `analyze_audit.py`. `python -m experiments.test_experiments` runs CPU checks.
- Helper scripts in `hpc3/`: `run_l40s.sh` (batch wrapper), `reproduce.sh` (all reproduction configs, resumable), `inventory_results.py` (summarizes every result file).
- The newest versions of `experiments/` and `hpc3/` are in `~/hpc3_bundle.zip`. Before Phase 1, unzip it to `/tmp/hpc3_bundle`, `diff -r` it against the repo, and commit the repo's current state. Then copy the bundle's files in. Report any file where the repo version had changes the bundle lacks, and keep those changes, merging by hand if needed. If the zip is missing, stop and ask me to upload it.

## Cluster rules (HPC3, Slurm)

- My accounts: `EECS195221_CLASS_GPU` (charged), `ynagaraj`. The `gpu32` partition is not available to me.
- GPUs: partition `gpu` (A30 24 GB, A100, V100) with `-A EECS195221_CLASS_GPU`, and `free-gpu32` (L40S 48 GB, free but preemptible, no `-A`). Do not use V100s: they have no bf16.
- Qwen3-4B (about 8 GB) and Qwen3-8B (about 16 GB) fit on an A30. Use `--gres=gpu:A30:1` or `--gres=gpu:L40S:1`.
- The interactive session I'm in ends after a few hours. Anything longer than about 30 minutes must go in a batch job (`sbatch`), so it survives when I disconnect. Never run `srun` from inside an existing allocation. Never run GPU work on a login node.
- Set `export HF_HOME=/pub/$USER/hf_cache` so model downloads don't fill my home quota.
- Check your jobs with `squeue -u $USER`. Cancel anything you started that is stuck with `scancel`. Report every job ID you submit.

## Safety rules

- Never delete or overwrite an existing result or log file. Write new results under `results/` with names that say what they are.
- Before changing any code, make sure the work is committed, then create a branch (`git checkout -b cleanup-ste`). Commit in small steps with clear messages. Do not push unless I ask.
- If a run fails, do not retry it more than once. Read the end of its log, write what failed and why in the report, and go on with the rest.
- Stop and ask me before: spending more than about 20 GPU-hours in total on the class account, changing upstream behaviour in a way that changes results, or anything you can't undo.

## Phase 1: inventory what is already here

Goal: know what experiments were already run, with which settings, and whether each is trustworthy.

1. Run `git status`, `git log --oneline -15` and `ls`. List the folders and any results, logs or output files.
2. Run `python hpc3/inventory_results.py . --csv results/inventory.csv`.
3. For each row, decide whether it is trustworthy: the method was on (latent steps > 0), batch size 1 for latent runs, finished, and the full test set. Compare it with the paper's number (the `paper` and `gap` columns). One standard error is sqrt(p(1-p)/n). Treat gaps under 2 standard errors as "matches".
4. Check `squeue -u $USER` for jobs still running, and say what they are.

Done when: `REPORT.md` has a "Phase 1: existing results" section, containing a table of what exists, which results are trustworthy, and what is missing.

## Phase 2: check the environment

1. Confirm the conda env (`conda activate latentmas` or whichever env has torch). Check `python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"` and `nvidia-smi`.
2. Check that `hpc3/repro-fixes.patch` is applied (`git apply --check -R hpc3/repro-fixes.patch` succeeds if it is). Apply it if not.
3. Run `python -m experiments.test_experiments`. All checks must pass.
4. Smoke test: run each method on 4 questions to check that the code paths work before spending GPU hours:
   `OUT=results/smoke MODELS=Qwen/Qwen3-4B TASKS=gsm8k LAYOUTS=sequential STEPS=20 MAX_SAMPLES=4 MAX_NEW=512 bash hpc3/reproduce.sh`
   The `OUT=results/smoke` keeps these 4-question logs apart from the real runs.

Done when: every check passes, or the report says exactly which one fails and why.

## Phase 3: reproduce the paper on Qwen3-4B, then Qwen3-8B

Goal: match the paper's Table 1 within about 2 standard errors, with the method actually switched on.

1. Start with Qwen3-4B on MBPP+ (378 items) and GSM8K (1,319 items), sequential layout:
   `MODE=submit SBATCH_ARGS="-p gpu -A EECS195221_CLASS_GPU --gres=gpu:A30:1 -c 4 --mem=32G -t 1-00:00:00" MODELS=Qwen/Qwen3-4B TASKS="mbppplus gsm8k" LAYOUTS=sequential bash hpc3/reproduce.sh`
   If A30s are busy, use `SBATCH_ARGS="-p free-gpu32 --gres=gpu:L40S:1 -c 4 --mem=32G -t 1-00:00:00 --requeue"` instead.
2. While they run, check progress every so often with `squeue` and `tail` on the logs. Don't sit in a polling loop. When they finish, run the inventory script again.
3. Paper targets (Single / TextMAS / LatentMAS), sequential: Qwen3-4B MBPP+ 63.5 / 69.8 / 73.5, GSM8K 82.4 / 89.8 / 88.2. Qwen3-8B MBPP+ 64.8 / 69.5 / 74.6, GSM8K 81.1 / 92.3 / 93.8. `inventory_results.py` holds the full table for 4B and 8B (both layouts).
4. If the 4B results are within about 2 standard errors, submit the same for Qwen3-8B. Then add the hierarchical layout and the other tasks (`humanevalplus medqa arc_challenge arc_easy`). Estimate the GPU-hours first. If the total would pass about 20 GPU-hours on the class account, put the rest on `free-gpu32` or ask me.
5. If a result is far from the paper (more than 3 standard errors), check the usual causes first: wrong latent steps, batch size, truncated thinking (look for Judger outputs without `</think>`, and for "no python code block" errors), and the transformers version.

Done when: the report has a table with columns model, task, layout, method, latent steps, realign, n, ours, paper, gap, gap in SE, and a one-paragraph verdict on each model.

## Phase 4: pilot the content audit (cheap, no generation)

Run the teacher-forced pilot on both models and report the numbers it prints:
```
for M in Qwen/Qwen3-4B Qwen/Qwen3-8B; do
  python -m experiments.exp5_content_audit   --model_name $M --task gsm8k --max_samples 200 --latent_steps 20 --batch_size 8 --no_generate
  python -m experiments.exp6_decode_messages --model_name $M --task gsm8k --max_samples 200 --latent_steps 20 --batch_size 8
done
```
Run it as one batch job if it will take longer than about 30 minutes.

Report: the decomposition table (agent prompts, extra positions, any real message, question-specific content) in log p(gold), with its intervals. Also the mean cos(ridge, identity) for each model. My prediction is about 1.0 for 4B (tied) and clearly lower for 8B (untied), so say whether that holds. Also the per-agent patching deltas and the logit-lens gold ranks. Do not interpret beyond what the numbers show. Say plainly when an interval includes zero.

## Phase 5: clean up the codebase (on the `cleanup-ste` branch)

Goal: code that a new student can read on the first pass, with **no change in behaviour**.

Scope: my own code (`experiments/`, `hpc3/`, `analysis/`, any scripts I added). For the upstream files (`run.py`, `models.py`, `methods/`, `prompts.py`, `data.py`, `utils.py`), only add docstrings and comments, remove dead code (`import pdb`, unused imports), and fix names that mislead. Do not change their logic: the reproduction depends on it.

Writing rules (Clear Technical English) for every docstring, comment, name and the README:
- Start each module and function docstring with what it does and why it exists, in one sentence.
- One idea per sentence. Common, concrete words. Keep the correct technical term, and define it the first time.
- Expand acronyms on first use (KV cache = key/value cache).
- Name the actor and use the active voice: "The Judger reads the cache", not "the cache is read".
- State cause and effect directly. Put steps in the order they happen.
- Give tensor shapes in comments where a tensor is created or reshaped, for example `# [batch, positions, hidden]`.
- Comments explain *why*, not what the next line does.
- Function names are verbs that say what the function returns or does (`build_latent_memory`, `score_answer`). Variables are nouns that say what they hold. No one-letter names except loop indices and math symbols that match the paper (h, e, W_a), and define those in a comment.
- Remove filler, vague intensifiers and decorative words.
- Separate facts, assumptions and TODOs: mark assumptions `# Assumption:` and open items `# TODO:`.

Also:
- Add type hints to my own functions.
- Remove duplicated code if one shared helper can replace it, but only within `experiments/`.
- Update `experiments/README.md` so that each script has one line: what it compares, and the command to run it.

Check that behaviour is unchanged: run `python -m experiments.test_experiments` before and after. Also run `python -m experiments.exp5_content_audit --toy --model_name /tmp/tiny-qwen3 --max_samples 3 --latent_steps 4 --no_generate --seed 0` before and after, and diff the printed numbers. They must be identical. Commit each step separately.

Done when: the tests pass, the before/after numbers are identical, and the report lists what changed, file by file.

## Final report

End with a short summary at the top of `results/REPORT.md`:
- what already existed and whether it can be trusted
- the reproduction table and verdict
- the audit pilot numbers
- the cleanup changes
- every problem you hit, with the log path and the fix or the open question
- the job IDs still running, and what I should run next

Then print the path to the report and the summary in the terminal.
