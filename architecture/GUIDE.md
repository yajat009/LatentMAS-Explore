# Guide to the words, settings, benchmarks and files

Use this guide when a word or a setting in this repo is unclear. It has five parts:

1. [The method in plain words](#1-the-method-in-plain-words)
2. [Word list](#2-word-list)
3. [Settings](#3-settings)
4. [Benchmarks](#4-benchmarks)
5. [Every file](#5-every-file)

## 1. The method in plain words

The system runs one language model four times, with four different job descriptions:

| Order | Agent | Job |
|---|---|---|
| 1 | Planner | Make a plan |
| 2 | Critic | Check the plan |
| 3 | Refiner | Improve the plan |
| 4 | Judger | Write the final answer |

The agents can pass their work along in two ways:

- **TextMAS:** each agent writes text, and the next agent reads the text. This is what most multi-agent systems do.
- **LatentMAS:** the first three agents write no text. They pass the model's internal memory to the next agent. Only the Judger writes.

The paper claims that the second way is as accurate or more accurate, and faster.

## 2. Word list

| Word | Meaning |
|---|---|
| **Token** | A piece of text that the model reads or writes, about three quarters of a word. |
| **Hidden state** | The list of numbers that the model holds for one token. It is the model's internal meaning of that position. |
| **Embedding** | The list of numbers that the model takes in for one token. The model's input is embeddings, and its internal result is hidden states. |
| **KV cache** | The model's memory of everything it has read so far. KV means key/value, the two kinds of stored numbers. It exists so that the model does not reread the text for each new token. |
| **Shared memory** | In this repo, this means the KV cache that the agents pass along. Each agent adds to it, and the Judger reads all of it. The paper calls it "latent working memory". |
| **Latent** | Internal: in numbers, not in text. |
| **Latent step** | One silent thinking step. The model takes its last hidden state and feeds it back in as if it were the next input. No text appears, and the memory grows by one position. |
| **Latent steps = 0** | No silent steps. In this code, the Judger then also gets no shared memory, so it answers alone. |
| **`W_a` (realignment matrix)** | A fixed conversion from hidden states to embeddings, because the two are different kinds of numbers. Without it, the code feeds the hidden state in unconverted, and only rescales its length. |
| **Tied embeddings** | The model uses one table for input and output. Qwen3-4B does, so `W_a` changes nothing there. Qwen3-8B and 14B do not. |
| **Batch size** | How many questions the GPU runs at once. |
| **Padding** | Filler tokens that make short prompts as long as the longest one in a batch. The bug in this code lets the model read the filler. |
| **Temperature, top-p** | Settings for how random the model's word choice is. |
| **Seed** | A number that fixes the randomness, so that a run can repeat. |
| **`<think>`** | The tag that starts the reasoning section of a Qwen3 answer. `</think>` ends it, and the final answer comes after. |
| **Checkpoint** | A file that saves each finished question, so that a stopped run can continue. |
| **Baseline, Single** | One model, one prompt, no agents. This is the comparison point. |
| **Ablation** | A test that removes one part to see what that part does. |
| **Paired test (McNemar)** | Compares two runs question by question. A "win" is a question that only one run got right. A small p-value (below 0.05) means that the difference is unlikely to be luck. |
| **log p(gold)** | How likely the model finds the correct answer. Closer to 0 is better. The audit uses it in place of accuracy. |
| **Teacher forcing** | Scoring a given answer without letting the model write its own. |

## 3. Settings

### `architecture/run.py`: settings that change a result

| Setting | Default | What it does | What to use |
|---|---|---|---|
| `--method` | required | `baseline`, `text_mas` or `latent_mas` | The method to test |
| `--model_name` | required | `Qwen/Qwen3-4B`, `8B` or `14B`. A larger number means a larger model. | The model to test |
| `--task` | `gsm8k` | Which benchmark to run. See [Benchmarks](#4-benchmarks). | The benchmark to test |
| `--latent_steps` | **0** | Silent steps per agent. At 0, the shared memory is not used. | Always set it |
| `--generate_bs` | **20** | Batch size | **1** for `latent_mas` |
| `--think` | off | Adds the text `<think>` to the prompt. The model reasons with or without it. | Report it for every run |
| `--latent_space_realign` | off | Turns on `W_a` | On for 8B and 14B. It changes nothing on 4B. |
| `--prompt` | `sequential` | The agent layout. `sequential` is Planner, Critic, Refiner, Judger. `hierarchical` is three subject experts and a summarizer. | `sequential` |
| `--max_new_tokens` | 4096 | The longest answer allowed. A long answer that hits this limit fails. | 4096 |
| `--max_samples` | -1 | How many questions to run. -1 means all of them. | -1 for a final result |
| `--temperature` | 0.6 | Randomness of word choice | Keep it |
| `--top_p` | 0.95 | Limits word choice to the likeliest words | Keep it |
| `--seed` | 42 | Fixes the randomness | Use 3 different seeds for a final result |

### `architecture/run.py`: settings for special tests

| Setting | What it does |
|---|---|
| `--pad_fix` | Turns on our fix for the padding bug. No full run has tested it. |
| `--sequential_info_only` | Each agent passes on only what it added, not the full history. |
| `--latent_only` | Passes on only the silent steps, not the prompts. |
| `--text_mas_context_length` | Limits how much earlier text a TextMAS agent reads. |

### `architecture/run.py`: settings that do not change a result

| Setting | What it does |
|---|---|
| `--checkpoint` | The file that saves progress |
| `--device` | Which GPU to use |
| `--split` | Which part of the dataset to use. Keep `test`. |
| `--use_vllm`, `--enable_prefix_caching`, `--use_second_HF_model`, `--device2`, `--tensor_parallel_size`, `--gpu_memory_utilization` | Settings for vLLM, a faster engine. **Do not use vLLM here**, because it mis-scores code tasks. |

### `results/test_runs/` scripts

These scripts use the same ideas with slightly different names and defaults:

| Setting | Meaning |
|---|---|
| `--batch_size` | Same as `--generate_bs`. Batching is safe here, because these scripts fix the padding bug. |
| `--max_samples` | Default 50, not all questions |
| `--latent_steps` | Default 20, not 0 |
| `--toy` | Uses 3 built-in questions, for a quick check |
| `--no_generate` | Content audit only: scores the correct answer and writes no text. This is much faster. |
| `--conditions` | Content audit only: which memory conditions to test (see below) |
| `--mapping` | Message decoding only: `identity` (no `W_a`) or `ridge` (with `W_a`) |

The content audit gives the Judger seven kinds of memory:

| Condition | What the Judger gets | What it tests |
|---|---|---|
| `judger_alone` | No memory | The starting point |
| `prompts_only` | The agents' prompts, with no silent steps | The effect of the prompts |
| `pause` | Blank silent steps | The effect of extra positions alone |
| `swapped` | Real memory from a **different** question | Whether the memory needs to match the question |
| `latent_identity` | Real memory, without `W_a` | The method as the code runs it by default |
| `latent_ridge` | Real memory, with `W_a` | The method as the paper describes it |
| `latent_random` | Real memory, with a random conversion | Whether `W_a` has to be the right one |

## 4. Benchmarks

Each benchmark is a set of questions with known answers. Accuracy is the share of questions answered correctly.
The difficulty column is the paper's single-model score on Qwen3-4B, where the paper gives one. A lower score means a harder benchmark.

| `--task` | Questions | What it asks | Answer form | Scoring | Paper's 4B score | Tested here? |
|---|---|---|---|---|---|---|
| `arc_easy` | 2,376 | Easy school science | Multiple choice | Exact letter | 95.4 | No |
| `arc_challenge` | 1,172 | Harder school science | Multiple choice | Exact letter | 89.2 | No |
| `gsm8k` | 1,319 | School math word problems | A number | Exact string | 82.4 | Yes, first 100 at most |
| `humanevalplus` | 164 | Write a Python function | Code | Runs hidden tests | 75.0 | No |
| `mbppplus` | 378 | Write a short Python function | Code | Runs hidden tests | 63.5 | **Yes, the main test** |
| `medqa` | 300 | Medical exam questions | Multiple choice | Exact letter | 47.7 | No |
| `gpqa` | 198 | Graduate-level science | Multiple choice | Exact letter | 39.9 on 8B (no 4B score) | Yes, on 14B |
| `aime2024` | 30 | Hard math contest | A number | Exact number | 50.0 on 8B (no 4B score) | No |
| `aime2025` | 30 | Hard math contest | A number | Exact number | 46.7 on 8B (no 4B score) | No |


## 5. Every file

### Top level

| File | Purpose |
|---|---|
| `README.md` | Front page of the repo |
| `LICENSE` | The upstream license |
| `requirements.txt` | Python packages to install |
| `.gitignore` | Files that git must not save |
| `data/medqa.json` | The MedQA questions. The other benchmarks download on first use. |

### `architecture/`: the original method

| File | Purpose |
|---|---|
| `README.md` | How the method works, and every change from upstream |
| `GUIDE.md` | This guide |
| `run.py` | **Start here.** The command you run. It reads the settings and runs the questions. |
| `models.py` | Loads the model. Holds the silent-step loop (`generate_latent_batch`) and `W_a`. |
| `methods/latent_mas.py` | **The method.** Runs the four agents and passes the memory. |
| `methods/text_mas.py` | The comparison method, where agents write text |
| `methods/baseline.py` | The comparison method with one model alone |
| `methods/__init__.py` | Lists the four agents |
| `prompts.py` | The text instructions for each agent |
| `data.py` | Loads each benchmark |
| `utils.py` | Reads the answer from the model's output and runs code safely |
| `__init__.py` | Marks the folder as a Python package |
| `assets/` | The paper's figures, tables, logo and slides |
| `example_logs/` | The two run logs that the authors released. They are key evidence. |

### `results/`: what we found

| File | Purpose |
|---|---|
| `README.md` | **Start here.** The results on one page. |
| `details/` | One folder per question, with full numbers. `details/README.md` lists corrected claims. |
| `data/runs.csv` | One row per run: its settings and score |
| `data/question_outcomes.csv` | One row per run and question: right or wrong |
| `data/content_audit_scores.csv` | One row per audit question and condition |
| `raw/` | Raw logs. They are on your Mac and HPC3, and not on GitHub. |

### `results/test_runs/`: the scripts that ran the tests

| File | Purpose |
|---|---|
| `README.md` | Commands, and a table of old file names |
| `run_content_audit.py` | Tests the seven memory conditions above |
| `run_message_decoding.py` | Swaps one agent's memory at a time, and reads each silent step as words |
| `run_memory_controls.py` | A smaller memory test, with real accuracy scores |
| `run_adaptive_latent_steps.py` | Stops silent steps early. The content audit uses its stop rule. |
| `summarize_content_audit.py` | Combines content-audit runs from several seeds |
| `check_batch_padding.py` | Checks that batching does not change answers in this pipeline |
| `test_experiments.py` | Quick CPU checks of the code |
| `latent_pipeline.py` | Shared code: our corrected version of the four-agent pipeline |
| `latent_mapping.py` | Shared code: the ways to convert a hidden state to an embedding |
| `audit_utils.py` | Shared code: the audit's conditions and scoring |
| `eval_utils.py` | Shared code: settings, data loading and scoring |
| `make_tiny_model.py` | Builds a very small model for the CPU checks |
| `__init__.py` | Lets these scripts find `architecture/` |

`results/test_runs/analysis/` reads finished runs:

| File | Purpose |
|---|---|
| `recompute_result_tables.py` | **Recomputes every paired result** from `results/data/` |
| `export_compact_results.py` | Builds `results/data/` from the raw logs |
| `compare_methods_from_checkpoints.py` | Compares runs on accuracy, tokens and time |
| `count_tokens_from_logs.py` | Counts tokens in a finished log. It confirmed that the released logs match the paper. |
| `measure_realign_matrix.py` | Measures how much `W_a` changes a vector |
| `trace_kv_cache.py` | Prints how the memory grows, agent by agent |
| `test_truncate_past.py` | Tests the memory-trimming function |

`results/test_runs/cluster/` runs jobs on HPC3:

| File | Purpose |
|---|---|
| `README.md` | Cluster facts, such as accounts and GPUs |
| `setup_environment.sh` | Installs the Python environment. Run it once. |
| `download_models_and_datasets.sh` | Downloads the models and data. Run it once. |
| `run_one_configuration.sbatch` | Runs one test as a job, and restarts it if the cluster stops it |
| `reproduce_paper_tables.sh` | Runs every paper configuration |
| `submit_latent_steps_sweep.sh` | Runs one test at several latent-step values |
| `run_on_free_l40s.sh` | Runs any command on a free GPU |
