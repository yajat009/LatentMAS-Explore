"""Load data, score answers and print reports, so that all experiments use the same rules.

The scoring reuses the repo's answer extractors, so the numbers stay comparable with run.py.
The reports are paired: every condition runs on the same questions, so the code compares
correctness question by question and not only two accuracies.
"""

import argparse
import json
import math
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import data
from utils import extract_gsm8k_answer, extract_markdown_python_block, normalize_answer, run_with_timeout

CODE_TASKS = {"mbppplus", "humanevalplus"}

DATASET_LOADERS = {
    "gsm8k": data.load_gsm8k,
    "aime2024": data.load_aime2024,
    "aime2025": data.load_aime2025,
    "gpqa": data.load_gpqa_diamond,
    "arc_easy": data.load_arc_easy,
    "arc_challenge": data.load_arc_challenge,
    "mbppplus": data.load_mbppplus,
    "humanevalplus": data.load_humanevalplus,
    "medqa": data.load_medqa,
}

# These offline items let every script run a smoke test without a download.
TOY_ITEMS = {
    "gsm8k": [
        {"question": "Tom has 3 apples and buys 4 more. How many apples does he have?", "gold": "7"},
        {"question": "A box holds 6 eggs. How many eggs are in 5 boxes?", "gold": "30"},
        {"question": "Sara had 20 dollars and spent 8. How many dollars are left?", "gold": "12"},
    ],
    "mbppplus": [
        {"question": "Write a function add(a, b) that returns the sum of a and b.\n"
                     "Your answer will be tested on test cases like:\nassert add(1, 2) == 3\n",
         "gold": "assert add(1, 2) == 3\nassert add(-1, 1) == 0\n"},
        {"question": "Write a function is_even(n) that returns True if n is even.\n"
                     "Your answer will be tested on test cases like:\nassert is_even(4) == True\n",
         "gold": "assert is_even(4) == True\nassert is_even(7) == False\n"},
    ],
}


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags that every experiment script shares."""
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--task", default="gsm8k", choices=sorted(DATASET_LOADERS))
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--latent_steps", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--think", action="store_true", help="Append <think> to every prompt, as run.py --think does.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu.")
    parser.add_argument("--output_dir", default="results/raw/test_runs")
    parser.add_argument("--toy", action="store_true", help="Use a few built-in questions instead of the dataset.")


def add_adaptive_step_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags that control adaptive latent steps, which the adaptive-steps and content-audit scripts share."""
    parser.add_argument("--max_latent_steps", type=int, default=80)
    parser.add_argument("--min_latent_steps", type=int, default=5)
    parser.add_argument("--cosine_threshold", type=float, default=0.99)


def load_items(task: str, max_samples: int, toy: bool = False) -> List[Dict]:
    """Return up to `max_samples` items, as dicts with "question" and "gold", from the dataset or TOY_ITEMS."""
    if toy:
        return TOY_ITEMS["mbppplus" if task in CODE_TASKS else "gsm8k"][:max_samples]
    items = DATASET_LOADERS[task]()
    return list(islice(items, max_samples)) if max_samples > 0 else list(items)


def iterate_batches(items: List[Dict], batch_size: int) -> Iterator[Tuple[int, List[Dict], List[str]]]:
    """Yield (index of the first item, items, question strings) for each batch in order."""
    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]
        yield batch_start, batch, [item["question"] for item in batch]


def select_donor_questions(items: List[Dict], batch_start: int, batch_len: int) -> List[str]:
    """Return the question that follows each batch item, so that a swapped control reads another question's memory.

    The last item wraps around to the first item.
    """
    return [items[(batch_start + offset + 1) % len(items)]["question"] for offset in range(batch_len)]


def run_code_with_tests(code: str, tests: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """Run `code` followed by `tests` in a subprocess, and return (passed, error_message)."""
    return run_with_timeout(code + "\n" + tests, timeout=timeout)


def score_answer(task: str, answer_text: str, item: Dict) -> Dict:
    """Score one answer with the rules of methods/latent_mas.py, and return the result as a dict."""
    gold = str(item.get("gold", "")).strip()

    if task in CODE_TASKS:
        code = extract_markdown_python_block(answer_text)
        if code is None:
            return {"prediction": None, "correct": False, "error": "no python code block"}
        passed, error = run_code_with_tests(code, gold)
        return {"prediction": code, "correct": bool(passed), "error": error}

    prediction = normalize_answer(extract_gsm8k_answer(answer_text))
    if task in {"aime2024", "aime2025"}:
        try:
            correct = int(prediction) == int(gold)
        except (TypeError, ValueError):
            correct = False
    else:
        correct = bool(prediction) and prediction == gold.lower()
    return {"prediction": prediction, "correct": correct, "error": None}


def compute_mcnemar_exact_p(only_a_correct: int, only_b_correct: int) -> float:
    """Return the two-sided exact McNemar p-value, computed on the questions where two conditions disagree."""
    disagreements = only_a_correct + only_b_correct
    if disagreements == 0:
        return 1.0
    smaller_count = min(only_a_correct, only_b_correct)
    tail = sum(math.comb(disagreements, i) for i in range(smaller_count + 1)) / 2 ** disagreements
    return min(1.0, 2 * tail)


def collect_correct_by_condition(rows: List[Dict[str, Any]], conditions: Iterable[str]) -> Dict[str, List[bool]]:
    """Return {condition: per-question correctness}, in the order of `rows`, as print_paired_report expects."""
    return {condition: [row["correct"] for row in rows if row["condition"] == condition]
            for condition in conditions}


def print_paired_report(correct_by_condition: Dict[str, List[bool]], baseline: str) -> None:
    """Print the accuracy per condition, and the wins and losses against the baseline on the same questions."""
    baseline_correct = correct_by_condition[baseline]
    print(f"\n{'condition':<28}{'accuracy':>10}{'wins':>6}{'losses':>8}{'p (McNemar)':>13}")
    for name, correct in correct_by_condition.items():
        accuracy = sum(correct) / max(1, len(correct))
        if name == baseline:
            print(f"{name:<28}{accuracy:>10.3f}{'-':>6}{'-':>8}{'(baseline)':>13}")
            continue
        pairs = list(zip(correct, baseline_correct))
        wins = sum(this_correct and not base_correct for this_correct, base_correct in pairs)
        losses = sum(base_correct and not this_correct for this_correct, base_correct in pairs)
        print(f"{name:<28}{accuracy:>10.3f}{wins:>6}{losses:>8}{compute_mcnemar_exact_p(wins, losses):>13.3f}")


def get_model_tag(model_name: str) -> str:
    """Return the last path part of a model name or folder, for use in output file names."""
    return model_name.rstrip("/").split("/")[-1]


def save_jsonl(rows: List[Dict], path: str) -> None:
    """Write one JSON object per line to `path`, and print how many rows it wrote."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row, default=str) + "\n")
    print(f"Saved {len(rows)} rows to {path}")
