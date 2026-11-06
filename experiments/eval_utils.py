"""Data loading, scoring and reporting shared by the experiments.

Scoring reuses the repo's own answer extractors so numbers stay comparable with run.py.
Reporting is paired: every condition is run on the same questions, so we compare
per-question correctness, not just two accuracies.
"""

import argparse
import json
import math
from itertools import islice
from pathlib import Path
from typing import Dict, List

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

# A handful of offline items so every script can be smoke-tested without downloads.
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
    parser.add_argument("--output_dir", default="experiments/results")
    parser.add_argument("--toy", action="store_true", help="Use a few built-in questions instead of the dataset.")


def load_items(task: str, max_samples: int, toy: bool = False) -> List[Dict]:
    if toy:
        return TOY_ITEMS["mbppplus" if task in CODE_TASKS else "gsm8k"][:max_samples]
    items = DATASET_LOADERS[task]()
    return list(islice(items, max_samples)) if max_samples > 0 else list(items)


def run_code_with_tests(code: str, tests: str, timeout: int = 10):
    """Run `code` followed by `tests` in a subprocess. Returns (passed, error_message)."""
    return run_with_timeout(code + "\n" + tests, timeout=timeout)


def score_answer(task: str, answer_text: str, item: Dict) -> Dict:
    """Same rules as methods/latent_mas.py, returned as a dict."""
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


def mcnemar_exact_p(only_a_correct: int, only_b_correct: int) -> float:
    """Two-sided exact McNemar test on the questions where the two conditions disagree."""
    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0
    k = min(only_a_correct, only_b_correct)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def print_paired_report(correct_by_condition: Dict[str, List[bool]], baseline: str) -> None:
    """Accuracy per condition, and wins/losses against the baseline on the same questions."""
    base = correct_by_condition[baseline]
    print(f"\n{'condition':<28}{'accuracy':>10}{'wins':>6}{'losses':>8}{'p (McNemar)':>13}")
    for name, correct in correct_by_condition.items():
        accuracy = sum(correct) / max(1, len(correct))
        if name == baseline:
            print(f"{name:<28}{accuracy:>10.3f}{'-':>6}{'-':>8}{'(baseline)':>13}")
            continue
        wins = sum(c and not b for c, b in zip(correct, base))
        losses = sum(b and not c for c, b in zip(correct, base))
        print(f"{name:<28}{accuracy:>10.3f}{wins:>6}{losses:>8}{mcnemar_exact_p(wins, losses):>13.3f}")


def save_jsonl(rows: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    print(f"Saved {len(rows)} rows to {path}")
