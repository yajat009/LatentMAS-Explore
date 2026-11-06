#!/usr/bin/env python3
"""Recompute every paired result in results/ from the CSV files in results/data/, with no GPU and no raw logs.

Each line compares two runs on the questions that both finished. A "win" is a question that
run B got right and run A got wrong. The p-value is the exact two-sided McNemar test.
The script also checks that both runs scored the same questions, and it stops if they did not.

Usage, from the repo root:  python results/test_runs/analysis/recompute_result_tables.py
"""

import csv
import random
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Dict, List, Tuple

DATA = Path("results/data")
# Each entry is (section title, list of (run A, run B)). Run names leave out the model and task prefix.
M4 = "Qwen3-4B_mbppplus_"
COMPARISONS = [
    ("01 Paper comparison, Qwen3-4B MBPP+", [
        (M4 + "baseline_steps0_bs1_think0", M4 + "latent_mas_steps0_bs1_think1"),
        (M4 + "baseline_steps0_bs1_think0", M4 + "latent_mas_steps20_bs1_think1"),
        (M4 + "baseline_steps0_bs15_think1", M4 + "latent_mas_steps20_bs1_think1"),
        (M4 + "baseline_steps0_bs15_think1", M4 + "text_mas_steps0_bs8_think1"),
        (M4 + "baseline_steps0_bs15_think1", M4 + "latent_mas_steps10_bs15_think1"),
    ]),
    ("02 Latent steps and the --think flag, Qwen3-4B MBPP+", [
        (M4 + "latent_mas_steps0_bs1_think1", M4 + "latent_mas_steps10_bs1_think1"),
        (M4 + "latent_mas_steps0_bs1_think1", M4 + "latent_mas_steps20_bs1_think1"),
        (M4 + "latent_mas_steps10_bs1_think1", M4 + "latent_mas_steps10_bs1_think0"),
        (M4 + "latent_mas_steps20_bs1_think1", M4 + "latent_mas_steps20_bs1_think0"),
        (M4 + "latent_mas_steps40_bs1_think1", M4 + "latent_mas_steps40_bs1_think0_realign"),
        (M4 + "latent_mas_steps10_bs1_think0", M4 + "latent_mas_steps20_bs1_think0"),
        (M4 + "latent_mas_steps0_bs1_think1", M4 + "latent_mas_steps20_bs1_think0"),
    ]),
    ("03 Model scale", [
        ("Qwen3-14B_mbppplus_latent_mas_steps0_bs1_think1", "Qwen3-14B_mbppplus_latent_mas_steps10_bs1_think1"),
        ("Qwen3-8B_mbppplus_baseline_steps0_bs4_think0", "Qwen3-8B_mbppplus_latent_mas_steps20_bs1_think1"),
        ("Qwen3-14B_gpqa_baseline_steps0_bs4_think1", "Qwen3-14B_gpqa_latent_mas_steps10_bs4_think1"),
        ("Qwen3-14B_gpqa_text_mas_steps0_bs4_think1", "Qwen3-14B_gpqa_latent_mas_steps10_bs4_think1"),
    ]),
    ("04 Padding, Qwen3-4B MBPP+", [
        (M4 + "latent_mas_steps10_bs1_think1", M4 + "latent_mas_steps10_bs15_think1"),
        (M4 + "latent_mas_steps10_bs8_think1", M4 + "latent_mas_steps10_bs15_think1"),
        (M4 + "baseline_steps0_bs1_think0", M4 + "baseline_steps0_bs15_think1"),
    ]),
    ("07 GSM8K, Qwen3-4B", [
        ("Qwen3-4B_gsm8k_baseline_steps0_bs1_think0", "Qwen3-4B_gsm8k_latent_mas_steps20_bs1_think1"),
        ("Qwen3-4B_gsm8k_baseline_steps0_bs1_think0", "Qwen3-4B_gsm8k_text_mas_steps0_bs1_think0"),
    ]),
]
# Each entry is (printed name, condition X, condition Y). The table shows X minus Y.
AUDIT_DIFFERENCES = [
    ("agent prompts", "prompts_only", "judger_alone"),
    ("extra positions", "pause", "prompts_only"),
    ("any real message", "swapped", "pause"),
    ("question-specific content", "latent_identity", "swapped"),
    ("ridge W_a against identity", "latent_ridge", "latent_identity"),
]


def compute_mcnemar_p(wins: int, losses: int) -> float:
    """Return the exact two-sided McNemar p-value for the discordant counts."""
    total = wins + losses
    if total == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(total, k) for k in range(min(wins, losses) + 1)) / 2 ** total)


def read_outcomes() -> Dict[str, List[Tuple[str, int, str]]]:
    """Return, for each run, its list of (question hash, correct, numeric-correct) in question order."""
    outcomes: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)
    with open(DATA / "question_outcomes.csv") as file:
        for row in csv.DictReader(file):
            outcomes[row["run_id"]].append((row["question_hash"], int(row["correct"]), row["correct_numeric"]))
    return outcomes


def print_comparison(outcomes: Dict, run_a: str, run_b: str) -> None:
    """Print accuracy and paired wins and losses for two runs on their shared first questions."""
    if run_a not in outcomes or run_b not in outcomes:
        print(f"  missing data: {run_a if run_a not in outcomes else run_b}")
        return
    shared = min(len(outcomes[run_a]), len(outcomes[run_b]))
    rows_a, rows_b = outcomes[run_a][:shared], outcomes[run_b][:shared]
    if any(a[0] != b[0] for a, b in zip(rows_a, rows_b)):
        raise ValueError(f"{run_a} and {run_b} did not score the same questions")
    correct_a, correct_b = sum(a[1] for a in rows_a), sum(b[1] for b in rows_b)
    wins = sum(b[1] and not a[1] for a, b in zip(rows_a, rows_b))
    losses = sum(a[1] and not b[1] for a, b in zip(rows_a, rows_b))
    print(f"  n={shared:3d}  {correct_a:3d} -> {correct_b:3d}  B wins {wins:3d}, B losses {losses:3d}, "
          f"p={compute_mcnemar_p(wins, losses):.2g}\n      A {run_a}\n      B {run_b}")


def compute_bootstrap_interval(differences: List[float], seed: int = 0, repeats: int = 4000) -> Tuple[float, float]:
    """Return a 95% percentile bootstrap interval for the mean of the paired differences."""
    generator = random.Random(seed)
    size = len(differences)
    means = sorted(sum(generator.choices(differences, k=size)) / size for _ in range(repeats))
    return means[int(0.025 * repeats)], means[int(0.975 * repeats)]


def print_content_audit() -> None:
    """Print the mean log p(gold) differences of the content audit, with their intervals."""
    scores: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    with open(DATA / "content_audit_scores.csv") as file:
        for row in csv.DictReader(file):
            scores[row["model"]][row["condition"]][int(row["question_index"])] = float(row["gold_logprob"])
    print("08 Content audit, GSM8K, 200 questions, log p(gold), X minus Y")
    for model, conditions in sorted(scores.items()):
        for name, condition_x, condition_y in AUDIT_DIFFERENCES:
            differences = [conditions[condition_x][i] - conditions[condition_y][i] for i in conditions[condition_x]]
            low, high = compute_bootstrap_interval(differences)
            print(f"  {model:9s} {name:27s} {sum(differences) / len(differences):+7.3f} [{low:+.3f}, {high:+.3f}]")


def main() -> None:
    """Print every paired comparison, then the numeric rescoring of GSM8K, then the content audit."""
    outcomes = read_outcomes()
    for title, pairs in COMPARISONS:
        print(title)
        for run_a, run_b in pairs:
            print_comparison(outcomes, run_a, run_b)
    print("07 GSM8K numeric rescoring (57.00 counts as 57)")
    for run_id, rows in sorted(outcomes.items()):
        if "_gsm8k_" in run_id:
            print(f"  {run_id}: string {sum(r[1] for r in rows)}/{len(rows)}, "
                  f"numeric {sum(int(r[2]) for r in rows)}/{len(rows)}")
    print_content_audit()


if __name__ == "__main__":
    main()
