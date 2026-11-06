"""Combine exp5 content-audit runs across seeds, to report one table with error bars and paired tests.

For each condition, the script prints the mean accuracy over seeds and the range across seeds.
For each part of the decomposition, the script prints:

  * the accuracy difference averaged over seeds,
  * a 95% paired bootstrap interval over questions (it first averages each question over seeds),
  * a Holm-corrected p-value from a paired sign-flip permutation test.

Usage:
    python -m experiments.analyze_audit experiments/results/exp5_Qwen3-4B_gsm8k_m20_seed*.jsonl
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from experiments.audit_utils import compute_bootstrap_ci
from experiments.exp5_content_audit import DECOMPOSITION


def load_audit_rows(paths: List[str]) -> Tuple[Dict, Dict]:
    """Read exp5 JSONL files and return (correct, logp).

    correct[condition][seed][index] is 0.0 or 1.0.
    logp[condition][index] is a list with one value per seed.
    """
    correct = defaultdict(lambda: defaultdict(dict))
    logp = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path) as jsonl_file:
            for line in jsonl_file:
                row = json.loads(line)
                if row.get("correct") is not None:
                    correct[row["condition"]][row["seed"]][row["index"]] = float(bool(row["correct"]))
                if row.get("gold_logprob") is not None:
                    logp[row["condition"]][row["index"]].append(row["gold_logprob"])
    return correct, logp


def average_correct_per_question(correct_by_seed: Dict[int, Dict[int, float]]) -> Tuple[List[int], torch.Tensor]:
    """Return the question indices that all seeds share, and each question's mean correctness over seeds."""
    indices = sorted(set.intersection(*(set(by_index) for by_index in correct_by_seed.values())))
    correct_table = torch.tensor([[correct_by_seed[seed][index] for seed in sorted(correct_by_seed)]
                                  for index in indices])  # [questions, seeds]
    return indices, correct_table.mean(1)


def compute_permutation_p(diffs: torch.Tensor, n_perm: int = 20000, seed: int = 0) -> float:
    """Return the two-sided p-value of a paired sign-flip permutation test on the mean of `diffs`."""
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n_perm, len(diffs)), generator=generator).double() * 2 - 1  # [n_perm, questions]
    null = (signs * diffs).mean(1).abs()
    return ((null >= diffs.mean().abs() - 1e-12).sum().item() + 1) / (n_perm + 1)


def adjust_p_values_holm(p_values: List[float]) -> List[float]:
    """Return Holm-adjusted p-values, which control the error rate over several tests, in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def main() -> None:
    """Print the per-condition table, then the accuracy and log p(gold) decompositions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    correct, logp = load_audit_rows(args.paths)

    print(f"{'condition':<18}{'seeds':>6}{'n':>6}{'mean acc':>10}{'seed range':>18}")
    for condition, by_seed in correct.items():
        accuracies = [sum(by_index.values()) / len(by_index) for by_index in by_seed.values()]
        n_questions = len(next(iter(by_seed.values())))
        print(f"{condition:<18}{len(accuracies):>6}{n_questions:>6}{sum(accuracies) / len(accuracies):>10.3f}"
              f"{f'{min(accuracies):.3f}-{max(accuracies):.3f}':>18}")

    results = []
    for name, condition_x, condition_y in DECOMPOSITION:
        if condition_x not in correct or condition_y not in correct:
            continue
        indices_x, means_x = average_correct_per_question(correct[condition_x])
        indices_y, means_y = average_correct_per_question(correct[condition_y])
        common = sorted(set(indices_x) & set(indices_y))
        paired_x = torch.tensor([means_x[indices_x.index(i)].item() for i in common], dtype=torch.float64)
        paired_y = torch.tensor([means_y[indices_y.index(i)].item() for i in common], dtype=torch.float64)
        diffs = paired_x - paired_y
        results.append((name, diffs.mean().item(), compute_bootstrap_ci(diffs), compute_permutation_p(diffs),
                        len(common)))

    if results:
        adjusted = adjust_p_values_holm([result[3] for result in results])
        print(f"\n{'part (X minus Y)':<28}{'acc diff':>10}{'95% CI':>20}{'p (Holm)':>10}{'n':>6}")
        for (name, mean, (lower, upper), _, n_common), p_value in zip(results, adjusted):
            print(f"{name:<28}{mean:>10.3f}{f'[{lower:.3f}, {upper:.3f}]':>20}{p_value:>10.4f}{n_common:>6}")

    if logp:
        print(f"\n{'part (X minus Y)':<28}{'log p(gold) diff':>18}{'95% CI':>22}")
        for name, condition_x, condition_y in DECOMPOSITION:
            if condition_x not in logp or condition_y not in logp:
                continue
            logp_x, logp_y = logp[condition_x], logp[condition_y]
            common = sorted(set(logp_x) & set(logp_y))
            diffs = torch.tensor([sum(logp_x[i]) / len(logp_x[i]) - sum(logp_y[i]) / len(logp_y[i])
                                  for i in common], dtype=torch.float64)
            lower, upper = compute_bootstrap_ci(diffs)
            print(f"{name:<28}{diffs.mean().item():>18.3f}{f'[{lower:.3f}, {upper:.3f}]':>22}")


if __name__ == "__main__":
    main()
