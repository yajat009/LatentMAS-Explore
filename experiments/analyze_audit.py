"""Combine exp5 content-audit runs across seeds into one table with error bars and paired tests.

For each condition: mean accuracy over seeds and the spread (min to max across seeds).
For each part of the decomposition: the accuracy difference averaged over seeds, a 95% paired
bootstrap interval over questions (each question's correctness averaged over seeds first), and a
Holm-corrected p-value from a paired sign-flip permutation test.

Usage:
    python -m experiments.analyze_audit experiments/results/exp5_Qwen3-4B_gsm8k_m20_seed*.jsonl
"""

import argparse
import json
from collections import defaultdict

import torch

from experiments.exp5_content_audit import DECOMPOSITION


def load(paths):
    # correct[condition][seed][index] = 0/1 ; logp[condition][index] = list over seeds
    correct = defaultdict(lambda: defaultdict(dict))
    logp = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("correct") is not None:
                    correct[r["condition"]][r["seed"]][r["index"]] = float(bool(r["correct"]))
                if r.get("gold_logprob") is not None:
                    logp[r["condition"]][r["index"]].append(r["gold_logprob"])
    return correct, logp


def per_question_mean(correct_by_seed):
    indices = sorted(set.intersection(*(set(d) for d in correct_by_seed.values())))
    return indices, torch.tensor([[correct_by_seed[s][i] for s in sorted(correct_by_seed)] for i in indices]).mean(1)


def permutation_p(diffs: torch.Tensor, n_perm: int = 20000, seed: int = 0) -> float:
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n_perm, len(diffs)), generator=generator).double() * 2 - 1
    null = (signs * diffs).mean(1).abs()
    return ((null >= diffs.mean().abs() - 1e-12).sum().item() + 1) / (n_perm + 1)


def bootstrap_ci(diffs: torch.Tensor, n_boot: int = 5000, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    means = diffs[torch.randint(0, len(diffs), (n_boot, len(diffs)), generator=generator)].mean(1)
    return means.quantile(0.025).item(), means.quantile(0.975).item()


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    correct, logp = load(args.paths)

    print(f"{'condition':<18}{'seeds':>6}{'n':>6}{'mean acc':>10}{'seed range':>18}")
    for condition, by_seed in correct.items():
        accs = [sum(d.values()) / len(d) for d in by_seed.values()]
        n = len(next(iter(by_seed.values())))
        print(f"{condition:<18}{len(accs):>6}{n:>6}{sum(accs) / len(accs):>10.3f}"
              f"{f'{min(accs):.3f}-{max(accs):.3f}':>18}")

    results = []
    for name, x, y in DECOMPOSITION:
        if x not in correct or y not in correct:
            continue
        ix, mx = per_question_mean(correct[x])
        iy, my = per_question_mean(correct[y])
        common = sorted(set(ix) & set(iy))
        dx = torch.tensor([mx[ix.index(i)].item() for i in common], dtype=torch.float64)
        dy = torch.tensor([my[iy.index(i)].item() for i in common], dtype=torch.float64)
        diffs = dx - dy
        results.append((name, diffs.mean().item(), bootstrap_ci(diffs), permutation_p(diffs), len(common)))

    if results:
        adjusted = holm([r[3] for r in results])
        print(f"\n{'part (X minus Y)':<28}{'acc diff':>10}{'95% CI':>20}{'p (Holm)':>10}{'n':>6}")
        for (name, mean, (lo, hi), _, n), p in zip(results, adjusted):
            print(f"{name:<28}{mean:>10.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}{p:>10.4f}{n:>6}")

    if logp:
        print(f"\n{'part (X minus Y)':<28}{'log p(gold) diff':>18}{'95% CI':>22}")
        for name, x, y in DECOMPOSITION:
            if x not in logp or y not in logp:
                continue
            common = sorted(set(logp[x]) & set(logp[y]))
            diffs = torch.tensor([sum(logp[x][i]) / len(logp[x][i]) - sum(logp[y][i]) / len(logp[y][i])
                                  for i in common], dtype=torch.float64)
            lo, hi = bootstrap_ci(diffs)
            print(f"{name:<28}{diffs.mean().item():>18.3f}{f'[{lo:.3f}, {hi:.3f}]':>22}")


if __name__ == "__main__":
    main()
