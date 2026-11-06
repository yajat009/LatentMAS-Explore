"""Compare the README-command GSM8K runs with the paper and with the authors' released logs.

Each run evaluated the same fixed random subset of the GSM8K test set (run.py --subset_seed 0).
The script prints accuracy, the standard error, the gap to the paper's Table 1, a paired
comparison of LatentMAS against the single-agent baseline, and a paired comparison against
the authors' own per-question results on the same questions.

Usage, from the repo root:
    python hpc3/compare_readme_repro.py [authors_ok.json]

authors_ok.json is optional. It maps a log name to {problem number: correct}, parsed from the
logs the authors link in their README ("additional experiment logs").
"""

import json
import math
import random
import sys
from pathlib import Path

OUT = Path("results/readme_repro")
PAPER = {("4b", "baseline"): 82.4, ("4b", "latent_mas"): 88.2, ("8b", "baseline"): 81.1, ("8b", "latent_mas"): 93.8}
GSM8K_TEST_SIZE = 1319
SUBSET_SIZE = 100
SUBSET_SEED = 0


def read_records(path: Path) -> list[dict]:
    """Return the per-question records of one checkpoint, or an empty list if it is missing."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break  # A killed job can leave a truncated last line.
    return records


def to_number(text) -> float | None:
    """Parse an answer string as a number, or return None if it is not one."""
    try:
        return float(str(text).replace(",", "").strip())
    except ValueError:
        return None


def mcnemar_exact_p(wins: int, losses: int) -> float:
    """Return the two-sided exact McNemar p-value for paired wins and losses."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    authors = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else {}
    # The same call run.py makes, so position i of a checkpoint is test question subset_ids[i].
    subset_ids = sorted(random.Random(SUBSET_SEED).sample(range(GSM8K_TEST_SIZE), SUBSET_SIZE))
    runs = {(m, method): read_records(OUT / f"{m}_gsm8k_{method}.ckpt.jsonl")
            for m in ("4b", "8b") for method in ("baseline", "latent_mas")}

    print("| Model | Method | n | Ours % | SE | Numeric rescore % | Paper % | Gap | Gap in SE | Median answer chars | Has </think> % |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (m, method), records in runs.items():
        n = len(records)
        if n == 0:
            print(f"| Qwen3-{m.upper()} | {method} | 0 | no data | | | {PAPER[(m, method)]} | | | | |")
            continue
        correct = sum(bool(r["correct"]) for r in records)
        numeric = sum(1 for r in records if to_number(r["prediction"]) is not None
                      and to_number(r["prediction"]) == to_number(r["gold"]))
        p = correct / n
        se = math.sqrt(p * (1 - p) / n)
        gap = 100 * p - PAPER[(m, method)]
        outputs = sorted(len(str(r["agents"][-1]["output"])) for r in records)
        closed = sum("</think>" in str(r["agents"][-1]["output"]) for r in records)
        print(f"| Qwen3-{m.upper()} | {method} | {n} | {100 * p:.1f} | {100 * se:.1f} | {100 * numeric / n:.1f} "
              f"| {PAPER[(m, method)]} | {gap:+.1f} | {gap / (100 * se):+.1f} | {outputs[n // 2]} | {100 * closed / n:.0f} |")

    print("\nPaired, LatentMAS against baseline on the same questions:")
    for m in ("4b", "8b"):
        base, latent = runs[(m, "baseline")], runs[(m, "latent_mas")]
        n = min(len(base), len(latent))
        if n == 0:
            print(f"  Qwen3-{m.upper()}: no shared questions")
            continue
        wins = sum(bool(l["correct"]) and not bool(b["correct"]) for b, l in zip(base[:n], latent[:n]))
        losses = sum(bool(b["correct"]) and not bool(l["correct"]) for b, l in zip(base[:n], latent[:n]))
        print(f"  Qwen3-{m.upper()}: n={n}, baseline {sum(bool(b['correct']) for b in base[:n])}, "
              f"LatentMAS {sum(bool(l['correct']) for l in latent[:n])}, LatentMAS wins {wins}, losses {losses}, "
              f"exact McNemar p = {mcnemar_exact_p(wins, losses):.3f}")

    if authors:
        print("\nPaired, our LatentMAS run against the authors' released LatentMAS log, same questions:")
        for m in ("4b", "8b"):
            theirs_all = authors.get(f"gsm8k_{m}_seq", {})
            latent = runs[(m, "latent_mas")]
            n = len(latent)
            if not theirs_all or n == 0:
                continue
            theirs = [bool(theirs_all[str(i + 1)]) for i in subset_ids[:n]]  # Log problem numbers start at 1.
            ours = [bool(r["correct"]) for r in latent]
            print(f"  Qwen3-{m.upper()}: n={n}, ours {sum(ours)}, authors {sum(theirs)}, "
                  f"agree on {sum(a == b for a, b in zip(ours, theirs))}, only ours right "
                  f"{sum(a and not b for a, b in zip(ours, theirs))}, only authors right "
                  f"{sum(b and not a for a, b in zip(ours, theirs))}")


if __name__ == "__main__":
    main()
