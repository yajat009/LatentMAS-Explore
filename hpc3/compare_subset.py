"""Compare methods on the questions they all finished, because a subset is not the full test set.

The script reads run.py checkpoint files. For each group of runs it cuts every run to the
shared first n questions, then prints accuracy, the standard error, the gap to the paper,
paired wins and losses against the baseline, and the share of Judger outputs that never
closed their <think> block.

Usage, from the repo root:  python hpc3/compare_subset.py
"""

import json
import math
from pathlib import Path

NEW = Path("results/repro_4h")
OLD = Path("yajat/results")
# Each group: (title, paper values by method, runs as (label, paper key, checkpoint path)).
GROUPS = [
    ("Qwen3-4B MBPP+", {"baseline": 63.5, "text_mas": 69.8, "latent_mas": 73.5}, [
        ("baseline (old run)", "baseline", OLD / "baseline_Qwen3-4B_mbppplus_sequential_ls0_bs15_think1.ckpt.jsonl"),
        ("text_mas (old run)", "text_mas", OLD / "text_mas_Qwen3-4B_mbppplus_sequential_ls0_bs8_think1.ckpt.jsonl"),
        ("latent_mas m20", "latent_mas", NEW / "Qwen3-4B_mbppplus_latent_mas_m20.ckpt.jsonl"),
    ]),
    ("Qwen3-4B GSM8K", {"baseline": 82.4, "text_mas": 89.8, "latent_mas": 88.2}, [
        ("baseline", "baseline", NEW / "Qwen3-4B_gsm8k_baseline.ckpt.jsonl"),
        ("text_mas", "text_mas", NEW / "Qwen3-4B_gsm8k_text_mas.ckpt.jsonl"),
        ("latent_mas m20", "latent_mas", NEW / "Qwen3-4B_gsm8k_latent_mas_m20.ckpt.jsonl"),
    ]),
    ("Qwen3-8B MBPP+", {"baseline": 64.8, "text_mas": 69.5, "latent_mas": 74.6}, [
        ("baseline", "baseline", NEW / "Qwen3-8B_mbppplus_baseline.ckpt.jsonl"),
        ("latent_mas m20", "latent_mas", NEW / "Qwen3-8B_mbppplus_latent_mas_m20.ckpt.jsonl"),
        ("latent_mas m20 realign", "latent_mas", NEW / "Qwen3-8B_mbppplus_latent_mas_m20_realign.ckpt.jsonl"),
    ]),
]


def read_records(path: Path) -> list[dict]:
    """Return the per-question records of one checkpoint, or an empty list if it is missing."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def judger_output(record: dict) -> str:
    """Return the text the final agent wrote for one question."""
    agents = record.get("agents") or []
    return str(agents[-1].get("output", "")) if agents else str(record.get("raw_prediction", ""))


def print_group(title: str, paper: dict, runs: list) -> None:
    """Print one comparison table, cut to the questions every available run finished."""
    loaded = [(label, key, read_records(path)) for label, key, path in runs]
    loaded = [run for run in loaded if run[2]]
    if not loaded:
        print(f"\n{title}: no data")
        return
    shared = min(len(records) for _, _, records in loaded)
    print(f"\n{title}: first {shared} questions ({', '.join(f'{l} has {len(r)}' for l, _, r in loaded)})")
    print("| Method | n | Ours % | SE | Paper % (full set) | Gap | Gap in SE | Wins vs first row | Losses | No </think> % |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    reference = [bool(r["correct"]) for r in loaded[0][2][:shared]]
    for label, key, records in loaded:
        outcomes = [bool(r["correct"]) for r in records[:shared]]
        accuracy = sum(outcomes) / shared
        standard_error = math.sqrt(accuracy * (1 - accuracy) / shared)
        gap = 100 * accuracy - paper[key]
        gap_in_se = f"{gap / (100 * standard_error):+.1f}" if standard_error else "-"
        wins = sum(o and not b for o, b in zip(outcomes, reference))
        losses = sum(b and not o for o, b in zip(outcomes, reference))
        unclosed = sum("</think>" not in judger_output(r) for r in records[:shared]) / shared
        print(f"| {label} | {shared} | {100 * accuracy:.1f} | {100 * standard_error:.1f} | {paper[key]} "
              f"| {gap:+.1f} | {gap_in_se} | {wins} | {losses} | {100 * unclosed:.1f} |")


if __name__ == "__main__":
    for group in GROUPS:
        print_group(*group)
