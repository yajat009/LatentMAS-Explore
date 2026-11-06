#!/usr/bin/env python3
"""Export small CSV files from the raw run checkpoints, so that the repo can hold the evidence for every result.

The raw checkpoints in results/raw/ are about 260 MB, so git ignores them. This script keeps the
fields that the result tables need:

  results/data/runs.csv                 one row per run: its settings, n and accuracy
  results/data/question_outcomes.csv    one row per run and question: correct or not
  results/data/content_audit_scores.csv one row per audit question and condition: log p(gold)

Each question has a short hash, so a reader can confirm that two runs scored the same question.

Usage, from the repo root:  python results/test_runs/analysis/export_compact_results.py
"""

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

RAW = Path("results/raw")
OUT = Path("results/data")
FULL_TEST_SET_SIZE = {"mbppplus": 378, "gsm8k": 1319, "gpqa": 198}
SWEEP_NAME = re.compile(r"(?P<method>baseline|text_mas|latent_mas)_(?P<model>Qwen3-\d+B)_(?P<task>[a-z0-9]+)_"
                        r"sequential_ls(?P<steps>\d+)_bs(?P<batch>\d+)_think(?P<think>[01])(?P<realign>_realign)?$")
# Runs from the reduced reproduction and the audit. Their file names do not hold every setting.
# Each value is (model, task, method, latent steps, batch size, --think flag).
NAMED_RUNS = {
    "repro_4h/Qwen3-4B_mbppplus_latent_mas_m20": ("Qwen3-4B", "mbppplus", "latent_mas", 20, 1, 1),
    "repro_4h/Qwen3-4B_gsm8k_baseline": ("Qwen3-4B", "gsm8k", "baseline", 0, 8, 0),
    "repro_4h/Qwen3-4B_gsm8k_text_mas": ("Qwen3-4B", "gsm8k", "text_mas", 0, 8, 0),
    "repro_4h/Qwen3-4B_gsm8k_latent_mas_m20": ("Qwen3-4B", "gsm8k", "latent_mas", 20, 1, 1),
    "repro_4h/Qwen3-8B_mbppplus_baseline": ("Qwen3-8B", "mbppplus", "baseline", 0, 4, 0),
    "repro_4h/Qwen3-8B_mbppplus_latent_mas_m20": ("Qwen3-8B", "mbppplus", "latent_mas", 20, 1, 1),
    "audit_tests/4b_gsm8k_baseline_bs1": ("Qwen3-4B", "gsm8k", "baseline", 0, 1, 0),
    "audit_tests/4b_gsm8k_text_mas_bs1": ("Qwen3-4B", "gsm8k", "text_mas", 0, 1, 0),
    "audit_tests/4b_mbpp_baseline_bs1": ("Qwen3-4B", "mbppplus", "baseline", 0, 1, 0),
    "audit_tests/4b_mbpp_latent_m0_think": ("Qwen3-4B", "mbppplus", "latent_mas", 0, 1, 1),
    "audit_tests/4b_mbpp_latent_m20_nothink": ("Qwen3-4B", "mbppplus", "latent_mas", 20, 1, 0),
}


def read_jsonl(path: Path) -> Iterator[Dict]:
    """Yield each record of a JSONL file, and skip a line that a killed job cut short."""
    with open(path, errors="replace") as file:
        for line in file:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def describe_run(path: Path) -> Optional[Dict]:
    """Return the settings of one checkpoint, or None if the file is not part of the result set."""
    relative_name = str(path.relative_to(RAW)).removesuffix(".ckpt.jsonl")
    if relative_name in NAMED_RUNS:
        model, task, method, steps, batch, think = NAMED_RUNS[relative_name]
        realign = 0
    else:
        match = SWEEP_NAME.match(path.name.removesuffix(".ckpt.jsonl"))
        if not match:
            return None
        model, task, method = match["model"], match["task"], match["method"]
        steps, batch, think = int(match["steps"]), int(match["batch"]), int(match["think"])
        realign = int(bool(match["realign"]))
    run_id = f"{model}_{task}_{method}_steps{steps}_bs{batch}_think{think}" + ("_realign" if realign else "")
    return {"run_id": run_id, "model": model, "task": task, "method": method, "latent_steps": steps,
            "batch_size": batch, "think_flag": think, "realign": realign, "source": relative_name}


def is_numeric_match(prediction: object, gold: object) -> bool:
    """Return True if both values parse as the same number, so that 57.00 matches 57."""
    try:
        return abs(float(str(prediction).replace(",", "")) - float(str(gold).replace(",", ""))) < 1e-6
    except ValueError:
        return False


def export_runs() -> None:
    """Write runs.csv and question_outcomes.csv from every checkpoint under results/raw/."""
    # A later run can resume from an earlier checkpoint, so two files can hold one run.
    # The longer file holds all of the shorter one, so the script keeps only the longer file.
    longest: Dict[str, Tuple[Dict, List[Dict]]] = {}
    for path in sorted(RAW.rglob("*.ckpt.jsonl")):
        run = describe_run(path)
        if run is None:
            continue
        records = list(read_jsonl(path))
        if run["run_id"] not in longest or len(records) > len(longest[run["run_id"]][1]):
            longest[run["run_id"]] = (run, records)

    runs: List[Dict] = []
    outcomes: List[Dict] = []
    for run, records in longest.values():
        for index, record in enumerate(records):
            judger_output = record["agents"][-1].get("output") or ""
            outcomes.append({
                "run_id": run["run_id"], "question_index": index,
                "question_hash": hashlib.sha1(record["question"].encode()).hexdigest()[:8],
                "correct": int(bool(record["correct"])),
                "correct_numeric": (int(is_numeric_match(record.get("prediction"), record.get("gold")))
                                    if run["task"] == "gsm8k" else ""),
                "closed_think": int("</think>" in judger_output),
            })
        correct = sum(bool(record["correct"]) for record in records)
        size = FULL_TEST_SET_SIZE[run["task"]]
        runs.append({**run, "n": len(records), "correct": correct,
                     "accuracy_pct": round(100 * correct / len(records), 1),
                     "full_test_set": int(len(records) == size)})
    write_csv(OUT / "runs.csv", sorted(runs, key=lambda run: run["run_id"]))
    write_csv(OUT / "question_outcomes.csv", outcomes)
    print(f"{len(runs)} runs, {len(outcomes)} question outcomes")


def export_content_audit() -> None:
    """Write content_audit_scores.csv from the two content-audit pilot runs."""
    rows = []
    for path in sorted((RAW / "audit_pilot").glob("exp5_*.jsonl")):
        for record in read_jsonl(path):
            rows.append({"model": record["model"].split("/")[-1], "question_index": record["index"],
                         "condition": record["condition"], "gold_logprob": round(record["gold_logprob"], 4),
                         "gold_top1": int(bool(record["gold_top1"]))})
    write_csv(OUT / "content_audit_scores.csv", rows)
    print(f"{len(rows)} content-audit rows")


def write_csv(path: Path, rows: List[Dict]) -> None:
    """Write the rows to a CSV file, with the keys of the first row as the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    export_runs()
    export_content_audit()
