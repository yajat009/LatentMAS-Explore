"""Find every result already in the repo and summarize it in one table, so that old runs can be compared with the paper.

The script reads three kinds of files and never changes them:

  1. run.py logs (Slurm .out files, or saved stdout). The script reads the "  key: value"
     argument header, counts the "Result: ... | OK=True/False" lines, and reads the final
     JSON summary line. Each log becomes one row: method, model, task, layout, latent steps,
     realign flag, batch size, accuracy, number of questions, and a warning column.
  2. results/raw/test_runs/*.jsonl from the results/test_runs/ scripts. Each (file, condition)
     becomes one row with its accuracy and mean teacher-forced log p(gold).
  3. Other files that look like results (*.json, *.csv). The script lists these by name only.

Warnings flag the known pitfalls:

  * latent_mas with latent_steps 0, because the Judger then gets no cache.
  * generate_bs above 1 for latent_mas, because padding then leaks into the shared cache.
  * The old "muscle" method name.
  * Logs without a final summary, because the run crashed or is still running.

Usage (from the repo root):
    python results/test_runs/analysis/inventory_run_results.py                 # scans the current folder
    python results/test_runs/analysis/inventory_run_results.py ~/LatentMAS-Explore --csv inventory.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

ResultRow = Dict[str, Any]

# (model, task, layout) -> (Single, TextMAS, LatentMAS). The values come from architecture/assets/main_table1.png and main_table2.png.
PAPER_TABLE = {
    ("Qwen3-4B", "gsm8k", "sequential"): (82.4, 89.8, 88.2),
    ("Qwen3-4B", "mbppplus", "sequential"): (63.5, 69.8, 73.5),
    ("Qwen3-4B", "humanevalplus", "sequential"): (75.0, 79.7, 79.9),
    ("Qwen3-4B", "medqa", "sequential"): (47.7, 65.3, 66.3),
    ("Qwen3-4B", "arc_easy", "sequential"): (95.4, 96.4, 98.6),
    ("Qwen3-4B", "arc_challenge", "sequential"): (89.2, 90.0, 92.3),
    ("Qwen3-4B", "gsm8k", "hierarchical"): (82.4, 89.4, 88.4),
    ("Qwen3-4B", "mbppplus", "hierarchical"): (63.5, 69.3, 70.6),
    ("Qwen3-4B", "humanevalplus", "hierarchical"): (75.0, 76.2, 79.3),
    ("Qwen3-4B", "medqa", "hierarchical"): (47.7, 65.0, 67.3),
    ("Qwen3-4B", "arc_easy", "hierarchical"): (95.4, 97.1, 96.8),
    ("Qwen3-4B", "arc_challenge", "hierarchical"): (89.2, 92.5, 91.7),
    ("Qwen3-8B", "gsm8k", "sequential"): (81.1, 92.3, 93.8),
    ("Qwen3-8B", "mbppplus", "sequential"): (64.8, 69.5, 74.6),
    ("Qwen3-8B", "humanevalplus", "sequential"): (74.4, 80.5, 80.5),
    ("Qwen3-8B", "medqa", "sequential"): (53.0, 75.0, 75.3),
    ("Qwen3-8B", "arc_easy", "sequential"): (95.6, 99.1, 98.8),
    ("Qwen3-8B", "arc_challenge", "sequential"): (91.0, 94.6, 94.4),
    ("Qwen3-8B", "gsm8k", "hierarchical"): (81.1, 90.4, 89.5),
    ("Qwen3-8B", "mbppplus", "hierarchical"): (64.8, 71.9, 72.2),
    ("Qwen3-8B", "humanevalplus", "hierarchical"): (74.4, 76.8, 78.0),
    ("Qwen3-8B", "medqa", "hierarchical"): (53.0, 76.3, 77.0),
    ("Qwen3-8B", "arc_easy", "hierarchical"): (95.6, 98.2, 98.3),
    ("Qwen3-8B", "arc_challenge", "hierarchical"): (91.0, 93.3, 93.9),
    ("Qwen3-14B", "gsm8k", "sequential"): (83.7, 93.8, 95.2),
    ("Qwen3-14B", "mbppplus", "sequential"): (68.5, 72.8, 75.7),
    ("Qwen3-14B", "humanevalplus", "sequential"): (76.8, 81.1, 86.5),
}
METHOD_COLUMN = {"baseline": 0, "text_mas": 1, "latent_mas": 2}
# The pattern matches only the end of a result line, because gold code can span several lines.
RESULT_LINE = re.compile(r"\| OK=(True|False)\s*$", re.MULTILINE)
HEADER_LINE = re.compile(r"^  (\w+): (.*)$", re.MULTILINE)
LATENT_STEPS_MARKER = re.compile(r"^\[Latent Steps\]\s*\n\s*(\d+)", re.MULTILINE)
# Older logs have no argument header, so their settings come from the file name.
ARM_NAME = re.compile(r"(?P<method>baseline|text_mas|latent_mas)_(?P<model>Qwen3-\d+B)_(?P<task>\w+?)_"
                      r"(?P<layout>sequential|hierarchical)_ls(?P<steps>\d+)_bs(?P<batch>\d+)_think(?P<think>[01])"
                      r"(?P<realign>_realign)?")


def parse_settings_from_name(path: Path) -> Dict[str, Optional[str]]:
    """Return the run settings encoded in a file name, or an empty dict if it has none."""
    match = ARM_NAME.search(path.name)
    return match.groupdict() if match else {}


def look_up_paper_value(model: str, task: str, layout: str, method: str) -> Optional[float]:
    """Return the paper's accuracy for a run, or None if the paper has no such entry."""
    # The single-agent baseline has no layout, so its value is stored under "sequential".
    key = (model.split("/")[-1], task, layout if method != "baseline" else "sequential")
    row = PAPER_TABLE.get(key)
    return None if row is None or method not in METHOD_COLUMN else row[METHOD_COLUMN[method]]


def read_run_log(path: Path) -> Optional[ResultRow]:
    """Return one summary row for a run.py log, or None if the file is not one."""
    text = path.read_text(errors="replace")
    if "Result: " not in text and '"accuracy"' not in text:
        return None
    header = dict(HEADER_LINE.findall(text.split("Question 0")[0]))
    named = parse_settings_from_name(path)
    if named:  # An argument header, when present, wins over the file name.
        header.setdefault("method", named["method"])
        header.setdefault("model_name", "Qwen/" + named["model"])
        header.setdefault("task", named["task"])
        header.setdefault("prompt", named["layout"])
        header.setdefault("latent_steps", named["steps"])
        header.setdefault("generate_bs", named["batch"])
        header.setdefault("think", "True" if named["think"] == "1" else "False")
        header.setdefault("latent_space_realign", str(bool(named["realign"])))
    marker = LATENT_STEPS_MARKER.search(text)
    if marker:
        header.setdefault("latent_steps", marker.group(1))
    outcomes = RESULT_LINE.findall(text)
    summary = None
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{") and '"accuracy"' in line:
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                pass
            break

    method = header.get("method", summary.get("method") if summary else "?")
    model = header.get("model_name", summary.get("model") if summary else "?")
    task = header.get("task", "?")
    layout = header.get("prompt", "sequential")
    steps = header.get("latent_steps", "0 (default)" if method == "latent_mas" else "-")
    batch = header.get("generate_bs", "20 (default)")
    realign = header.get("latent_space_realign", "False")
    n_done = summary["max_samples"] if summary and summary.get("max_samples", -1) > 0 else len(outcomes)
    accuracy = summary["accuracy"] * 100 if summary else (100 * outcomes.count("True") / n_done if n_done else None)

    warnings = []
    if summary is None:
        warnings.append("no final summary: crashed, cut off, or still running")
    if method in ("latent_mas", "muscle") and str(steps).split()[0] == "0":
        warnings.append("latent_steps=0: Judger got no cache")
    if method in ("latent_mas", "muscle") and str(batch).split()[0] not in ("1",):
        warnings.append("generate_bs>1: padding leak")
    if method == "muscle" or (summary and summary.get("method") == "muscle"):
        warnings.append("old code version (method 'muscle')")

    paper = look_up_paper_value(model, task, layout, "latent_mas" if method == "muscle" else method)
    return {"file": str(path), "kind": "run.py", "method": method, "model": model, "task": task,
            "layout": layout, "latent_steps": steps, "realign": realign, "generate_bs": batch,
            "think": header.get("think", "?"), "n": n_done,
            "accuracy": None if accuracy is None else round(accuracy, 1),
            "paper": paper, "gap": None if (paper is None or accuracy is None) else round(accuracy - paper, 1),
            "warnings": "; ".join(warnings)}


def read_experiment_jsonl(path: Path) -> List[ResultRow]:
    """Return one summary row per condition in a JSONL file from results/test_runs/ or a run.py checkpoint."""
    by_condition = {}
    named = parse_settings_from_name(path)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        by_condition.setdefault(record.get("condition", "-"), []).append(record)
    rows = []
    for condition, records in by_condition.items():
        scored = [record for record in records if record.get("correct") is not None]
        logprobs = [record["gold_logprob"] for record in records
                    if isinstance(record.get("gold_logprob"), (int, float))]
        rows.append({"file": str(path), "kind": "checkpoint" if named else "experiments",
                     "method": named.get("method", condition),
                     "model": named.get("model", records[0].get("model", "?")),
                     "task": named.get("task", records[0].get("task", "?")),
                     "layout": named.get("layout", "sequential"),
                     "latent_steps": named.get("steps", "-"),
                     "realign": str(bool(named.get("realign"))) if named else "-",
                     "generate_bs": named.get("batch", "-"), "think": named.get("think", "-"), "n": len(records),
                     "accuracy": (round(100 * sum(bool(record["correct"]) for record in scored) / len(scored), 1)
                                  if scored else None),
                     "paper": None, "gap": None,
                     "warnings": f"mean log p(gold) {sum(logprobs) / len(logprobs):.3f}" if logprobs else ""})
    return rows


def main() -> None:
    """Scan the folder, print the table and the unparsed files, and optionally write a CSV file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--csv", help="Also write the table to this CSV file.")
    args = parser.parse_args()
    root = Path(args.root).expanduser()

    rows, other = [], []
    skip = {".git", "__pycache__", "data", ".cache", "hf_cache"}
    for path in sorted(root.rglob("*")):
        # The check reads only the part below root, because the home folder itself is under /data on HPC3.
        if not path.is_file() or any(part in skip for part in path.relative_to(root).parts):
            continue
        try:
            if path.suffix in {".out", ".log", ".txt", ".err"} and path.stat().st_size < 500_000_000:
                row = read_run_log(path)
                if row:
                    rows.append(row)
            elif path.suffix == ".jsonl":
                rows.extend(read_experiment_jsonl(path))
            elif path.suffix in {".json", ".csv"} and "result" in str(path).lower():
                other.append(str(path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
            other.append(f"{path} (unreadable: {error})")

    columns = ["kind", "method", "model", "task", "layout", "latent_steps", "realign", "generate_bs", "think",
               "n", "accuracy", "paper", "gap", "warnings", "file"]
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in rows:
        print("| " + " | ".join("" if row[column] is None else str(row[column]) for column in columns) + " |")
    print(f"\n{len(rows)} result rows found under {root}.")
    if other:
        print("Other result-like files (not parsed):")
        for other_path in other:
            print("  " + other_path)
    if args.csv:
        with open(args.csv, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
