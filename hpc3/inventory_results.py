"""Find every result already in the repo and summarize it in one table.

It reads three kinds of files and never changes them:

  1. run.py logs (Slurm .out files, or saved stdout). The script reads the "  key: value"
     argument header, counts the "Result: ... | OK=True/False" lines, and reads the final
     JSON summary line. Each log becomes one row: method, model, task, layout, latent steps,
     realign flag, batch size, accuracy, number of questions, and a warning column.
  2. experiments/results/*.jsonl from the experiments/ scripts. Each (file, condition)
     becomes one row with its accuracy and mean teacher-forced log p(gold).
  3. Anything else that looks like a result (*.json, *.csv) is listed by name only.

Warnings flag the known pitfalls: latent_mas with latent_steps 0 (the Judger gets no
cache), generate_bs above 1 for latent_mas (padding leak), an old "muscle" method name,
and logs that never reached the final summary (crashed or still running).

Usage (from the repo root):
    python hpc3/inventory_results.py                 # scans the current folder
    python hpc3/inventory_results.py ~/LatentMAS-Explore --csv inventory.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path

PAPER_TABLE = {  # (model, task, layout) -> (Single, TextMAS, LatentMAS), from assets/main_table1/2.png
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
RESULT_LINE = re.compile(r"\| OK=(True|False)\s*$", re.MULTILINE)  # gold code can span lines
HEADER_LINE = re.compile(r"^  (\w+): (.*)$", re.MULTILINE)
LATENT_STEPS_MARKER = re.compile(r"^\[Latent Steps\]\s*\n\s*(\d+)", re.MULTILINE)
# Older logs have no argument header. Their settings are in the file name instead.
ARM_NAME = re.compile(r"(?P<method>baseline|text_mas|latent_mas)_(?P<model>Qwen3-\d+B)_(?P<task>\w+?)_"
                      r"(?P<layout>sequential|hierarchical)_ls(?P<steps>\d+)_bs(?P<batch>\d+)_think(?P<think>[01])"
                      r"(?P<realign>_realign)?")


def settings_from_name(path: Path) -> dict:
    """Return the run settings encoded in a file name, or an empty dict if it has none."""
    match = ARM_NAME.search(path.name)
    return match.groupdict() if match else {}


def paper_value(model: str, task: str, layout: str, method: str):
    key = (model.split("/")[-1], task, layout if method != "baseline" else "sequential")
    row = PAPER_TABLE.get(key)
    return None if row is None or method not in METHOD_COLUMN else row[METHOD_COLUMN[method]]


def read_run_log(path: Path):
    """Return one summary row for a run.py log, or None if the file is not one."""
    text = path.read_text(errors="replace")
    if "Result: " not in text and '"accuracy"' not in text:
        return None
    header = dict(HEADER_LINE.findall(text.split("Question 0")[0]))
    named = settings_from_name(path)
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

    paper = paper_value(model, task, layout, "latent_mas" if method == "muscle" else method)
    return {"file": str(path), "kind": "run.py", "method": method, "model": model, "task": task,
            "layout": layout, "latent_steps": steps, "realign": realign, "generate_bs": batch,
            "think": header.get("think", "?"), "n": n_done,
            "accuracy": None if accuracy is None else round(accuracy, 1),
            "paper": paper, "gap": None if (paper is None or accuracy is None) else round(accuracy - paper, 1),
            "warnings": "; ".join(warnings)}


def read_experiment_jsonl(path: Path):
    """One row per condition in an experiments/ results file."""
    by_condition = {}
    named = settings_from_name(path)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_condition.setdefault(r.get("condition", "-"), []).append(r)
    rows = []
    for condition, rs in by_condition.items():
        scored = [r for r in rs if r.get("correct") is not None]
        lps = [r["gold_logprob"] for r in rs if isinstance(r.get("gold_logprob"), (int, float))]
        rows.append({"file": str(path), "kind": "checkpoint" if named else "experiments",
                     "method": named.get("method", condition),
                     "model": named.get("model", rs[0].get("model", "?")),
                     "task": named.get("task", rs[0].get("task", "?")),
                     "layout": named.get("layout", "sequential"),
                     "latent_steps": named.get("steps", "-"),
                     "realign": str(bool(named.get("realign"))) if named else "-",
                     "generate_bs": named.get("batch", "-"), "think": named.get("think", "-"), "n": len(rs),
                     "accuracy": round(100 * sum(bool(r["correct"]) for r in scored) / len(scored), 1) if scored else None,
                     "paper": None, "gap": None,
                     "warnings": f"mean log p(gold) {sum(lps) / len(lps):.3f}" if lps else ""})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--csv", help="Also write the table to this CSV file.")
    args = parser.parse_args()
    root = Path(args.root).expanduser()

    rows, other = [], []
    skip = {".git", "__pycache__", "data", ".cache", "hf_cache"}
    for path in sorted(root.rglob("*")):
        # Check only the part below root: the home folder itself is under /data on HPC3.
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
    for r in rows:
        print("| " + " | ".join("" if r[c] is None else str(r[c]) for c in columns) + " |")
    print(f"\n{len(rows)} result rows found under {root}.")
    if other:
        print("Other result-like files (not parsed):")
        for o in other:
            print("  " + o)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
