#!/usr/bin/env python
"""Compare method arms on the prefix of problems that all arms have finished, so that partial runs are usable.

results/test_runs/analysis/count_tokens_from_logs.py parses a finished stdout log. These runs are often preempted, so most logs
are partial. run.py writes the --checkpoint JSONL after every batch. That file carries the full
agent trace (`input_tokens` and `output` for each agent), so this script can recover every
number that the paper compares while a run is still going.

Arms are comparable only on the same problems, so the script reports everything on the common
prefix. run.py resumes by skipping a prefix, so record i is problem i in every arm.

Wall-clock time comes from `_elapsed_sec`. run.py stamps it on the last record of each batch and
carries it across preemption. It is meaningful only if every arm ran on the same GPU model.
All of these arms ran on L40S.

Usage:
    python results/test_runs/analysis/compare_methods_from_checkpoints.py                        # every mbppplus/4B arm
    python results/test_runs/analysis/compare_methods_from_checkpoints.py --glob 'results/raw/sweeps/*gsm8k*.ckpt.jsonl'
"""
import argparse
import ast
import glob
import io
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

ROLES = ["planner", "critic", "refiner", "judger", "singleagent"]


def read_checkpoint(path: str) -> Tuple[List[Dict], List[Tuple[int, float]]]:
    """Return the checkpoint records and the (records so far, elapsed seconds) mark of each batch."""
    recs, marks = [], []
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                break  # A preemption kill can truncate the final line.
            el = rec.pop("_elapsed_sec", None)
            recs.append(rec)
            if el is not None:
                marks.append((len(recs), el))
    return recs, marks


def get_agent_records(rec: Dict) -> List[Dict]:
    """Return the agent trace of a record, as a list of dicts."""
    a = rec.get("agents")
    if isinstance(a, str):
        # Older records stored the trace as its repr() string.
        try:
            a = ast.literal_eval(a)
        except (ValueError, SyntaxError):
            return []
    return a or []


def find_elapsed_seconds(marks: List[Tuple[int, float]], n: int) -> Optional[Tuple[int, float]]:
    """Return the mark of the last completed batch that ends at or before problem n, or None."""
    best = None
    for count, el in marks:
        if count <= n:
            best = (count, el)
    return best


def main() -> None:
    """Load all arms, compute their statistics on the common prefix, print them and write the JSON summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/raw/sweeps/*mbppplus*.ckpt.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="results/raw/sweeps/arm_comparison.json")
    ap.add_argument("--no-tokens", action="store_true",
                    help="skip loading the tokenizer (accuracy + timing only)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no checkpoints match {args.glob}")

    arms = {}
    for p in paths:
        name = os.path.basename(p).replace(".ckpt.jsonl", "")
        recs, marks = read_checkpoint(p)
        if recs:
            arms[name] = (recs, marks, p)

    n = min(len(r) for r, _, _ in arms.values())
    print(f"{len(arms)} arms; common prefix = {n} problems "
          f"(per-arm: {', '.join(str(len(r)) for r, _, _ in arms.values())})\n")

    tok = None
    if not args.no_tokens:
        os.environ.setdefault("HF_HOME", f"/pub/{os.environ.get('USER','')}/hf")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = []
    for name, (recs, marks, path) in sorted(arms.items()):
        pre = recs[:n]
        correct = sum(1 for r in pre if str(r.get("correct")) == "True")
        row = {"arm": name, "n": n, "correct": correct, "accuracy": correct / n}

        mark = find_elapsed_seconds(marks, n)
        if mark:
            row["timed_n"], row["elapsed_sec"] = mark
            row["sec_per_problem"] = mark[1] / mark[0]

        if tok is not None:
            gen = defaultdict(int)
            prompt = defaultdict(int)
            lat = defaultdict(int)
            for r in pre:
                for a in get_agent_records(r):
                    role = a.get("role", "?")
                    out = a.get("output") or ""
                    # The latent agents write no text, which is the paper's claim.
                    gen[role] += len(tok(out, add_special_tokens=False)["input_ids"]) if out else 0
                    lat[role] += a.get("latent_steps") or 0
                    # `input_tokens` and `input_ids` are token lists, not counts.
                    ids = a.get("input_ids") or a.get("input_tokens") or []
                    prompt[role] += len(ids)
            row["gen_tokens_per_problem"] = {k: round(v / n, 1) for k, v in gen.items()}
            row["gen_tokens_total_per_problem"] = round(sum(gen.values()) / n, 1)
            row["prompt_tokens_per_problem"] = {k: round(v / n, 1) for k, v in prompt.items()}
            row["prompt_tokens_total_per_problem"] = round(sum(prompt.values()) / n, 1)
            # Latent steps are KV (key/value) cache entries, not tokens. The paper's Token
            # column does not count them, so this script reports them separately.
            row["latent_steps_per_problem"] = round(sum(lat.values()) / n, 1)
        rows.append(row)

    w = max(len(r["arm"]) for r in rows)
    head = (f"{'arm'.ljust(w)}  {'acc':>7}  {'sec/prob':>9}  {'gen tok':>8}  "
            f"{'prompt tok':>10}  {'latent':>6}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['arm'].ljust(w)}  {r['accuracy']*100:6.2f}%  "
              f"{r.get('sec_per_problem', float('nan')):9.1f}  "
              f"{r.get('gen_tokens_total_per_problem', float('nan')):8.1f}  "
              f"{r.get('prompt_tokens_total_per_problem', float('nan')):10.1f}  "
              f"{r.get('latent_steps_per_problem', float('nan')):6.1f}")

    if tok is not None:
        print("\ngenerated tokens per problem, by agent role "
              "(latent agents should be ~0 -- that is the mechanism):")
        for r in rows:
            parts = " ".join(f"{k}={v}" for k, v in sorted(r["gen_tokens_per_problem"].items())
                             if v > 0)
            print(f"  {r['arm'].ljust(w)}  {parts or '(none)'}")

    with io.open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"common_prefix": n, "arms": rows}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
