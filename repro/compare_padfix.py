#!/usr/bin/env python
"""Measure what --pad_fix costs or buys, holding everything else fixed.

Both arms run the same problems, the same latent_steps, the same batch size and
the same seed; the only difference is the three-part padding fix. Anything that
moves is the padding artifact.

Run the two arms first (see repro/README.md), then:

    python repro/compare_padfix.py --off repro/results/padfix_off.ckpt.jsonl \\
                                   --on  repro/results/padfix_on.ckpt.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_arms import load, agents_of  # noqa: E402


def summarise(path, label):
    recs, marks = load(path)
    return {"label": label, "path": path, "recs": recs, "marks": marks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", default="repro/results/padfix_off.ckpt.jsonl")
    ap.add_argument("--on", default="repro/results/padfix_on.ckpt.jsonl")
    ap.add_argument("--out", default="repro/results/padfix_comparison.json")
    args = ap.parse_args()

    arms = []
    for path, label in ((args.off, "pad_fix=off (reproduction path)"),
                        (args.on, "pad_fix=on (left pad + real masks + fixed slice)")):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        arms.append(summarise(path, label))

    n = min(len(a["recs"]) for a in arms)
    print(f"common prefix = {n} problems\n")

    rows = []
    for a in arms:
        pre = a["recs"][:n]
        correct = sum(1 for r in pre if str(r.get("correct")) == "True")
        chars = [len(r.get("raw_prediction") or "") for r in pre]
        empty = sum(1 for r in pre if not (r.get("raw_prediction") or "").strip())
        row = {
            "label": a["label"],
            "n": n,
            "correct": correct,
            "accuracy": round(correct / n, 4),
            "judger_chars_mean": round(sum(chars) / max(len(chars), 1), 1),
            "empty_outputs": empty,
        }
        if a["marks"]:
            last = [m for m in a["marks"] if m[0] <= n]
            if last:
                row["sec_per_problem"] = round(last[-1][1] / last[-1][0], 2)
        rows.append(row)
        print(f"{row['label']}")
        print(f"  accuracy        {row['accuracy']:.4f}  ({correct}/{n})")
        print(f"  judger chars    {row['judger_chars_mean']:.0f}")
        print(f"  empty outputs   {row['empty_outputs']}")
        if "sec_per_problem" in row:
            print(f"  sec/problem     {row['sec_per_problem']:.1f}")
        print()

    delta = rows[1]["accuracy"] - rows[0]["accuracy"]
    print(f"delta (on - off): {delta:+.4f} accuracy over {n} problems")
    # An empty-output count that only the fixed arm shows means the slice is wrong,
    # not that the fix helped -- worth calling out rather than reporting as a win.
    if rows[1]["empty_outputs"] > rows[0]["empty_outputs"]:
        print("WARNING: the fixed arm produced MORE empty outputs. Check the "
              "generation slice before believing the accuracy delta.")

    out = {"common_prefix": n, "rows": rows, "accuracy_delta_on_minus_off": round(delta, 4)}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
