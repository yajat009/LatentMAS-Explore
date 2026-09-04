#!/usr/bin/env python
"""Summarise the latent_steps sweep: accuracy as a function of the silent-thought
budget, with and without --latent_space_realign.

Answers the question repro/README.md leaves open. Only ls=0 and ls=10 had ever
been run, and accuracy falls from 0.756 to 0.378 between them. Two stories fit
those two points and they imply opposite things:

  monotonic decline -> the latent channel is actively harmful at this model size
  a cliff           -> something breaks past a specific number of steps

The realign arm tests the leading mechanical explanation for either. Without
--latent_space_realign, models.py sets the output->input map to the identity, so
a hidden state from output space is fed back into an input-embedding slot with
only a norm rescale. If that is the problem, the realign arm recovers.

Reads the per-arm checkpoint JSONLs, so it works mid-sweep -- reporting every
arm on the common prefix of problems they have all finished.

    python repro/analyze_sweep.py
    python repro/analyze_sweep.py --min-n 20     # include partial arms
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_arms import load, agents_of, elapsed_at  # noqa: E402

PAT = re.compile(r"_ls(\d+)_bs(\d+)_think\d+(_realign)?$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="repro/results/latent_mas_*_mbppplus_sequential_ls*_bs1_think1*.ckpt.jsonl")
    ap.add_argument("--out", default="repro/results/ls_sweep.json")
    ap.add_argument("--min-n", type=int, default=1,
                    help="ignore arms with fewer than this many finished problems")
    args = ap.parse_args()

    arms = []
    for p in sorted(glob.glob(args.glob)):
        name = os.path.basename(p).replace(".ckpt.jsonl", "")
        m = PAT.search(name)
        if not m:
            continue
        recs, marks = load(p)
        if len(recs) < args.min_n:
            continue
        arms.append({
            "arm": name,
            "latent_steps": int(m.group(1)),
            "realign": bool(m.group(3)),
            "recs": recs,
            "marks": marks,
        })

    if not arms:
        raise SystemExit(f"no sweep checkpoints match {args.glob}")

    n = min(len(a["recs"]) for a in arms)
    print(f"{len(arms)} arms; common prefix = {n} problems")
    print(f"per-arm finished: {', '.join(str(len(a['recs'])) for a in arms)}\n")

    rows = []
    for a in arms:
        pre = a["recs"][:n]
        correct = sum(1 for r in pre if str(r.get("correct")) == "True")
        row = {
            "latent_steps": a["latent_steps"],
            "realign": a["realign"],
            "n": n,
            "n_finished": len(a["recs"]),
            "correct": correct,
            "accuracy": round(correct / n, 4),
        }
        mark = elapsed_at(a["marks"], n)
        if mark:
            row["sec_per_problem"] = round(mark[1] / mark[0], 2)
        # the judger is the only agent that emits text; its length is the cheapest
        # signal for "did the model stop generating early", which is part of why
        # the ls=10 arm looks cheap in the token column.
        chars = [len(r.get("raw_prediction") or "") for r in pre]
        row["judger_chars_mean"] = round(sum(chars) / max(len(chars), 1), 1)
        row["empty_predictions"] = sum(1 for r in pre if not (r.get("prediction") or ""))
        rows.append(row)

    rows.sort(key=lambda r: (r["realign"], r["latent_steps"]))

    hdr = f"{'ls':>4} {'realign':>8} {'acc':>7} {'correct':>9} {'s/prob':>8} {'judger chars':>13} {'empty':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['latent_steps']:>4} {str(r['realign']):>8} {r['accuracy']:>7.3f} "
              f"{str(r['correct']) + '/' + str(r['n']):>9} "
              f"{r.get('sec_per_problem', float('nan')):>8.1f} "
              f"{r['judger_chars_mean']:>13.0f} {r['empty_predictions']:>6}")

    # Shape verdict, stated only when the sweep is dense enough to support one.
    base = [r for r in rows if not r["realign"]]
    verdict = None
    if len(base) >= 4:
        accs = [r["accuracy"] for r in sorted(base, key=lambda r: r["latent_steps"])]
        drops = [accs[i] - accs[i + 1] for i in range(len(accs) - 1)]
        worst = max(drops) if drops else 0.0
        total = accs[0] - accs[-1]
        if total <= 0.05:
            verdict = "flat -- latent_steps barely matters over this range"
        elif worst >= 0.6 * total and total > 0:
            step = sorted(base, key=lambda r: r["latent_steps"])[drops.index(worst) + 1]["latent_steps"]
            verdict = (f"CLIFF -- {worst:.3f} of the {total:.3f} total decline lands in a single "
                       f"step, at ls={step}. Suggests a bug at that length, not a gradual cost.")
        else:
            verdict = (f"MONOTONIC -- the {total:.3f} decline is spread across steps, no single "
                       f"cliff. Suggests the latent channel is genuinely harmful at this scale.")
        print(f"\nshape: {verdict}")

    out = {"common_prefix": n, "rows": rows, "shape_verdict": verdict}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
