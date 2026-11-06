"""Content audit: does the latent channel carry question-specific information?

Every condition uses the same Judger prompt on the same questions. Only the shared memory differs:

    A judger_alone     no memory
    B prompts_only     the three agent prompts in the cache, no latent steps (m = 0, cache kept)
    C pause            agent prompts + the same number of latent positions, all filled with one
                       content-free vector (pause tokens)
    D swapped          the whole agent memory built for a DIFFERENT question (same step counts)
    E latent_identity  real LatentMAS memory, repo default mapping (W_a = I)
      latent_ridge     real memory, the paper's W_a
      latent_random    real memory, h rotated by a fixed random orthogonal matrix

The total gain E - A splits into  (B-A) prompts + (C-B) positions/compute + (D-C) any message
+ (E-D) question-specific content. Only E-D is communication.

With --adaptive, latent_identity stops each row per agent when its hidden state converges, and
every other latent condition is forced to use exactly the same per-row, per-agent step counts,
so all controls are compute-matched question by question.

Two scores per question:
    correct        the generated answer is right (skipped with --no_generate)
    gold_logprob   teacher-forced log p(gold answer) after "...\\boxed{" (math/multiple-choice tasks)

Usage:
    python -m experiments.exp5_content_audit --model_name Qwen/Qwen3-4B --task gsm8k --max_samples -1 \
        --latent_steps 20 --think --seed 1
Combine seeds afterwards with experiments.analyze_audit.
"""

import argparse

import torch
import torch.nn.functional as F

from experiments.latent_pipeline import build_memory_per_agent
from experiments.audit_utils import (ConstantMapping, RandomRotationMapping, RecordingMapping,
                                     make_forced_steps_rule, score_gold_logprob,
                                     compute_paired_mean_difference)
from experiments.eval_utils import (CODE_TASKS, add_common_arguments, load_items, compute_mcnemar_exact_p, save_jsonl,
                                    score_answer)
from experiments.exp2_adaptive_steps import stop_when_converged
from experiments.latent_mapping import build_mapping
from experiments.latent_pipeline import LATENT_AGENTS, generate_text, build_judger_prompts, load_model

ALL_CONDITIONS = ["judger_alone", "prompts_only", "pause", "swapped",
                  "latent_identity", "latent_ridge", "latent_random"]
DECOMPOSITION = [("agent prompts", "prompts_only", "judger_alone"),
                 ("extra positions / compute", "pause", "prompts_only"),
                 ("any real message", "swapped", "pause"),
                 ("question-specific content", "latent_identity", "swapped"),
                 ("total", "latent_identity", "judger_alone")]


def same_for_all(mapping):
    return {agent: mapping for agent in LATENT_AGENTS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--conditions", default=",".join(ALL_CONDITIONS),
                        help="Comma-separated subset of: " + ", ".join(ALL_CONDITIONS))
    parser.add_argument("--adaptive", action="store_true", help="Adaptive steps, controls matched per row.")
    parser.add_argument("--max_latent_steps", type=int, default=80)
    parser.add_argument("--min_latent_steps", type=int, default=5)
    parser.add_argument("--cosine_threshold", type=float, default=0.99)
    parser.add_argument("--no_generate", action="store_true", help="Only teacher-forced scores (much faster).")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    use_logprob = args.task not in CODE_TASKS

    model, tokenizer = load_model(args.model_name, args.device)
    identity = build_mapping("identity", model)
    ridge = build_mapping("ridge", model)
    mappings = {"identity": identity, "ridge": ridge, "random": RandomRotationMapping(model, seed=0),
                "pause": ConstantMapping(model)}

    items = load_items(args.task, args.max_samples, args.toy)
    donor_items = items[1:] + items[:1]  # each question's donor is the next one
    step_budget = args.max_latent_steps if args.adaptive else args.latent_steps

    rows = []
    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start:batch_start + args.batch_size]
        questions = [item["question"] for item in batch]
        donors = [item["question"] for item in donor_items[batch_start:batch_start + len(batch)]]
        golds = [str(item.get("gold", "")) for item in batch]

        def build(qs, mapping, stop_rules=None, steps=step_budget):
            return build_memory_per_agent(model, tokenizer, qs, model_name=args.model_name, task=args.task,
                                          latent_steps=steps, mappings=same_for_all(mapping), think=args.think,
                                          stop_rules=stop_rules)

        # 1. The real run decides how many steps each row/agent uses.
        recorder = RecordingMapping(identity)
        first_rules = None
        if args.adaptive:
            rule = stop_when_converged(args.cosine_threshold, args.min_latent_steps)
            first_rules = {agent: rule for agent in LATENT_AGENTS}
        real_memory, real_stats = build(questions, recorder, first_rules)
        steps = {agent: real_stats[agent]["steps_used"] for agent in LATENT_AGENTS}
        matched = {agent: make_forced_steps_rule(steps[agent]) for agent in LATENT_AGENTS} if args.adaptive else None

        # How much does W_a change the latent vector? cos(rescale(h W_a), rescale(h)), per row.
        if recorder.hidden_states:
            hs = torch.stack(recorder.hidden_states).to(model.device)  # [steps_total, batch, d]
            cos = F.cosine_similarity(ridge(hs.flatten(0, 1)).float(), identity(hs.flatten(0, 1)).float(), dim=-1)
            ridge_cos = cos.view(hs.shape[0], hs.shape[1]).mean(dim=0).tolist()
        else:
            ridge_cos = [None] * len(batch)

        # 2. Memories for every condition, all with the same per-row step counts.
        memories = {}
        for condition in conditions:
            if condition == "judger_alone":
                memories[condition] = None
            elif condition == "prompts_only":
                memories[condition], _ = build(questions, identity, steps=0)
            elif condition == "pause":
                memories[condition], _ = build(questions, mappings["pause"], matched)
            elif condition == "swapped":
                memories[condition], _ = build(donors, identity, matched)
            elif condition == "latent_identity":
                memories[condition] = real_memory
            elif condition == "latent_ridge":
                memories[condition], _ = build(questions, ridge, matched)
            elif condition == "latent_random":
                memories[condition], _ = build(questions, mappings["random"], matched)
            else:
                raise ValueError(f"Unknown condition: {condition}")

        # 3. Score each condition.
        gen_prompts = build_judger_prompts(tokenizer, args.model_name, args.task, questions, args.think)
        tf_prompts = build_judger_prompts(tokenizer, args.model_name, args.task, questions, False)
        for condition, memory in memories.items():
            if args.no_generate:
                results = [{"answer": None, "correct": None} for _ in batch]
            else:
                answers = generate_text(model, tokenizer, gen_prompts, memory, max_new_tokens=args.max_new_tokens,
                                        temperature=args.temperature, top_p=args.top_p)
                results = [{"answer": a, **score_answer(args.task, a, item)} for a, item in zip(answers, batch)]
            if use_logprob:
                logprobs, top1 = score_gold_logprob(model, tokenizer, tf_prompts, golds, memory)
            else:
                logprobs, top1 = [None] * len(batch), [None] * len(batch)

            for offset, result in enumerate(results):
                rows.append({"index": batch_start + offset, "condition": condition, "seed": args.seed,
                             "model": args.model_name, "task": args.task,
                             "steps_per_agent": {a: steps[a][offset] for a in LATENT_AGENTS},
                             "ridge_identity_cos": ridge_cos[offset],
                             "gold_logprob": logprobs[offset], "gold_top1": top1[offset], **result})
        print(f"done {batch_start + len(batch)}/{len(items)}", flush=True)

    report(rows, conditions, args)
    tag = "adaptive" if args.adaptive else f"m{args.latent_steps}"
    model_tag = args.model_name.rstrip("/").split("/")[-1]
    save_jsonl(rows, f"{args.output_dir}/exp5_{model_tag}_{args.task}_{tag}_seed{args.seed}.jsonl")


def report(rows, conditions, args) -> None:
    by = {c: sorted([r for r in rows if r["condition"] == c], key=lambda r: r["index"]) for c in conditions}
    print(f"\n{'condition':<18}{'accuracy':>10}{'mean log p(gold)':>18}{'gold top-1':>12}")
    for c in conditions:
        rs = by[c]
        acc = (sum(bool(r["correct"]) for r in rs) / len(rs)) if not args.no_generate else float("nan")
        lps = [r["gold_logprob"] for r in rs if r["gold_logprob"] is not None]
        t1 = [r["gold_top1"] for r in rs if r["gold_top1"] is not None]
        lp = sum(lps) / len(lps) if lps else float("nan")
        top = sum(t1) / len(t1) if t1 else float("nan")
        print(f"{c:<18}{acc:>10.3f}{lp:>18.3f}{top:>12.3f}")

    print("\nDecomposition (X minus Y, paired on the same questions):")
    print(f"{'part':<28}{'acc diff':>10}{'wins':>6}{'losses':>8}{'p':>8}{'logp diff [95% CI]':>32}")
    for name, x, y in DECOMPOSITION:
        if x not in by or y not in by:
            continue
        line = f"{name:<28}"
        if not args.no_generate:
            cx = [bool(r["correct"]) for r in by[x]]
            cy = [bool(r["correct"]) for r in by[y]]
            wins = sum(a and not b for a, b in zip(cx, cy))
            losses = sum(b and not a for a, b in zip(cx, cy))
            line += f"{(wins - losses) / len(cx):>10.3f}{wins:>6}{losses:>8}{compute_mcnemar_exact_p(wins, losses):>8.3f}"
        else:
            line += f"{'-':>10}{'-':>6}{'-':>8}{'-':>8}"
        lx = [r["gold_logprob"] for r in by[x]]
        ly = [r["gold_logprob"] for r in by[y]]
        if None not in lx and None not in ly:
            mean, (lo, hi) = compute_paired_mean_difference(lx, ly)
            line += f"{mean:>14.3f} [{lo:>6.3f}, {hi:>6.3f}]"
        print(line)

    real = by.get("latent_identity") or next(iter(by.values()))
    means = {a: sum(r["steps_per_agent"][a] for r in real) / len(real) for a in LATENT_AGENTS}
    print("\nMean latent steps per agent: " + "  ".join(f"{a}={m:.1f}" for a, m in means.items()))
    cos = [r["ridge_identity_cos"] for r in real if r["ridge_identity_cos"] is not None]
    if cos:
        print(f"Mean cos(ridge vector, identity vector): {sum(cos) / len(cos):.4f}  "
              "(about 1.0 means W_a does nothing, as expected for tied embeddings)")


if __name__ == "__main__":
    main()
