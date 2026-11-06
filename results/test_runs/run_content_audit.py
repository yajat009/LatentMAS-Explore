"""Audit the content of the latent channel, to test whether it carries question-specific information.

Every condition uses the same Judger prompt on the same questions. Only the shared memory differs:

    A judger_alone     no memory
    B prompts_only     the three agent prompts in the cache, no latent steps (m = 0, cache kept)
    C pause            agent prompts and the same number of latent positions, all filled with one
                       content-free vector (pause tokens)
    D swapped          the whole agent memory built for a DIFFERENT question (same step counts)
    E latent_identity  real LatentMAS memory, with the repo's default mapping (W_a = I)
      latent_ridge     real memory, with the paper's W_a
      latent_random    real memory, with h rotated by a fixed random orthogonal matrix

The total gain E - A splits into (B-A) prompts + (C-B) positions and compute + (D-C) any message
+ (E-D) question-specific content. Only E-D is communication.

With --adaptive, latent_identity stops each row per agent when its hidden state converges. Every
other latent condition then uses the same per-row, per-agent step counts, so all controls match
in compute question by question.

The script gives two scores per question:

    correct        the generated answer is right (skipped with --no_generate)
    gold_logprob   teacher-forced log p(gold answer) after "...\\boxed{" (math and multiple-choice tasks)

Usage:
    python -m results.test_runs.run_content_audit --model_name Qwen/Qwen3-4B --task gsm8k --max_samples -1 \
        --latent_steps 20 --think --seed 1
Combine seeds afterwards with experiments.summarize_content_audit.
"""

import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.functional import cosine_similarity

from results.test_runs.audit_utils import (ConstantMapping, RandomRotationMapping, RecordingMapping,
                                     compute_paired_mean_difference, make_forced_steps_rule, score_gold_logprob)
from results.test_runs.eval_utils import (CODE_TASKS, add_adaptive_step_arguments, add_common_arguments,
                                    compute_mcnemar_exact_p, get_model_tag, iterate_batches, load_items, save_jsonl,
                                    score_answer, select_donor_questions)
from results.test_runs.run_adaptive_latent_steps import make_converged_stop_rule
from results.test_runs.latent_mapping import LatentMapping, build_mapping
from results.test_runs.latent_pipeline import (LATENT_AGENTS, AgentStats, SharedMemory, StopRule, build_judger_prompts,
                                         build_memory_per_agent, generate_with_args, load_model,
                                         repeat_for_each_agent)

ALL_CONDITIONS = ["judger_alone", "prompts_only", "pause", "swapped",
                  "latent_identity", "latent_ridge", "latent_random"]
# Each entry is (printed name, condition X, condition Y). The report shows X minus Y.
DECOMPOSITION = [("agent prompts", "prompts_only", "judger_alone"),
                 ("extra positions / compute", "pause", "prompts_only"),
                 ("any real message", "swapped", "pause"),
                 ("question-specific content", "latent_identity", "swapped"),
                 ("total", "latent_identity", "judger_alone")]


def main() -> None:
    """Build and score every memory condition on each batch, then print the report and save the rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--conditions", default=",".join(ALL_CONDITIONS),
                        help="Comma-separated subset of: " + ", ".join(ALL_CONDITIONS))
    parser.add_argument("--adaptive", action="store_true", help="Adaptive steps, controls matched per row.")
    add_adaptive_step_arguments(parser)
    parser.add_argument("--no_generate", action="store_true", help="Only teacher-forced scores (much faster).")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    conditions = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]
    use_logprob = args.task not in CODE_TASKS

    model, tokenizer = load_model(args.model_name, args.device)
    identity = build_mapping("identity", model)
    ridge = build_mapping("ridge", model)
    mappings = {"identity": identity, "ridge": ridge, "random": RandomRotationMapping(model, seed=0),
                "pause": ConstantMapping(model)}

    items = load_items(args.task, args.max_samples, args.toy)
    step_budget = args.max_latent_steps if args.adaptive else args.latent_steps

    def build_memory(memory_questions: List[str], mapping: LatentMapping,
                     stop_rules: Optional[Dict[str, StopRule]] = None,
                     latent_steps: int = step_budget) -> Tuple[SharedMemory, AgentStats]:
        """Return the memory and stats for these questions, with one mapping for all agents."""
        return build_memory_per_agent(model, tokenizer, memory_questions, model_name=args.model_name,
                                      task=args.task, latent_steps=latent_steps,
                                      mappings=repeat_for_each_agent(mapping), think=args.think,
                                      stop_rules=stop_rules)

    rows = []
    for batch_start, batch, questions in iterate_batches(items, args.batch_size):
        donors = select_donor_questions(items, batch_start, len(batch))
        golds = [str(item.get("gold", "")) for item in batch]

        # 1. The real run decides how many steps each row and agent uses.
        recorder = RecordingMapping(identity)
        first_rules = None
        if args.adaptive:
            first_rules = repeat_for_each_agent(make_converged_stop_rule(args.cosine_threshold,
                                                                         args.min_latent_steps))
        real_memory, real_stats = build_memory(questions, recorder, first_rules)
        steps_used = {agent: real_stats[agent]["steps_used"] for agent in LATENT_AGENTS}
        matched_rules = None
        if args.adaptive:
            matched_rules = {agent: make_forced_steps_rule(steps_used[agent]) for agent in LATENT_AGENTS}

        # This measures how much W_a changes the latent vector: cos(rescale(h W_a), rescale(h)), per row.
        if recorder.hidden_states:
            hidden_states = torch.stack(recorder.hidden_states).to(model.device)  # [steps_total, batch, hidden_dim]
            flat_hidden = hidden_states.flatten(0, 1)                             # [steps_total * batch, hidden_dim]
            cosines = cosine_similarity(ridge(flat_hidden).float(), identity(flat_hidden).float(), dim=-1)
            ridge_cos = cosines.view(hidden_states.shape[0], hidden_states.shape[1]).mean(dim=0).tolist()
        else:
            ridge_cos = [None] * len(batch)

        # 2. Build the memory of every condition, all with the same per-row step counts.
        memories = {}
        for condition in conditions:
            if condition == "judger_alone":
                memories[condition] = None
            elif condition == "prompts_only":
                memories[condition], _ = build_memory(questions, identity, latent_steps=0)
            elif condition == "pause":
                memories[condition], _ = build_memory(questions, mappings["pause"], matched_rules)
            elif condition == "swapped":
                memories[condition], _ = build_memory(donors, identity, matched_rules)
            elif condition == "latent_identity":
                memories[condition] = real_memory
            elif condition == "latent_ridge":
                memories[condition], _ = build_memory(questions, ridge, matched_rules)
            elif condition == "latent_random":
                memories[condition], _ = build_memory(questions, mappings["random"], matched_rules)
            else:
                raise ValueError(f"Unknown condition: {condition}")

        # 3. Score each condition.
        generation_prompts = build_judger_prompts(tokenizer, args.model_name, args.task, questions, args.think)
        teacher_forced_prompts = build_judger_prompts(tokenizer, args.model_name, args.task, questions, False)
        for condition, memory in memories.items():
            if args.no_generate:
                results = [{"answer": None, "correct": None} for _ in batch]
            else:
                answers = generate_with_args(model, tokenizer, generation_prompts, memory, args)
                results = [{"answer": answer, **score_answer(args.task, answer, item)}
                           for answer, item in zip(answers, batch)]
            if use_logprob:
                logprobs, top1_hits = score_gold_logprob(model, tokenizer, teacher_forced_prompts, golds, memory)
            else:
                logprobs, top1_hits = [None] * len(batch), [None] * len(batch)

            for offset, result in enumerate(results):
                rows.append({"index": batch_start + offset, "condition": condition, "seed": args.seed,
                             "model": args.model_name, "task": args.task,
                             "steps_per_agent": {agent: steps_used[agent][offset] for agent in LATENT_AGENTS},
                             "ridge_identity_cos": ridge_cos[offset],
                             "gold_logprob": logprobs[offset], "gold_top1": top1_hits[offset], **result})
        print(f"done {batch_start + len(batch)}/{len(items)}", flush=True)

    print_report(rows, conditions, args)
    tag = "adaptive" if args.adaptive else f"m{args.latent_steps}"
    save_jsonl(rows, f"{args.output_dir}/content_audit_{get_model_tag(args.model_name)}_{args.task}_{tag}_seed{args.seed}.jsonl")


def print_report(rows: List[Dict[str, Any]], conditions: List[str], args: argparse.Namespace) -> None:
    """Print the score table, the paired decomposition, the mean step counts and the W_a cosine."""
    rows_by_condition = {condition: sorted([row for row in rows if row["condition"] == condition],
                                           key=lambda row: row["index"])
                         for condition in conditions}
    print(f"\n{'condition':<18}{'accuracy':>10}{'mean log p(gold)':>18}{'gold top-1':>12}")
    for condition in conditions:
        condition_rows = rows_by_condition[condition]
        accuracy = float("nan")
        if not args.no_generate:
            accuracy = sum(bool(row["correct"]) for row in condition_rows) / len(condition_rows)
        logprobs = [row["gold_logprob"] for row in condition_rows if row["gold_logprob"] is not None]
        top1_hits = [row["gold_top1"] for row in condition_rows if row["gold_top1"] is not None]
        mean_logprob = sum(logprobs) / len(logprobs) if logprobs else float("nan")
        top1_rate = sum(top1_hits) / len(top1_hits) if top1_hits else float("nan")
        print(f"{condition:<18}{accuracy:>10.3f}{mean_logprob:>18.3f}{top1_rate:>12.3f}")

    print("\nDecomposition (X minus Y, paired on the same questions):")
    print(f"{'part':<28}{'acc diff':>10}{'wins':>6}{'losses':>8}{'p':>8}{'logp diff [95% CI]':>32}")
    for name, condition_x, condition_y in DECOMPOSITION:
        if condition_x not in rows_by_condition or condition_y not in rows_by_condition:
            continue
        rows_x, rows_y = rows_by_condition[condition_x], rows_by_condition[condition_y]
        line = f"{name:<28}"
        if not args.no_generate:
            correct_x = [bool(row["correct"]) for row in rows_x]
            correct_y = [bool(row["correct"]) for row in rows_y]
            wins = sum(hit_x and not hit_y for hit_x, hit_y in zip(correct_x, correct_y))
            losses = sum(hit_y and not hit_x for hit_x, hit_y in zip(correct_x, correct_y))
            line += (f"{(wins - losses) / len(correct_x):>10.3f}{wins:>6}{losses:>8}"
                     f"{compute_mcnemar_exact_p(wins, losses):>8.3f}")
        else:
            line += f"{'-':>10}{'-':>6}{'-':>8}{'-':>8}"
        logprobs_x = [row["gold_logprob"] for row in rows_x]
        logprobs_y = [row["gold_logprob"] for row in rows_y]
        if None not in logprobs_x and None not in logprobs_y:
            mean, (lower, upper) = compute_paired_mean_difference(logprobs_x, logprobs_y)
            line += f"{mean:>14.3f} [{lower:>6.3f}, {upper:>6.3f}]"
        print(line)

    real_rows = rows_by_condition.get("latent_identity") or next(iter(rows_by_condition.values()))
    means = {agent: sum(row["steps_per_agent"][agent] for row in real_rows) / len(real_rows)
             for agent in LATENT_AGENTS}
    print("\nMean latent steps per agent: " + "  ".join(f"{agent}={mean:.1f}" for agent, mean in means.items()))
    cosines = [row["ridge_identity_cos"] for row in real_rows if row["ridge_identity_cos"] is not None]
    if cosines:
        print(f"Mean cos(ridge vector, identity vector): {sum(cosines) / len(cosines):.4f}  "
              "(about 1.0 means W_a does nothing, as expected for tied embeddings)")


if __name__ == "__main__":
    main()
