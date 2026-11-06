"""Let each question and each agent choose its number of latent steps, to test whether adaptive budgets beat a fixed m.

In place of one fixed m per task, a row stops when one of these conditions holds:

    converged   its hidden state stops moving: cosine(h_t, h_{t-1}) > --cosine_threshold
    confident   the model is sure of its next token: entropy(softmax(h W_out^T)) < --entropy_threshold

Both rules stop at --max_latent_steps at the latest.

The script compares these conditions on the same questions:

    fixed_m{latent_steps}, fixed_m{max_latent_steps}, stop_when_converged, stop_when_confident

Usage:
    python -m results.test_runs.run_adaptive_latent_steps --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 100 --think
"""

import argparse

import torch
from torch.nn.functional import cosine_similarity
from transformers import PreTrainedModel

from results.test_runs.eval_utils import (add_adaptive_step_arguments, add_common_arguments, collect_correct_by_condition,
                                    iterate_batches, load_items, print_paired_report, save_jsonl, score_answer)
from results.test_runs.latent_mapping import build_mapping
from results.test_runs.latent_pipeline import (StopRule, build_judger_prompts, build_latent_memory, generate_with_args,
                                         load_model)


def make_converged_stop_rule(cosine_threshold: float, min_steps: int) -> StopRule:
    """Return a stop rule that stops a row once its hidden state stops moving, after at least min_steps."""
    def should_stop(step: int, previous_hidden: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Return True for each row whose last two hidden states are almost parallel."""
        converged = cosine_similarity(hidden.float(), previous_hidden.float(), dim=-1) > cosine_threshold
        return converged & (step + 1 >= min_steps)
    return should_stop


def make_confident_stop_rule(model: PreTrainedModel, entropy_threshold: float, min_steps: int) -> StopRule:
    """Return a stop rule that stops a row once its next-token entropy is low, after at least min_steps."""
    def should_stop(step: int, previous_hidden: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Return True for each row whose next-token distribution has low entropy."""
        log_probs = torch.log_softmax(model.get_output_embeddings()(hidden).float(), dim=-1)  # [batch, vocab]
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        return (entropy < entropy_threshold) & (step + 1 >= min_steps)
    return should_stop


def main() -> None:
    """Run every step budget on each batch, then print the mean steps and the paired report, and save the rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    add_adaptive_step_arguments(parser)
    parser.add_argument("--entropy_threshold", type=float, default=1.0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.model_name, args.device)
    mapping = build_mapping("identity", model)

    # condition name -> (step budget, stop rule or None)
    conditions = {
        f"fixed_m{args.latent_steps}": (args.latent_steps, None),
        f"fixed_m{args.max_latent_steps}": (args.max_latent_steps, None),
        "stop_when_converged": (args.max_latent_steps,
                                make_converged_stop_rule(args.cosine_threshold, args.min_latent_steps)),
        "stop_when_confident": (args.max_latent_steps,
                                make_confident_stop_rule(model, args.entropy_threshold, args.min_latent_steps)),
    }

    items = load_items(args.task, args.max_samples, args.toy)
    rows = []
    for batch_start, batch, questions in iterate_batches(items, args.batch_size):
        prompts = build_judger_prompts(tokenizer, args.model_name, args.task, questions, args.think)

        for name, (step_budget, stop_rule) in conditions.items():
            memory, stats = build_latent_memory(model, tokenizer, questions, model_name=args.model_name,
                                                task=args.task, latent_steps=step_budget, mapping=mapping,
                                                think=args.think, should_stop=stop_rule)
            answers = generate_with_args(model, tokenizer, prompts, memory, args)
            for offset, (item, answer) in enumerate(zip(batch, answers)):
                steps_per_agent = {agent: agent_stats["steps_used"][offset] for agent, agent_stats in stats.items()}
                rows.append({"index": batch_start + offset, "condition": name, "answer": answer,
                             "steps_per_agent": steps_per_agent, **score_answer(args.task, answer, item)})
        print(f"done {batch_start + len(batch)}/{len(items)}")

    print("\nMean latent steps per agent:")
    for name in conditions:
        condition_rows = [row for row in rows if row["condition"] == name]
        means = {agent: sum(row["steps_per_agent"][agent] for row in condition_rows) / len(condition_rows)
                 for agent in condition_rows[0]["steps_per_agent"]}
        print(f"  {name:<22} " + "  ".join(f"{agent}={mean:.1f}" for agent, mean in means.items()))

    print_paired_report(collect_correct_by_condition(rows, conditions), baseline=f"fixed_m{args.latent_steps}")
    save_jsonl(rows, f"{args.output_dir}/adaptive_latent_steps_{args.task}.jsonl")


if __name__ == "__main__":
    main()
