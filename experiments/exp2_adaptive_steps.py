"""Direction 2: let each question (and each agent) decide how many latent steps it needs.

Instead of one fixed m per task, a row stops thinking when either
    converged   its hidden state stops moving: cosine(h_t, h_{t-1}) > --cosine_threshold
    confident   the model is sure of its next token: entropy(softmax(h W_out^T)) < --entropy_threshold
Both are capped at --max_latent_steps.

Conditions compared on the same questions:
    fixed_m{latent_steps}, fixed_m{max_latent_steps}, stop_when_converged, stop_when_confident

Usage:
    python -m experiments.exp2_adaptive_steps --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 100 --think
"""

import argparse

import torch
import torch.nn.functional as F

from experiments.eval_utils import add_common_arguments, load_items, print_paired_report, save_jsonl, score_answer
from experiments.latent_mapping import build_mapping
from experiments.latent_pipeline import build_latent_memory, generate_text, judger_prompts, load_model


def stop_when_converged(cosine_threshold: float, min_steps: int):
    def should_stop(step, previous_hidden, hidden):
        converged = F.cosine_similarity(hidden.float(), previous_hidden.float(), dim=-1) > cosine_threshold
        return converged & (step + 1 >= min_steps)
    return should_stop


def stop_when_confident(model, entropy_threshold: float, min_steps: int):
    def should_stop(step, previous_hidden, hidden):
        log_probs = torch.log_softmax(model.get_output_embeddings()(hidden).float(), dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        return (entropy < entropy_threshold) & (step + 1 >= min_steps)
    return should_stop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max_latent_steps", type=int, default=80)
    parser.add_argument("--min_latent_steps", type=int, default=5)
    parser.add_argument("--cosine_threshold", type=float, default=0.99)
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
                                stop_when_converged(args.cosine_threshold, args.min_latent_steps)),
        "stop_when_confident": (args.max_latent_steps,
                                stop_when_confident(model, args.entropy_threshold, args.min_latent_steps)),
    }

    items = load_items(args.task, args.max_samples, args.toy)
    rows = []
    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start:batch_start + args.batch_size]
        questions = [item["question"] for item in batch]
        prompts = judger_prompts(tokenizer, args.model_name, args.task, questions, args.think)

        for name, (step_budget, stop_rule) in conditions.items():
            memory, stats = build_latent_memory(model, tokenizer, questions, model_name=args.model_name,
                                                task=args.task, latent_steps=step_budget, mapping=mapping,
                                                think=args.think, should_stop=stop_rule)
            answers = generate_text(model, tokenizer, prompts, memory, max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature, top_p=args.top_p)
            for offset, (item, answer) in enumerate(zip(batch, answers)):
                steps_per_agent = {agent: s["steps_used"][offset] for agent, s in stats.items()}
                rows.append({"index": batch_start + offset, "condition": name, "answer": answer,
                             "steps_per_agent": steps_per_agent, **score_answer(args.task, answer, item)})
        print(f"done {batch_start + len(batch)}/{len(items)}")

    print("\nMean latent steps per agent:")
    for name in conditions:
        rows_for = [r for r in rows if r["condition"] == name]
        means = {agent: sum(r["steps_per_agent"][agent] for r in rows_for) / len(rows_for)
                 for agent in rows_for[0]["steps_per_agent"]}
        print(f"  {name:<22} " + "  ".join(f"{agent}={mean:.1f}" for agent, mean in means.items()))

    correct_by_condition = {name: [r["correct"] for r in rows if r["condition"] == name] for name in conditions}
    print_paired_report(correct_by_condition, baseline=f"fixed_m{args.latent_steps}")
    save_jsonl(rows, f"{args.output_dir}/exp2_{args.task}.jsonl")


if __name__ == "__main__":
    main()
