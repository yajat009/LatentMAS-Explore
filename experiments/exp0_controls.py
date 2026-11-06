"""Direction 0, part 2: is the latent channel actually carrying information?

Runs the same Judger prompt under four kinds of shared memory:

    judger_alone     no memory at all (what the repo does at latent_steps=0)
    latent_identity  Planner/Critic/Refiner memory, repo-default mapping
    latent_ridge     same, with the paper's W_a realignment
    swapped_memory   memory built for a DIFFERENT question (the causal-audit control)

If latent_* beats judger_alone but swapped_memory does too, the gain comes from
extra context/compute, not from question-specific communication.

Usage:
    python -m experiments.exp0_controls --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 300 --think
"""

import argparse

import torch

from experiments.eval_utils import add_common_arguments, load_items, print_paired_report, save_jsonl, score_answer
from experiments.latent_mapping import build_mapping
from experiments.latent_pipeline import build_latent_memory, generate_text, judger_prompts, load_model

CONDITIONS = ["judger_alone", "latent_identity", "latent_ridge", "swapped_memory"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.model_name, args.device)
    mappings = {"identity": build_mapping("identity", model), "ridge": build_mapping("ridge", model)}
    items = load_items(args.task, args.max_samples, args.toy)
    # Each question's "donor" is the next question in the list.
    donor_items = items[1:] + items[:1]

    def latent_memory(questions, mapping_name):
        memory, _ = build_latent_memory(model, tokenizer, questions, model_name=args.model_name, task=args.task,
                                        latent_steps=args.latent_steps, mapping=mappings[mapping_name],
                                        think=args.think)
        return memory

    rows = []
    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start:batch_start + args.batch_size]
        questions = [item["question"] for item in batch]
        donors = [item["question"] for item in donor_items[batch_start:batch_start + len(batch)]]
        memory_by_condition = {
            "judger_alone": None,
            "latent_identity": latent_memory(questions, "identity"),
            "latent_ridge": latent_memory(questions, "ridge"),
            "swapped_memory": latent_memory(donors, "identity"),
        }
        prompts = judger_prompts(tokenizer, args.model_name, args.task, questions, args.think)

        for condition, memory in memory_by_condition.items():
            answers = generate_text(model, tokenizer, prompts, memory, max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature, top_p=args.top_p)
            for offset, (item, answer) in enumerate(zip(batch, answers)):
                rows.append({"index": batch_start + offset, "condition": condition, "answer": answer,
                             **score_answer(args.task, answer, item)})
        print(f"done {batch_start + len(batch)}/{len(items)}")

    correct_by_condition = {c: [r["correct"] for r in rows if r["condition"] == c] for c in CONDITIONS}
    print_paired_report(correct_by_condition, baseline="judger_alone")
    save_jsonl(rows, f"{args.output_dir}/exp0_{args.task}_m{args.latent_steps}.jsonl")


if __name__ == "__main__":
    main()
