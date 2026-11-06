"""Compare the Judger under four kinds of memory, to test whether the latent channel carries information.

All conditions use the same Judger prompt:

    judger_alone     no memory at all (what the repo does at latent_steps=0)
    latent_identity  Planner, Critic and Refiner memory, with the repo's default mapping
    latent_ridge     the same memory, with the paper's W_a realignment
    swapped_memory   memory built for a DIFFERENT question (the causal-audit control)

Suppose latent_* beats judger_alone and swapped_memory beats it too. The gain then comes from
extra context or compute, and not from question-specific communication.

Usage:
    python -m experiments.exp0_controls --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 300 --think
"""

import argparse
from typing import List

import torch

from experiments.eval_utils import (add_common_arguments, collect_correct_by_condition, iterate_batches, load_items,
                                    print_paired_report, save_jsonl, score_answer, select_donor_questions)
from experiments.latent_mapping import build_mapping
from experiments.latent_pipeline import (SharedMemory, build_judger_prompts, build_latent_memory, generate_with_args,
                                         load_model)

CONDITIONS = ["judger_alone", "latent_identity", "latent_ridge", "swapped_memory"]


def main() -> None:
    """Run all conditions on each batch, then print the paired report and save the rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.model_name, args.device)
    mappings = {"identity": build_mapping("identity", model), "ridge": build_mapping("ridge", model)}
    items = load_items(args.task, args.max_samples, args.toy)

    def build_memory(memory_questions: List[str], mapping_name: str) -> SharedMemory:
        """Return the latent memory for these questions, built with the named mapping."""
        memory, _ = build_latent_memory(model, tokenizer, memory_questions, model_name=args.model_name,
                                        task=args.task, latent_steps=args.latent_steps,
                                        mapping=mappings[mapping_name], think=args.think)
        return memory

    rows = []
    for batch_start, batch, questions in iterate_batches(items, args.batch_size):
        donors = select_donor_questions(items, batch_start, len(batch))
        memory_by_condition = {
            "judger_alone": None,
            "latent_identity": build_memory(questions, "identity"),
            "latent_ridge": build_memory(questions, "ridge"),
            "swapped_memory": build_memory(donors, "identity"),
        }
        prompts = build_judger_prompts(tokenizer, args.model_name, args.task, questions, args.think)

        for condition, memory in memory_by_condition.items():
            answers = generate_with_args(model, tokenizer, prompts, memory, args)
            for offset, (item, answer) in enumerate(zip(batch, answers)):
                rows.append({"index": batch_start + offset, "condition": condition, "answer": answer,
                             **score_answer(args.task, answer, item)})
        print(f"done {batch_start + len(batch)}/{len(items)}")

    print_paired_report(collect_correct_by_condition(rows, CONDITIONS), baseline="judger_alone")
    save_jsonl(rows, f"{args.output_dir}/exp0_{args.task}_m{args.latent_steps}.jsonl")


if __name__ == "__main__":
    main()
