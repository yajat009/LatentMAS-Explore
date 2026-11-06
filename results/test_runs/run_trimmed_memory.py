"""Compare trimmed versions of the shared memory, to test how much of it the Judger needs.

The Judger normally attends to the prompts of all three agents (three copies of the question)
and to every latent step. The conditions are:

    full              everything (the repo's default)
    last_agent_only   only the Refiner's prompt and latent steps (the repo's sequential_info_only)
    last_latent_only  only the Refiner's latent steps (the repo's latent_only)
    all_latents_only  the latent steps of all three agents, without prompts

Trimming keeps the original RoPE (rotary position embedding) position of each cached key.
The Judger's tokens continue from the untrimmed length, so no position is reused.
See SharedMemory.keep_only.

Usage:
    python -m results.test_runs.run_trimmed_memory --model_name Qwen/Qwen3-4B --task mbppplus --max_samples 150 --think
"""

import argparse

import torch

from results.test_runs.eval_utils import (add_common_arguments, collect_correct_by_condition, iterate_batches, load_items,
                                    print_paired_report, save_jsonl, score_answer)
from results.test_runs.latent_mapping import build_mapping
from results.test_runs.latent_pipeline import build_judger_prompts, build_latent_memory, generate_with_args, load_model

# condition name -> which memory segments the Judger keeps
KEEP_RULES = {
    "full": lambda segment: True,
    "last_agent_only": lambda segment: segment.agent == "refiner",
    "last_latent_only": lambda segment: segment.agent == "refiner" and segment.kind == "latent",
    "all_latents_only": lambda segment: segment.kind == "latent",
}


def main() -> None:
    """Run every keep rule on each batch, then print the memory sizes and the paired report, and save the rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.model_name, args.device)
    mapping = build_mapping("identity", model)
    items = load_items(args.task, args.max_samples, args.toy)

    rows = []
    for batch_start, batch, questions in iterate_batches(items, args.batch_size):
        prompts = build_judger_prompts(tokenizer, args.model_name, args.task, questions, args.think)
        full_memory, _ = build_latent_memory(model, tokenizer, questions, model_name=args.model_name,
                                             task=args.task, latent_steps=args.latent_steps,
                                             mapping=mapping, think=args.think)

        for name, keep_rule in KEEP_RULES.items():
            memory = full_memory.keep_only(keep_rule)
            answers = generate_with_args(model, tokenizer, prompts, memory, args)
            for offset, (item, answer) in enumerate(zip(batch, answers)):
                memory_tokens = int(memory.attention_mask[offset].sum())
                rows.append({"index": batch_start + offset, "condition": name, "answer": answer,
                             "memory_tokens": memory_tokens, **score_answer(args.task, answer, item)})
        print(f"done {batch_start + len(batch)}/{len(items)}")

    print("\nMean memory size the Judger reads (tokens):")
    for name in KEEP_RULES:
        sizes = [row["memory_tokens"] for row in rows if row["condition"] == name]
        print(f"  {name:<18} {sum(sizes) / len(sizes):.0f}")

    print_paired_report(collect_correct_by_condition(rows, KEEP_RULES), baseline="full")
    save_jsonl(rows, f"{args.output_dir}/trimmed_memory_{args.task}_m{args.latent_steps}.jsonl")


if __name__ == "__main__":
    main()
