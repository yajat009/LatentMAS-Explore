"""Direction 3: give the Judger less to read.

The Judger normally attends to all three agents' prompts (three copies of the
question) plus every latent step. This compares trimmed versions of that memory:

    full              everything (repo default)
    last_agent_only   only the Refiner's prompt + latent steps (repo's sequential_info_only)
    last_latent_only  only the Refiner's latent steps (repo's latent_only)
    all_latents_only  the latent steps of all three agents, no prompts

Trimming keeps each cached key's original RoPE position, and the Judger's tokens
continue from the untrimmed length, so no position is reused (see SharedMemory.keep_only).

Usage:
    python -m experiments.exp3_trimmed_memory --model_name Qwen/Qwen3-4B --task mbppplus --max_samples 150 --think
"""

import argparse

import torch

from experiments.eval_utils import add_common_arguments, load_items, print_paired_report, save_jsonl, score_answer
from experiments.latent_mapping import build_mapping
from experiments.latent_pipeline import build_latent_memory, generate_text, judger_prompts, load_model

# condition name -> which memory segments the Judger keeps
KEEP_RULES = {
    "full": lambda segment: True,
    "last_agent_only": lambda segment: segment.agent == "refiner",
    "last_latent_only": lambda segment: segment.agent == "refiner" and segment.kind == "latent",
    "all_latents_only": lambda segment: segment.kind == "latent",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.model_name, args.device)
    mapping = build_mapping("identity", model)
    items = load_items(args.task, args.max_samples, args.toy)

    rows = []
    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start:batch_start + args.batch_size]
        questions = [item["question"] for item in batch]
        prompts = judger_prompts(tokenizer, args.model_name, args.task, questions, args.think)
        full_memory, _ = build_latent_memory(model, tokenizer, questions, model_name=args.model_name,
                                             task=args.task, latent_steps=args.latent_steps,
                                             mapping=mapping, think=args.think)

        for name, keep_rule in KEEP_RULES.items():
            memory = full_memory.keep_only(keep_rule)
            answers = generate_text(model, tokenizer, prompts, memory, max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature, top_p=args.top_p)
            for offset, (item, answer) in enumerate(zip(batch, answers)):
                memory_tokens = int(memory.attention_mask[offset].sum())
                rows.append({"index": batch_start + offset, "condition": name, "answer": answer,
                             "memory_tokens": memory_tokens, **score_answer(args.task, answer, item)})
        print(f"done {batch_start + len(batch)}/{len(items)}")

    print("\nMean memory size the Judger reads (tokens):")
    for name in KEEP_RULES:
        sizes = [r["memory_tokens"] for r in rows if r["condition"] == name]
        print(f"  {name:<18} {sum(sizes) / len(sizes):.0f}")

    correct_by_condition = {name: [r["correct"] for r in rows if r["condition"] == name] for name in KEEP_RULES}
    print_paired_report(correct_by_condition, baseline="full")
    save_jsonl(rows, f"{args.output_dir}/exp3_{args.task}_m{args.latent_steps}.jsonl")


if __name__ == "__main__":
    main()
