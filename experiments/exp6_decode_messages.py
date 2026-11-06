"""Decode the latent messages with a logit lens and activation patching, to test what they say and whether it matters.

This script generates no text, so it is cheap.

Logit lens
    At every latent step of every agent, the script applies the language-model head to the hidden
    state h_k. It records the top tokens and the rank of the gold answer's first token
    (rank 1 = the model's top choice). If latent thoughts carry the solution, the gold rank is
    small and falls over the steps.

Activation patching (causal)
    The script scores the Judger by teacher forcing: log p(gold answer) after "...\\boxed{".
    It then runs the pipeline again with ONE agent's latent vectors replaced, and measures the drop:
        swap_<agent>    that agent's latent vectors, replayed from a different question
        pause_<agent>   that agent's latent vectors, replaced by one content-free vector
        swap_all        all three agents' latent vectors, replayed from a different question
    Later agents run again on top of the patched memory, so this is the total effect of that
    agent's message. delta = log p(gold | real) - log p(gold | patched). If delta is about 0
    everywhere, the Judger ignores the latent content.

Usage:
    python -m experiments.exp6_decode_messages --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 300 \
        --latent_steps 20 --mapping identity
"""

import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from experiments.audit_utils import (ConstantMapping, RecordingMapping, ReplayMapping,
                                     compute_paired_mean_difference, score_gold_logprob)
from experiments.eval_utils import (CODE_TASKS, add_common_arguments, get_model_tag, iterate_batches, load_items,
                                    save_jsonl, select_donor_questions)
from experiments.latent_mapping import LatentMapping, build_mapping
from experiments.latent_pipeline import (LATENT_AGENTS, AgentStats, SharedMemory, build_judger_prompts,
                                         build_memory_per_agent, load_model, repeat_for_each_agent)


def get_gold_first_token_ids(tokenizer: PreTrainedTokenizerBase, gold: str) -> List[int]:
    """Return the token ids that can start the gold answer, with and without a leading space."""
    ids = set()
    for text in (gold, " " + gold):
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if encoded:
            ids.add(encoded[0])
    return sorted(ids)


@torch.no_grad()
def apply_logit_lens(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                     hidden_states: List[torch.Tensor], golds: List[str],
                     top_k: int) -> List[List[Dict[str, Any]]]:
    """Return, for each row, one dict per step with the gold rank and the top tokens.

    hidden_states is a list over steps of tensors of shape [batch, hidden_dim].
    """
    lm_head = model.get_output_embeddings()
    per_row = [[] for _ in golds]
    gold_ids = [get_gold_first_token_ids(tokenizer, gold) for gold in golds]
    for step, h in enumerate(hidden_states):  # h is the hidden state of this step, as in the paper.
        logits = lm_head(h.to(model.device, model.dtype)).float()  # [batch, vocab]
        top_ids = logits.topk(top_k, dim=-1).indices.tolist()
        order = logits.argsort(dim=-1, descending=True)            # [batch, vocab]
        ranks = torch.empty_like(order)                            # [batch, vocab]; ranks[row, token] = 0-based rank
        ranks.scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
        for row in range(len(golds)):
            gold_rank = min(ranks[row, i].item() for i in gold_ids[row]) + 1 if gold_ids[row] else None
            per_row[row].append({"step": step, "gold_rank": gold_rank,
                                 "top_tokens": [tokenizer.decode([token_id]) for token_id in top_ids[row]]})
    return per_row


def main() -> None:
    """Run the real, donor and patched pipelines on each batch, then print the report and save the rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--mapping", default="identity", choices=["identity", "ridge"])
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    if args.task in CODE_TASKS:
        raise SystemExit("Use a task with a short gold answer (gsm8k, arc_*, gpqa, medqa, aime*).")
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.model_name, args.device)
    mapping = build_mapping(args.mapping, model)
    pause = ConstantMapping(model)
    items = load_items(args.task, args.max_samples, args.toy)

    def build_memory(memory_questions: List[str],
                     agent_mappings: Dict[str, LatentMapping]) -> Tuple[SharedMemory, AgentStats]:
        """Return the memory and stats for these questions, with one mapping per agent."""
        return build_memory_per_agent(model, tokenizer, memory_questions, model_name=args.model_name,
                                      task=args.task, latent_steps=args.latent_steps,
                                      mappings=agent_mappings, think=args.think)

    rows = []
    for batch_start, batch, questions in iterate_batches(items, args.batch_size):
        donors = select_donor_questions(items, batch_start, len(batch))
        golds = [str(item.get("gold", "")) for item in batch]
        prompts = build_judger_prompts(tokenizer, args.model_name, args.task, questions, False)

        def score_memory(memory: Optional[SharedMemory]) -> List[float]:
            """Return log p(gold) for each row of this batch on top of `memory`."""
            return score_gold_logprob(model, tokenizer, prompts, golds, memory)[0]

        # The real run records every agent's hidden states for the logit lens.
        recorders = {agent: RecordingMapping(mapping) for agent in LATENT_AGENTS}
        real_memory, _ = build_memory(questions, recorders)
        # The donor run gives the latent vectors that each agent produced on a different question.
        _, donor_stats = build_memory(donors, repeat_for_each_agent(mapping))

        scores = {"real": score_memory(real_memory), "no_memory": score_memory(None)}
        for agent in LATENT_AGENTS:
            swapped_mappings = repeat_for_each_agent(mapping)
            swapped_mappings[agent] = ReplayMapping(donor_stats[agent]["latent_vectors"])
            scores[f"swap_{agent}"] = score_memory(build_memory(questions, swapped_mappings)[0])
            paused_mappings = repeat_for_each_agent(mapping)
            paused_mappings[agent] = pause
            scores[f"pause_{agent}"] = score_memory(build_memory(questions, paused_mappings)[0])
        swap_all_mappings = {agent: ReplayMapping(donor_stats[agent]["latent_vectors"]) for agent in LATENT_AGENTS}
        scores["swap_all"] = score_memory(build_memory(questions, swap_all_mappings)[0])

        lens_by_agent = {agent: apply_logit_lens(model, tokenizer, recorders[agent].hidden_states, golds, args.top_k)
                         for agent in LATENT_AGENTS}
        for offset in range(len(batch)):
            rows.append({"index": batch_start + offset, "gold": golds[offset], "model": args.model_name,
                         "task": args.task, "mapping": args.mapping, "latent_steps": args.latent_steps,
                         "gold_logprob": {name: values[offset] for name, values in scores.items()},
                         "logit_lens": {agent: lens_by_agent[agent][offset] for agent in LATENT_AGENTS}})
        print(f"done {batch_start + len(batch)}/{len(items)}", flush=True)

    print_report(rows)
    save_jsonl(rows, f"{args.output_dir}/exp6_{get_model_tag(args.model_name)}_{args.task}_{args.mapping}"
                     f"_m{args.latent_steps}.jsonl")


def print_report(rows: List[Dict[str, Any]]) -> None:
    """Print the activation-patching deltas and the logit-lens summary."""
    print("\nActivation patching: delta = log p(gold | real) - log p(gold | patched); > 0 means the patch hurt")
    real_scores = [row["gold_logprob"]["real"] for row in rows]
    print(f"{'patched':<18}{'mean delta':>12}{'95% CI':>22}")
    for name in rows[0]["gold_logprob"]:
        if name == "real":
            continue
        mean, (lower, upper) = compute_paired_mean_difference(real_scores, [row["gold_logprob"][name] for row in rows])
        print(f"{name:<18}{mean:>12.3f}   [{lower:>7.3f}, {upper:>7.3f}]")

    print("\nLogit lens: median rank of the gold answer's first token (1 = top choice)")
    for agent in LATENT_AGENTS:
        ranks_by_step = defaultdict(list)
        for row in rows:
            for step_record in row["logit_lens"][agent]:
                if step_record["gold_rank"] is not None:
                    ranks_by_step[step_record["step"]].append(step_record["gold_rank"])
        if not ranks_by_step:
            continue
        shown_steps = sorted(ranks_by_step)[:: max(1, len(ranks_by_step) // 5)]
        cells = "  ".join(f"step {step}: {int(torch.tensor(ranks_by_step[step]).float().median())}"
                          for step in shown_steps)
        top10_share = (sum(rank <= 10 for ranks in ranks_by_step.values() for rank in ranks)
                       / sum(map(len, ranks_by_step.values())))
        print(f"  {agent:<8} {cells}   | share of steps with gold in top 10: {top10_share:.3f}")
    example = rows[0]["logit_lens"][LATENT_AGENTS[-1]]
    if example:
        print(f"\nExample (question 0, {LATENT_AGENTS[-1]}, step 0) top tokens: {example[0]['top_tokens']}")


if __name__ == "__main__":
    main()
