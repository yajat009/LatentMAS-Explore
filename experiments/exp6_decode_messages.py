"""Decode the latent messages: logit lens + activation patching. No text generation, so it is cheap.

Logit lens
    At every latent step of every agent, apply the LM head to the hidden state h_k and record
    the top tokens and the rank of the gold answer's first token (rank 1 = the model's top choice).
    If latent thoughts carry the solution, the gold rank should be small and fall over the steps.

Activation patching (causal)
    Score the Judger by teacher forcing: log p(gold answer) after "...\\boxed{".
    Then re-run the pipeline with ONE agent's latent vectors replaced, and measure the drop:
        swap_<agent>    that agent's latent vectors replayed from a different question
        pause_<agent>   that agent's latent vectors replaced by one content-free vector
        swap_all        all three agents' latent vectors replayed from a different question
    Later agents re-run on top of the patched memory, so this is the total effect of that
    agent's message. delta = log p(gold | real) - log p(gold | patched). delta ~ 0 everywhere
    means the Judger ignores the latent content.

Usage:
    python -m experiments.exp6_decode_messages --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 300 \
        --latent_steps 20 --mapping identity
"""

import argparse
from collections import defaultdict

import torch

from experiments.audit_utils import (ConstantMapping, RecordingMapping, ReplayMapping, build_memory_per_agent,
                                     gold_logprob, paired_mean_difference)
from experiments.eval_utils import CODE_TASKS, add_common_arguments, load_items, save_jsonl
from experiments.latent_mapping import build_mapping
from experiments.latent_pipeline import LATENT_AGENTS, judger_prompts, load_model


def gold_first_token_ids(tokenizer, gold: str):
    """Token ids that could start the gold answer, with and without a leading space."""
    ids = set()
    for text in (gold, " " + gold):
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if encoded:
            ids.add(encoded[0])
    return sorted(ids)


@torch.no_grad()
def logit_lens(model, tokenizer, hidden_states, golds, top_k: int):
    """hidden_states: list over steps of [batch, d]. Returns per row a list of per-step dicts."""
    lm_head = model.get_output_embeddings()
    per_row = [[] for _ in golds]
    gold_ids = [gold_first_token_ids(tokenizer, g) for g in golds]
    for step, h in enumerate(hidden_states):
        logits = lm_head(h.to(model.device, model.dtype)).float()  # [batch, V]
        top = logits.topk(top_k, dim=-1).indices.tolist()
        order = logits.argsort(dim=-1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
        for row in range(len(golds)):
            gold_rank = min(ranks[row, i].item() for i in gold_ids[row]) + 1 if gold_ids[row] else None
            per_row[row].append({"step": step, "gold_rank": gold_rank,
                                 "top_tokens": [tokenizer.decode([t]) for t in top[row]]})
    return per_row


def main() -> None:
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
    donor_items = items[1:] + items[:1]

    rows = []
    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start:batch_start + args.batch_size]
        questions = [item["question"] for item in batch]
        donors = [item["question"] for item in donor_items[batch_start:batch_start + len(batch)]]
        golds = [str(item.get("gold", "")) for item in batch]
        prompts = judger_prompts(tokenizer, args.model_name, args.task, questions, False)

        def build(qs, agent_mappings):
            return build_memory_per_agent(model, tokenizer, qs, model_name=args.model_name, task=args.task,
                                          latent_steps=args.latent_steps, mappings=agent_mappings, think=args.think)

        # Real run, recording every agent's hidden states for the logit lens.
        recorders = {agent: RecordingMapping(mapping) for agent in LATENT_AGENTS}
        real_memory, real_stats = build(questions, recorders)
        # Donor run: the latent vectors each agent produced on a different question.
        _, donor_stats = build(donors, {agent: mapping for agent in LATENT_AGENTS})

        scores = {"real": gold_logprob(model, tokenizer, prompts, golds, real_memory)[0],
                  "no_memory": gold_logprob(model, tokenizer, prompts, golds, None)[0]}
        for agent in LATENT_AGENTS:
            swap = {a: mapping for a in LATENT_AGENTS}
            swap[agent] = ReplayMapping(donor_stats[agent]["latent_vectors"])
            scores[f"swap_{agent}"] = gold_logprob(model, tokenizer, prompts, golds, build(questions, swap)[0])[0]
            paused = {a: mapping for a in LATENT_AGENTS}
            paused[agent] = pause
            scores[f"pause_{agent}"] = gold_logprob(model, tokenizer, prompts, golds, build(questions, paused)[0])[0]
        swap_all = {a: ReplayMapping(donor_stats[a]["latent_vectors"]) for a in LATENT_AGENTS}
        scores["swap_all"] = gold_logprob(model, tokenizer, prompts, golds, build(questions, swap_all)[0])[0]

        lens = {agent: logit_lens(model, tokenizer, recorders[agent].hidden_states, golds, args.top_k)
                for agent in LATENT_AGENTS}
        for offset, item in enumerate(batch):
            rows.append({"index": batch_start + offset, "gold": golds[offset], "model": args.model_name,
                         "task": args.task, "mapping": args.mapping, "latent_steps": args.latent_steps,
                         "gold_logprob": {k: v[offset] for k, v in scores.items()},
                         "logit_lens": {agent: lens[agent][offset] for agent in LATENT_AGENTS}})
        print(f"done {batch_start + len(batch)}/{len(items)}", flush=True)

    report(rows, args)
    model_tag = args.model_name.rstrip("/").split("/")[-1]
    save_jsonl(rows, f"{args.output_dir}/exp6_{model_tag}_{args.task}_{args.mapping}_m{args.latent_steps}.jsonl")


def report(rows, args) -> None:
    print("\nActivation patching: delta = log p(gold | real) - log p(gold | patched); > 0 means the patch hurt")
    real = [r["gold_logprob"]["real"] for r in rows]
    print(f"{'patched':<18}{'mean delta':>12}{'95% CI':>22}")
    for name in rows[0]["gold_logprob"]:
        if name == "real":
            continue
        mean, (lo, hi) = paired_mean_difference(real, [r["gold_logprob"][name] for r in rows])
        print(f"{name:<18}{mean:>12.3f}   [{lo:>7.3f}, {hi:>7.3f}]")

    print("\nLogit lens: median rank of the gold answer's first token (1 = top choice)")
    for agent in LATENT_AGENTS:
        by_step = defaultdict(list)
        for r in rows:
            for s in r["logit_lens"][agent]:
                if s["gold_rank"] is not None:
                    by_step[s["step"]].append(s["gold_rank"])
        if not by_step:
            continue
        shown = sorted(by_step)[:: max(1, len(by_step) // 5)]
        cells = "  ".join(f"step {s}: {int(torch.tensor(by_step[s]).float().median())}" for s in shown)
        top10 = sum(rank <= 10 for ranks in by_step.values() for rank in ranks) / sum(map(len, by_step.values()))
        print(f"  {agent:<8} {cells}   | share of steps with gold in top 10: {top10:.3f}")
    example = rows[0]["logit_lens"][LATENT_AGENTS[-1]]
    if example:
        print(f"\nExample (question 0, {LATENT_AGENTS[-1]}, step 0) top tokens: {example[0]['top_tokens']}")


if __name__ == "__main__":
    main()
