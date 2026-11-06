"""Direction 1: feed back soft token embeddings instead of raw hidden states.

Compares latent mappings on the same questions (see latent_mapping.py):

    identity          repo default, the baseline
    ridge             paper's W_a (near-identity on tied-embedding models like Qwen3-4B)
    soft_tau{T}       softmax(top-k logits / T) . W_in, one run per temperature
    mix               alpha * ridge + (1 - alpha) * soft

It also logs two drift diagnostics for every mapping:
    step_cosine      cosine between consecutive latent vectors (near 1 = the rollout is stuck)
    token_cosine     cosine to the nearest real token embedding (low = off the embedding manifold)

Usage:
    python -m experiments.exp1_soft_embeddings --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 200 --think
"""

import argparse

import torch
import torch.nn.functional as F

from experiments.eval_utils import add_common_arguments, load_items, print_paired_report, save_jsonl, score_answer
from experiments.latent_mapping import build_mapping, get_input_embedding_matrix
from experiments.latent_pipeline import build_latent_memory, generate_text, build_judger_prompts, load_model


def drift_diagnostics(latent_vectors, token_embeddings_normed):
    """Mean step-to-step cosine and mean nearest-token cosine over one agent's rollout."""
    if not latent_vectors:
        return {"step_cosine": None, "token_cosine": None}
    steps = torch.stack(latent_vectors, dim=1).float()  # [batch, steps, dim]
    step_cosine = F.cosine_similarity(steps[:, 1:], steps[:, :-1], dim=-1).mean().item() if steps.shape[1] > 1 else None
    token_cosine = (F.normalize(steps, dim=-1) @ token_embeddings_normed.T).max(dim=-1).values.mean().item()
    return {"step_cosine": step_cosine, "token_cosine": token_cosine}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--soft_temperatures", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--soft_top_k", type=int, default=50)
    parser.add_argument("--mix_alpha", type=float, default=0.5)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.model_name, args.device)
    mappings = {"identity": build_mapping("identity", model), "ridge": build_mapping("ridge", model)}
    for tau in args.soft_temperatures:
        mappings[f"soft_tau{tau}"] = build_mapping("soft", model, temperature=tau, top_k=args.soft_top_k)
    mappings["mix"] = build_mapping("mix", model, temperature=args.soft_temperatures[-1],
                                    top_k=args.soft_top_k, alpha=args.mix_alpha)
    token_embeddings_normed = F.normalize(get_input_embedding_matrix(model).float(), dim=-1)

    items = load_items(args.task, args.max_samples, args.toy)
    rows = []
    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start:batch_start + args.batch_size]
        questions = [item["question"] for item in batch]
        prompts = build_judger_prompts(tokenizer, args.model_name, args.task, questions, args.think)

        for name, mapping in mappings.items():
            memory, stats = build_latent_memory(model, tokenizer, questions, model_name=args.model_name,
                                                task=args.task, latent_steps=args.latent_steps,
                                                mapping=mapping, think=args.think)
            # Diagnostics from the Planner, the first agent, whose rollout is least affected by earlier agents.
            diagnostics = drift_diagnostics(stats["planner"]["latent_vectors"], token_embeddings_normed)
            answers = generate_text(model, tokenizer, prompts, memory, max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature, top_p=args.top_p)
            for offset, (item, answer) in enumerate(zip(batch, answers)):
                rows.append({"index": batch_start + offset, "condition": name, "answer": answer,
                             **diagnostics, **score_answer(args.task, answer, item)})
        print(f"done {batch_start + len(batch)}/{len(items)}")

    print("\nDrift diagnostics (Planner, mean over questions):")
    for name in mappings:
        rows_for = [r for r in rows if r["condition"] == name]
        step = [r["step_cosine"] for r in rows_for if r["step_cosine"] is not None]
        token = [r["token_cosine"] for r in rows_for if r["token_cosine"] is not None]
        print(f"  {name:<16} step_cosine={sum(step) / max(1, len(step)):.3f}  "
              f"token_cosine={sum(token) / max(1, len(token)):.3f}")

    correct_by_condition = {name: [r["correct"] for r in rows if r["condition"] == name] for name in mappings}
    print_paired_report(correct_by_condition, baseline="identity")
    save_jsonl(rows, f"{args.output_dir}/exp1_{args.task}_m{args.latent_steps}.jsonl")


if __name__ == "__main__":
    main()
