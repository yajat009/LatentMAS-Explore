"""Compare latent mappings on the same questions, to test whether soft token embeddings beat raw hidden states.

The conditions use the mappings of latent_mapping.py:

    identity          the repo's default, which is the baseline
    ridge             the paper's W_a (near identity on tied-embedding models like Qwen3-4B)
    soft_tau{T}       softmax(top-k logits / T) . W_in, one run per temperature
    mix               alpha * ridge + (1 - alpha) * soft

The script also logs two drift diagnostics for every mapping:

    step_cosine      cosine between consecutive latent vectors (near 1 means the rollout is stuck)
    token_cosine     cosine to the nearest real token embedding (low means far from real embeddings)

Usage:
    python -m results.test_runs.run_soft_embedding_mapping --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 200 --think
"""

import argparse
from typing import Dict, List, Optional

import torch
from torch.nn.functional import cosine_similarity, normalize

from results.test_runs.eval_utils import (add_common_arguments, collect_correct_by_condition, iterate_batches, load_items,
                                    print_paired_report, save_jsonl, score_answer)
from results.test_runs.latent_mapping import build_mapping, get_input_embedding_matrix
from results.test_runs.latent_pipeline import build_judger_prompts, build_latent_memory, generate_with_args, load_model


def compute_drift_diagnostics(latent_vectors: List[torch.Tensor],
                              token_embeddings_normed: torch.Tensor) -> Dict[str, Optional[float]]:
    """Return the mean step-to-step cosine and the mean nearest-token cosine over one agent's rollout."""
    if not latent_vectors:
        return {"step_cosine": None, "token_cosine": None}
    steps = torch.stack(latent_vectors, dim=1).float()  # [batch, steps, hidden_dim]
    step_cosine = cosine_similarity(steps[:, 1:], steps[:, :-1], dim=-1).mean().item() if steps.shape[1] > 1 else None
    token_cosine = (normalize(steps, dim=-1) @ token_embeddings_normed.T).max(dim=-1).values.mean().item()
    return {"step_cosine": step_cosine, "token_cosine": token_cosine}


def main() -> None:
    """Run every mapping on each batch, then print the diagnostics and the paired report, and save the rows."""
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
    token_embeddings_normed = normalize(get_input_embedding_matrix(model).float(), dim=-1)  # [vocab, hidden_dim]

    items = load_items(args.task, args.max_samples, args.toy)
    rows = []
    for batch_start, batch, questions in iterate_batches(items, args.batch_size):
        prompts = build_judger_prompts(tokenizer, args.model_name, args.task, questions, args.think)

        for name, mapping in mappings.items():
            memory, stats = build_latent_memory(model, tokenizer, questions, model_name=args.model_name,
                                                task=args.task, latent_steps=args.latent_steps,
                                                mapping=mapping, think=args.think)
            # The Planner runs first, so earlier agents affect its rollout the least.
            diagnostics = compute_drift_diagnostics(stats["planner"]["latent_vectors"], token_embeddings_normed)
            answers = generate_with_args(model, tokenizer, prompts, memory, args)
            for offset, (item, answer) in enumerate(zip(batch, answers)):
                rows.append({"index": batch_start + offset, "condition": name, "answer": answer,
                             **diagnostics, **score_answer(args.task, answer, item)})
        print(f"done {batch_start + len(batch)}/{len(items)}")

    print("\nDrift diagnostics (Planner, mean over questions):")
    for name in mappings:
        condition_rows = [row for row in rows if row["condition"] == name]
        step_cosines = [row["step_cosine"] for row in condition_rows if row["step_cosine"] is not None]
        token_cosines = [row["token_cosine"] for row in condition_rows if row["token_cosine"] is not None]
        print(f"  {name:<16} step_cosine={sum(step_cosines) / max(1, len(step_cosines)):.3f}  "
              f"token_cosine={sum(token_cosines) / max(1, len(token_cosines)):.3f}")

    print_paired_report(collect_correct_by_condition(rows, mappings), baseline="identity")
    save_jsonl(rows, f"{args.output_dir}/soft_embedding_mapping_{args.task}_m{args.latent_steps}.jsonl")


if __name__ == "__main__":
    main()
