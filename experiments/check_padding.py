"""Check that batching does not change the Judger's output, because every paired comparison relies on it.

The script builds the latent memory and the Judger's first-token logits for the same questions
twice: once as one padded batch, and once one question at a time. With the padding fix in
latent_pipeline.py, the two results agree up to float rounding.

Usage:
    python -m experiments.check_padding --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 4
"""

import argparse
from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from experiments.eval_utils import add_common_arguments, load_items
from experiments.latent_mapping import LatentMapping, build_mapping
from experiments.latent_pipeline import (append_to_memory, build_judger_prompts, build_latent_memory,
                                         load_model, tokenize_left_padded)


def compute_judger_first_token_logits(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                                      questions: List[str], args: argparse.Namespace,
                                      mapping: LatentMapping) -> torch.Tensor:
    """Return the Judger's logits for its first answer token, as [batch, vocab] in fp32."""
    memory, _ = build_latent_memory(model, tokenizer, questions, model_name=args.model_name, task=args.task,
                                    latent_steps=args.latent_steps, mapping=mapping, think=args.think)
    prompts = build_judger_prompts(tokenizer, args.model_name, args.task, questions, args.think)
    input_ids, input_mask = tokenize_left_padded(tokenizer, prompts, model.device)
    outputs = append_to_memory(model, memory, input_mask, input_ids=input_ids)
    return outputs.logits[:, -1].float()


def main() -> None:
    """Compare batched logits with one-at-a-time logits, and print PASS or FAIL."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, args.device)
    mapping = build_mapping("identity", model)
    questions = [item["question"] for item in load_items(args.task, args.max_samples, args.toy)]

    batched = compute_judger_first_token_logits(model, tokenizer, questions, args, mapping)
    one_by_one = torch.cat([compute_judger_first_token_logits(model, tokenizer, [question], args, mapping)
                            for question in questions])

    max_difference = (batched - one_by_one).abs().max().item()
    same_top_token = (batched.argmax(-1) == one_by_one.argmax(-1)).all().item()
    print(f"Max |logit difference| batched vs one-by-one: {max_difference:.2e}")
    print(f"Same greedy first token for every question: {same_top_token}")
    # bf16 on a GPU gives larger rounding noise than fp32 on a CPU, so its bound is looser.
    tolerance = 1e-3 if model.dtype == torch.float32 else 5e-1
    print("PASS" if max_difference < tolerance and same_top_token else "FAIL")


if __name__ == "__main__":
    main()
