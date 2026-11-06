"""Build a tiny Qwen3 model with random weights, so that the experiments can run without a GPU or a download.

The weights are random, so the answers are gibberish. The model only checks that the code
paths run end to end: shapes, masks, caches and scoring.

Usage (from the repo root):
    python -m results.test_runs.make_tiny_model --out /tmp/tiny-qwen3
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]

# This template has the layout of Qwen's chat template, without the tool-calling parts.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build_tokenizer(training_text: str) -> PreTrainedTokenizerFast:
    """Train a small byte-level BPE (byte-pair encoding) tokenizer on the repo's own prompt text."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=2000,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator([training_text], trainer=trainer)

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    wrapped.chat_template = CHAT_TEMPLATE
    return wrapped


def build_model(vocab_size: int, eos_token_id: int, pad_token_id: int) -> Qwen3ForCausalLM:
    """Return a 2-layer Qwen3 with tied embeddings, like Qwen3-0.6B and Qwen3-4B but tiny."""
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    return Qwen3ForCausalLM(config)


def main() -> None:
    """Train the tokenizer, build the model and save both to the output folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/tiny-qwen3",
                        help="Output folder. Keep 'qwen' in the name: the repo's prompts assert it.")
    args = parser.parse_args()

    # The tokenizer depends on the text of these two files, and the weights have no fixed seed.
    # A rebuilt model therefore differs from an earlier one, so saved toy outputs do not carry over.
    repo_root = Path(__file__).resolve().parents[2]
    training_text = (repo_root / "architecture" / "prompts.py").read_text() + (repo_root / "LICENSE").read_text()

    tokenizer = build_tokenizer(training_text)
    model = build_model(len(tokenizer), tokenizer.eos_token_id, tokenizer.pad_token_id)

    tokenizer.save_pretrained(args.out)
    model.save_pretrained(args.out)
    print(f"Saved tiny model ({sum(parameter.numel() for parameter in model.parameters()):,} params) to {args.out}")


if __name__ == "__main__":
    main()
