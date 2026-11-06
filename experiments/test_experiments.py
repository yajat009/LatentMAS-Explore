"""Quick CPU checks for the experiment code, using a tiny random model.

Usage (from the repo root):
    python -m experiments.test_experiments
"""

import os

import torch

from experiments.eval_utils import TOY_ITEMS, compute_mcnemar_exact_p, score_answer
from experiments.exp2_adaptive_steps import stop_when_converged
from experiments.exp4_execution_check import check_against_visible_tests
from experiments.latent_mapping import build_mapping
from experiments.latent_pipeline import (append_to_memory, build_latent_memory, build_judger_prompts, load_model,
                                         tokenize_left_padded)
from experiments.make_tiny_model import main as make_tiny_model

TINY_MODEL = "/tmp/tiny-qwen3"
QUESTIONS = [item["question"] for item in TOY_ITEMS["gsm8k"]]  # different lengths, so batches get padded


def judger_logits(model, tokenizer, questions, memory_transform=lambda m: m, should_stop=None):
    memory, _ = build_latent_memory(model, tokenizer, questions, model_name=TINY_MODEL, task="gsm8k",
                                    latent_steps=8, mapping=build_mapping("identity", model),
                                    should_stop=should_stop)
    memory = memory_transform(memory)
    ids, mask = tokenize_left_padded(tokenizer, build_judger_prompts(tokenizer, TINY_MODEL, "gsm8k", questions, False),
                                     model.device)
    return append_to_memory(model, memory, mask, input_ids=ids).logits[:, -1]


def batched_equals_one_by_one(model, tokenizer, **kwargs) -> bool:
    batched = judger_logits(model, tokenizer, QUESTIONS, **kwargs)
    one_by_one = torch.cat([judger_logits(model, tokenizer, [q], **kwargs) for q in QUESTIONS])
    return torch.allclose(batched, one_by_one, atol=1e-4)


def test_batching_does_not_change_results(model, tokenizer):
    assert batched_equals_one_by_one(model, tokenizer)


def test_early_stopping_is_batch_independent(model, tokenizer):
    assert batched_equals_one_by_one(model, tokenizer, should_stop=stop_when_converged(0.97, 2))


def test_keeping_every_segment_changes_nothing(model, tokenizer):
    full = judger_logits(model, tokenizer, QUESTIONS)
    kept = judger_logits(model, tokenizer, QUESTIONS, memory_transform=lambda m: m.keep_only(lambda s: True))
    assert torch.allclose(full, kept, atol=1e-5)


def test_ridge_is_identity_with_tied_embeddings(model, tokenizer):
    hidden = torch.randn(4, model.config.hidden_size)
    assert torch.allclose(build_mapping("ridge", model)(hidden), build_mapping("identity", model)(hidden), atol=1e-3)


def test_code_scoring_and_visible_test_check():
    item = TOY_ITEMS["mbppplus"][0]
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert score_answer("mbppplus", good, item)["correct"]
    assert not score_answer("mbppplus", bad, item)["correct"]
    assert check_against_visible_tests(good, item["question"])[0]
    passed, error = check_against_visible_tests(bad, item["question"])
    assert not passed and "AssertionError" in error


def test_math_scoring():
    assert score_answer("gsm8k", "so the answer is \\boxed{7}", {"gold": "7"})["correct"]
    assert not score_answer("gsm8k", "\\boxed{8}", {"gold": "7"})["correct"]


def test_mcnemar():
    assert compute_mcnemar_exact_p(0, 0) == 1.0
    assert compute_mcnemar_exact_p(10, 0) < 0.01


def test_forced_steps_are_exact_and_batch_independent(model, tokenizer):
    """Audit controls must use exactly the real run's per-row, per-agent step counts."""
    from experiments.audit_utils import ConstantMapping, make_forced_steps_rule, score_gold_logprob
    from experiments.latent_pipeline import build_memory_per_agent
    targets = {"planner": [3, 7, 5], "critic": [2, 2, 9], "refiner": [6, 1, 4]}
    pause = {a: ConstantMapping(model) for a in targets}

    def run(rows):
        return build_memory_per_agent(model, tokenizer, [QUESTIONS[r] for r in rows], model_name=TINY_MODEL,
                                      task="gsm8k", latent_steps=9, mappings=pause,
                                      stop_rules={a: make_forced_steps_rule([v[r] for r in rows]) for a, v in targets.items()})

    memory, stats = run([0, 1, 2])
    assert {a: stats[a]["steps_used"] for a in targets} == targets
    prompts = build_judger_prompts(tokenizer, TINY_MODEL, "gsm8k", QUESTIONS, False)
    golds = [str(item["gold"]) for item in TOY_ITEMS["gsm8k"]]
    batched = score_gold_logprob(model, tokenizer, prompts, golds, memory)[0]
    one_by_one = [score_gold_logprob(model, tokenizer, [prompts[r]], [golds[r]], run([r])[0])[0][0] for r in range(3)]
    assert all(abs(a - b) < 1e-3 for a, b in zip(batched, one_by_one))


if __name__ == "__main__":
    if not os.path.exists(TINY_MODEL):
        make_tiny_model()
    model, tokenizer = load_model(TINY_MODEL, "cpu")
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, test in tests:
        needs_model = test.__code__.co_argcount == 2
        test(model, tokenizer) if needs_model else test()
        print(f"ok  {name}")
    print(f"All {len(tests)} checks passed.")
