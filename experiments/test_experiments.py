"""Run quick CPU checks on the experiment code with a tiny random model, to catch batching and scoring bugs.

Usage (from the repo root):
    python -m experiments.test_experiments
"""

import os
from typing import Any, Callable, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from experiments.audit_utils import ConstantMapping, make_forced_steps_rule, score_gold_logprob
from experiments.eval_utils import TOY_ITEMS, compute_mcnemar_exact_p, score_answer
from experiments.exp2_adaptive_steps import make_converged_stop_rule
from experiments.exp4_execution_check import check_against_visible_tests
from experiments.latent_mapping import build_mapping
from experiments.latent_pipeline import (AgentStats, SharedMemory, StopRule, append_to_memory, build_judger_prompts,
                                         build_latent_memory, build_memory_per_agent, load_model,
                                         tokenize_left_padded)
from experiments.make_tiny_model import main as make_tiny_model

TINY_MODEL = "/tmp/tiny-qwen3"
QUESTIONS = [item["question"] for item in TOY_ITEMS["gsm8k"]]  # The lengths differ, so batches get padding.


def compute_judger_logits(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, questions: List[str],
                          memory_transform: Callable[[SharedMemory], SharedMemory] = lambda memory: memory,
                          should_stop: Optional[StopRule] = None) -> torch.Tensor:
    """Return the Judger's next-token logits, as [batch, vocab], on top of a transformed latent memory."""
    memory, _ = build_latent_memory(model, tokenizer, questions, model_name=TINY_MODEL, task="gsm8k",
                                    latent_steps=8, mapping=build_mapping("identity", model),
                                    should_stop=should_stop)
    memory = memory_transform(memory)
    input_ids, input_mask = tokenize_left_padded(
        tokenizer, build_judger_prompts(tokenizer, TINY_MODEL, "gsm8k", questions, False), model.device)
    return append_to_memory(model, memory, input_mask, input_ids=input_ids).logits[:, -1]


def check_batched_equals_one_by_one(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                                    **kwargs: Any) -> bool:
    """Return True if a padded batch gives the same logits as one question at a time."""
    batched = compute_judger_logits(model, tokenizer, QUESTIONS, **kwargs)
    one_by_one = torch.cat([compute_judger_logits(model, tokenizer, [question], **kwargs)
                            for question in QUESTIONS])
    return torch.allclose(batched, one_by_one, atol=1e-4)


def test_batching_does_not_change_results(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    """Check that the batch size does not change the Judger's logits."""
    assert check_batched_equals_one_by_one(model, tokenizer)


def test_early_stopping_is_batch_independent(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    """Check that rows that stop early do not affect the other rows of the batch."""
    assert check_batched_equals_one_by_one(model, tokenizer, should_stop=make_converged_stop_rule(0.97, 2))


def test_keeping_every_segment_changes_nothing(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    """Check that a keep_only call that keeps all segments leaves the logits unchanged."""
    full = compute_judger_logits(model, tokenizer, QUESTIONS)
    kept = compute_judger_logits(model, tokenizer, QUESTIONS,
                                 memory_transform=lambda memory: memory.keep_only(lambda segment: True))
    assert torch.allclose(full, kept, atol=1e-5)


def test_ridge_is_identity_with_tied_embeddings(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    """Check that W_a is almost the identity when the input and output embeddings are tied."""
    hidden = torch.randn(4, model.config.hidden_size)  # [batch, hidden_dim]
    assert torch.allclose(build_mapping("ridge", model)(hidden), build_mapping("identity", model)(hidden), atol=1e-3)


def test_code_scoring_and_visible_test_check() -> None:
    """Check that scoring and the visible-test check accept good code and reject bad code."""
    item = TOY_ITEMS["mbppplus"][0]
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert score_answer("mbppplus", good, item)["correct"]
    assert not score_answer("mbppplus", bad, item)["correct"]
    assert check_against_visible_tests(good, item["question"])[0]
    passed, error = check_against_visible_tests(bad, item["question"])
    assert not passed and "AssertionError" in error


def test_math_scoring() -> None:
    """Check that math scoring reads the boxed answer."""
    assert score_answer("gsm8k", "so the answer is \\boxed{7}", {"gold": "7"})["correct"]
    assert not score_answer("gsm8k", "\\boxed{8}", {"gold": "7"})["correct"]


def test_mcnemar() -> None:
    """Check the McNemar p-value at no disagreement and at one-sided disagreement."""
    assert compute_mcnemar_exact_p(0, 0) == 1.0
    assert compute_mcnemar_exact_p(10, 0) < 0.01


def test_forced_steps_are_exact_and_batch_independent(model: PreTrainedModel,
                                                      tokenizer: PreTrainedTokenizerBase) -> None:
    """Check that audit controls use exactly the real run's per-row, per-agent step counts."""
    targets = {"planner": [3, 7, 5], "critic": [2, 2, 9], "refiner": [6, 1, 4]}
    pause = {agent: ConstantMapping(model) for agent in targets}

    def run_rows(rows: List[int]) -> Tuple[SharedMemory, AgentStats]:
        """Build the pause memory for these question rows, with each row's forced step counts."""
        stop_rules = {agent: make_forced_steps_rule([steps[row] for row in rows]) for agent, steps in targets.items()}
        return build_memory_per_agent(model, tokenizer, [QUESTIONS[row] for row in rows], model_name=TINY_MODEL,
                                      task="gsm8k", latent_steps=9, mappings=pause, stop_rules=stop_rules)

    memory, stats = run_rows([0, 1, 2])
    assert {agent: stats[agent]["steps_used"] for agent in targets} == targets
    prompts = build_judger_prompts(tokenizer, TINY_MODEL, "gsm8k", QUESTIONS, False)
    golds = [str(item["gold"]) for item in TOY_ITEMS["gsm8k"]]
    batched = score_gold_logprob(model, tokenizer, prompts, golds, memory)[0]
    one_by_one = [score_gold_logprob(model, tokenizer, [prompts[row]], [golds[row]], run_rows([row])[0])[0][0]
                  for row in range(3)]
    assert all(abs(batched_score - single_score) < 1e-3
               for batched_score, single_score in zip(batched, one_by_one))


if __name__ == "__main__":
    if not os.path.exists(TINY_MODEL):
        make_tiny_model()
    model, tokenizer = load_model(TINY_MODEL, "cpu")
    tests = [(name, test_function) for name, test_function in globals().items() if name.startswith("test_")]
    for name, test in tests:
        # Tests that need the model take exactly two arguments.
        needs_model = test.__code__.co_argcount == 2
        test(model, tokenizer) if needs_model else test()
        print(f"ok  {name}")
    print(f"All {len(tests)} checks passed.")
