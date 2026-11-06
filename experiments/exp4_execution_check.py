"""Let the Judger run its code against the visible tests and retry once, to measure what self-repair adds.

MBPP+ and HumanEval+ questions show example asserts. After the Judger answers, the script runs
the code against those visible asserts only, and never against the hidden tests. If the asserts
fail, the Judger reads the error and writes the code again on top of the same latent memory.

Self-repair helps any pipeline, so the script measures it with and without latent memory.
The fair comparison is repair against repair:

    judger_alone / judger_alone+repair
    latent_mas   / latent_mas+repair

Usage:
    python -m experiments.exp4_execution_check --model_name Qwen/Qwen3-4B --task mbppplus --max_samples 378 --think
"""

import argparse
from typing import List, Optional, Tuple

import torch

from experiments.eval_utils import (CODE_TASKS, add_common_arguments, collect_correct_by_condition, iterate_batches,
                                    load_items, print_paired_report, run_code_with_tests, save_jsonl, score_answer)
from experiments.latent_mapping import build_mapping
from experiments.latent_pipeline import (ChatMessages, SharedMemory, build_agent_messages, build_latent_memory,
                                         generate_with_args, load_model, render_prompts)
from utils import extract_markdown_python_block


def extract_visible_tests(question: str) -> str:
    """Return the assert lines shown inside the question text."""
    return "\n".join(line.strip() for line in question.splitlines() if line.strip().startswith("assert"))


def check_against_visible_tests(answer: str, question: str) -> Tuple[bool, Optional[str]]:
    """Return (passed, short error message for the Judger)."""
    tests = extract_visible_tests(question)
    code = extract_markdown_python_block(answer)
    if code is None:
        return False, "No ```python code block was found in your answer."
    if not tests:
        return True, None
    passed, error = run_code_with_tests(code, tests)
    short_error = "\n".join((error or "").strip().splitlines()[-3:])
    return passed, short_error


def build_repair_messages(judger_messages: ChatMessages, answer: str, error: str) -> ChatMessages:
    """Return the Judger's chat with its failed answer and the error added, so that it can try again."""
    feedback = (f"Your code failed this visible test:\n{error}\n\n"
                "Fix the code and output the complete corrected solution in a ```python code block.")
    return judger_messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": feedback}]


def main() -> None:
    """Generate, check and repair answers on each batch, then print the paired report and save the rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(task="mbppplus")
    args = parser.parse_args()
    assert args.task in CODE_TASKS, "This experiment only makes sense for code tasks."
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args.model_name, args.device)
    mapping = build_mapping("identity", model)
    items = load_items(args.task, args.max_samples, args.toy)

    def generate_answers(messages_batch: List[ChatMessages], memory: Optional[SharedMemory]) -> List[str]:
        """Render the chats and decode one answer per chat on top of `memory`."""
        prompts = render_prompts(tokenizer, messages_batch, args.think)
        return generate_with_args(model, tokenizer, prompts, memory, args)

    rows = []
    for batch_start, batch, questions in iterate_batches(items, args.batch_size):
        judger_messages = build_agent_messages(args.model_name, args.task, "judger", questions)
        latent_memory, _ = build_latent_memory(model, tokenizer, questions, model_name=args.model_name,
                                               task=args.task, latent_steps=args.latent_steps,
                                               mapping=mapping, think=args.think)

        for name, memory in {"judger_alone": None, "latent_mas": latent_memory}.items():
            first_answers = generate_answers(judger_messages, memory)
            checks = [check_against_visible_tests(answer, question)
                      for answer, question in zip(first_answers, questions)]

            # Only the rows that failed their visible test retry. The others keep their first answer.
            failed_rows = [i for i, (passed, _) in enumerate(checks) if not passed]
            final_answers = list(first_answers)
            if failed_rows:
                retry_messages = [build_repair_messages(judger_messages[i], first_answers[i], checks[i][1])
                                  for i in failed_rows]
                retry_memory = memory.select_rows(failed_rows) if memory is not None else None
                for i, answer in zip(failed_rows, generate_answers(retry_messages, retry_memory)):
                    final_answers[i] = answer

            for offset, item in enumerate(batch):
                for condition, answer in [(name, first_answers[offset]), (f"{name}+repair", final_answers[offset])]:
                    rows.append({"index": batch_start + offset, "condition": condition, "answer": answer,
                                 "retried": offset in failed_rows, **score_answer(args.task, answer, item)})
        print(f"done {batch_start + len(batch)}/{len(items)}")

    conditions = ["judger_alone", "judger_alone+repair", "latent_mas", "latent_mas+repair"]
    print_paired_report(collect_correct_by_condition(rows, conditions), baseline="judger_alone")
    save_jsonl(rows, f"{args.output_dir}/exp4_{args.task}_m{args.latent_steps}.jsonl")


if __name__ == "__main__":
    main()
