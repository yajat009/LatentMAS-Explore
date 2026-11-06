"""Provide seeding, device choice, answer extraction and safe code execution, which every method needs."""

import os
import random
import re
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch, so that a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    """Return the requested device. Without a request, return a CUDA GPU if one exists, otherwise the CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Return the model's final answer from its text, for scoring.

    The function reads the last \\boxed{...} in the text. It returns the first number in the box,
    or the box content if the box holds no number. If the text has no box, the function returns
    the last number in the text. If the text has no number, it returns None.
    """
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    """Return the number after "####" in a GSM8K solution, which is the gold answer, or None."""
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    """Return the answer stripped and in lower case, so that string comparison ignores those differences."""
    if ans is None:
        return None
    return ans.strip().lower()


def extract_markdown_python_block(text: str) -> Optional[str]:
    """Return the code of the last ```python block in the text, or None if the text has no such block."""
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    """Run Python code in a separate process and return (ok, error), so that a hung answer cannot stop the run.

    `error` holds the traceback, the timeout message, or None.
    """
    def worker(ns, code):
        """Execute the code and store the result in the shared dict `ns`."""
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)

