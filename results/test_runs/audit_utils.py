"""Provide the mappings and scorers that the content audit and the message decoding share.

The extra mappings turn a hidden state into the next input embedding. They use the same norm
rescale as the repo, so the conditions differ only in the direction of the latent vector:

    ConstantMapping        pause-token control: the same content-free vector at every step
    RandomRotationMapping  e = rescale(h Q), where Q is a fixed random orthogonal matrix
    ReplayMapping          ignores h and replays latent vectors recorded on another question
    RecordingMapping       wraps any mapping and keeps every hidden state it receives

The other helpers:

    make_forced_steps_rule          makes a stop rule that gives each row an exact number of steps
    score_gold_logprob              returns the teacher-forced log p(gold answer) for the Judger
    compute_bootstrap_ci            returns a 95% bootstrap interval of a mean
    compute_paired_mean_difference  returns the mean paired difference and its 95% interval

Teacher forcing means that the code feeds the gold answer as input and reads the probability
that the model gives to each gold token, without generating text.
"""

from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from results.test_runs.latent_mapping import (LatentMapping, compute_mean_token_norm, get_input_embedding_matrix,
                                        rescale_to_norm)
from results.test_runs.latent_pipeline import SharedMemory, StopRule, append_to_memory

# The code appends this text after the Judger's generation prompt for teacher forcing.
# For Qwen3 this is the documented "empty thinking" format, followed by the answer slot.
DEFAULT_ANSWER_PREFIX = "<think>\n\n</think>\n\nThe final answer is \\boxed{"


class ConstantMapping:
    """Feed the same vector at every step, whatever h is: this is the pause-token control.

    The vector is the mean input embedding, rescaled to the mean token norm.
    It carries no information about the question.
    It takes the same positions and costs the same compute as a real latent step.
    """

    def __init__(self, model: PreTrainedModel) -> None:
        w_in = get_input_embedding_matrix(model).float()
        self.target_norm = compute_mean_token_norm(w_in)
        self.vector = rescale_to_norm(w_in.mean(dim=0, keepdim=True), self.target_norm)  # [1, hidden_dim]

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the constant vector for each row, as [batch, hidden_dim]."""
        return self.vector.to(hidden.device, hidden.dtype).expand(hidden.shape[0], -1)


class RandomRotationMapping:
    """Compute e = rescale(h Q), where Q keeps lengths and scrambles directions.

    A fixed seed makes Q repeatable.
    """

    def __init__(self, model: PreTrainedModel, seed: int = 0) -> None:
        w_in = get_input_embedding_matrix(model).float()
        hidden_dim = w_in.shape[1]
        generator = torch.Generator().manual_seed(seed)
        orthogonal, triangular = torch.linalg.qr(torch.randn(hidden_dim, hidden_dim, generator=generator))
        # The sign correction makes the distribution of Q uniform over orthogonal matrices.
        self.matrix = (orthogonal * torch.sign(torch.diagonal(triangular))).to(w_in.device)
        self.target_norm = compute_mean_token_norm(w_in)

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return rescale(h Q), with the same shape and dtype as `hidden`."""
        return rescale_to_norm(hidden.float() @ self.matrix, self.target_norm).to(hidden.dtype)


class ReplayMapping:
    """Feed back latent vectors recorded on a different question, one vector per step."""

    def __init__(self, vectors: List[torch.Tensor]) -> None:
        self.vectors = vectors
        self.step = 0

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the recorded vector of the current step. After the last step, repeat the last vector."""
        vector = self.vectors[min(self.step, len(self.vectors) - 1)]
        self.step += 1
        return vector.to(hidden.device, hidden.dtype)


class RecordingMapping:
    """Wrap a mapping and keep each hidden state it receives (h_0 ... h_{m-1}) on the CPU."""

    def __init__(self, inner: LatentMapping) -> None:
        self.inner = inner
        self.hidden_states: List[torch.Tensor] = []

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """Record `hidden`, then return the result of the wrapped mapping."""
        self.hidden_states.append(hidden.detach().float().cpu())
        return self.inner(hidden)


def make_forced_steps_rule(steps_per_row: List[int]) -> StopRule:
    """Return a stop rule under which row r stops after exactly steps_per_row[r] latent steps."""
    def should_stop(step: int, previous_hidden: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Return True for each row that has used all its steps."""
        targets = torch.tensor(steps_per_row, device=hidden.device)
        return (step + 1) >= targets
    return should_stop


@torch.no_grad()
def score_gold_logprob(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: List[str],
                       golds: List[str], memory: Optional[SharedMemory],
                       answer_prefix: str = DEFAULT_ANSWER_PREFIX) -> Tuple[List[float], List[bool]]:
    """Return the teacher-forced score of the gold answer for each row, as (log_probs, top1_hits).

    The function feeds prompt + answer_prefix + gold + "}" on top of a copy of the memory.
    The first list holds the summed log-probability of the tokens of gold + "}".
    The second list says whether the model's top choice at the first gold position is the first gold token.
    """
    batch_size = len(prompts)
    memory = memory.copy() if memory is not None else SharedMemory.empty(batch_size, model.device)

    # Assumption: tokenizing the context and the gold text separately gives the same token
    # boundary that the model would produce when it writes the answer.
    contexts = [tokenizer(prompt + answer_prefix, add_special_tokens=False)["input_ids"] for prompt in prompts]
    targets = [tokenizer(str(gold) + "}", add_special_tokens=False)["input_ids"] for gold in golds]
    token_rows = [context + target for context, target in zip(contexts, targets)]
    width = max(len(token_row) for token_row in token_rows)
    pad_id = tokenizer.pad_token_id
    # Left padding keeps the targets at the right end of every row.
    input_ids = torch.tensor([[pad_id] * (width - len(token_row)) + token_row for token_row in token_rows],
                             device=model.device)  # [batch, width]
    new_mask = torch.tensor([[0] * (width - len(token_row)) + [1] * len(token_row) for token_row in token_rows],
                            device=model.device)   # [batch, width]

    logits = append_to_memory(model, memory, new_mask, input_ids=input_ids).logits.float()  # [batch, width, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)

    scores, top1_hits = [], []
    for row, target in enumerate(targets):
        target_len = len(target)
        # Positions width-target_len-1 ... width-2 predict the target tokens.
        predicted = log_probs[row, width - target_len - 1: width - 1]  # [target_len, vocab]
        target_ids = torch.tensor(target, device=model.device)
        scores.append(predicted.gather(-1, target_ids.unsqueeze(-1)).sum().item())
        top1_hits.append(bool(predicted[0].argmax().item() == target[0]))
    return scores, top1_hits


def compute_bootstrap_ci(diffs: torch.Tensor, n_boot: int = 5000, seed: int = 0) -> Tuple[float, float]:
    """Return the 95% bootstrap interval of the mean of `diffs`, resampling items with replacement."""
    generator = torch.Generator().manual_seed(seed)
    sample_indices = torch.randint(0, len(diffs), (n_boot, len(diffs)), generator=generator)  # [n_boot, items]
    means = diffs[sample_indices].mean(dim=1)
    return means.quantile(0.025).item(), means.quantile(0.975).item()


def compute_paired_mean_difference(values_a: List[float], values_b: List[float], n_boot: int = 2000,
                                   seed: int = 0) -> Tuple[float, Tuple[float, float]]:
    """Return the mean of (a - b) over paired items, with its 95% bootstrap interval."""
    diffs = torch.tensor(values_a, dtype=torch.float64) - torch.tensor(values_b, dtype=torch.float64)
    if len(diffs) == 0:
        return 0.0, (0.0, 0.0)
    return diffs.mean().item(), compute_bootstrap_ci(diffs, n_boot, seed)
