"""Shared pieces for the content audit (exp5) and message decoding (exp6).

Extra mappings (hidden state -> next input embedding), all with the same norm rescale
as the repo, so that only the *direction* of the latent vector differs between them:

    ConstantMapping        pause-token control: the same content-free vector at every step
    RandomRotationMapping  e = rescale(h Q) with Q a fixed random orthogonal matrix
    ReplayMapping          ignore h and replay latent vectors recorded on another question
    RecordingMapping       wrap any mapping and keep every hidden state it was given

Plus:
    build_memory_per_agent  run Planner/Critic/Refiner with a different mapping and stop rule per agent
    forced_steps_rule       stop rule that makes each row take exactly a given number of steps
    gold_logprob            teacher-forced log p(gold answer) for the Judger on top of a memory
"""

from typing import Dict, List, Optional

import torch

from experiments.latent_mapping import input_embedding_matrix, rescale_to_norm
from experiments.latent_pipeline import (LATENT_AGENTS, SharedMemory, agent_prompts, append_to_memory,
                                         run_latent_agent)

# Appended after the Judger's generation prompt when scoring by teacher forcing.
# For Qwen3 this is the documented "empty thinking" format, followed by the answer slot.
DEFAULT_ANSWER_PREFIX = "<think>\n\n</think>\n\nThe final answer is \\boxed{"


class ConstantMapping:
    """Pause tokens: every step feeds the same vector, whatever h is.

    The vector is the mean input embedding, rescaled to the average token norm.
    It carries no information about the question, but occupies the same positions
    and costs the same compute as a real latent step.
    """

    def __init__(self, model):
        w_in = input_embedding_matrix(model).float()
        self.target_norm = w_in.norm(dim=-1).mean()
        self.vector = rescale_to_norm(w_in.mean(dim=0, keepdim=True), self.target_norm)  # [1, d]

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.vector.to(hidden.device, hidden.dtype).expand(hidden.shape[0], -1)


class RandomRotationMapping:
    """e = rescale(h Q). Q keeps lengths but scrambles directions (a fixed seed makes it repeatable)."""

    def __init__(self, model, seed: int = 0):
        w_in = input_embedding_matrix(model).float()
        dim = w_in.shape[1]
        generator = torch.Generator().manual_seed(seed)
        q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
        self.matrix = (q * torch.sign(torch.diagonal(r))).to(w_in.device)  # sign fix -> uniform rotation
        self.target_norm = w_in.norm(dim=-1).mean()

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return rescale_to_norm(hidden.float() @ self.matrix, self.target_norm).to(hidden.dtype)


class ReplayMapping:
    """Feed back latent vectors recorded on a different question, one per step."""

    def __init__(self, vectors: List[torch.Tensor]):
        self.vectors = vectors
        self.step = 0

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        vector = self.vectors[min(self.step, len(self.vectors) - 1)]
        self.step += 1
        return vector.to(hidden.device, hidden.dtype)


class RecordingMapping:
    """Wrap a mapping and keep each hidden state it receives (h_0 ... h_{m-1}), on the CPU."""

    def __init__(self, inner):
        self.inner = inner
        self.hidden_states: List[torch.Tensor] = []

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        self.hidden_states.append(hidden.detach().float().cpu())
        return self.inner(hidden)


def forced_steps_rule(steps_per_row: List[int]):
    """Stop rule: row r stops after exactly steps_per_row[r] latent steps."""
    def should_stop(step, previous_hidden, hidden):
        targets = torch.tensor(steps_per_row, device=hidden.device)
        return (step + 1) >= targets
    return should_stop


def build_memory_per_agent(model, tokenizer, questions: List[str], *, model_name: str, task: str,
                           latent_steps: int, mappings: Dict[str, object], think: bool = False,
                           stop_rules: Optional[Dict[str, object]] = None):
    """Like latent_pipeline.build_latent_memory, but each agent can have its own mapping and stop rule."""
    stop_rules = stop_rules or {}
    memory = SharedMemory.empty(len(questions), model.device)
    stats = {}
    for role in LATENT_AGENTS:
        prompts = agent_prompts(tokenizer, model_name, task, role, questions, think)
        stats[role] = run_latent_agent(model, tokenizer, memory, prompts, agent=role, latent_steps=latent_steps,
                                       mapping=mappings[role], should_stop=stop_rules.get(role))
    return memory, stats


@torch.no_grad()
def gold_logprob(model, tokenizer, prompts: List[str], golds: List[str], memory: Optional[SharedMemory],
                 answer_prefix: str = DEFAULT_ANSWER_PREFIX):
    """Teacher-forced score of the gold answer for each row.

    Feeds  prompt + answer_prefix + gold + "}"  on top of a copy of the memory and returns,
    per row, the summed log-probability of the gold tokens ("gold + }") and whether the
    model's top choice at the first gold position is the first gold token.
    """
    batch_size = len(prompts)
    memory = memory.copy() if memory is not None else SharedMemory.empty(batch_size, model.device)

    contexts = [tokenizer(p + answer_prefix, add_special_tokens=False)["input_ids"] for p in prompts]
    targets = [tokenizer(str(g) + "}", add_special_tokens=False)["input_ids"] for g in golds]
    rows = [c + t for c, t in zip(contexts, targets)]
    width = max(len(r) for r in rows)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.tensor([[pad_id] * (width - len(r)) + r for r in rows], device=model.device)
    new_mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], device=model.device)

    logits = append_to_memory(model, memory, new_mask, input_ids=input_ids).logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)

    scores, top1 = [], []
    for row, target in enumerate(targets):
        n = len(target)
        # positions width-n-1 ... width-2 predict the n target tokens
        predicted = log_probs[row, width - n - 1: width - 1]
        target_ids = torch.tensor(target, device=model.device)
        scores.append(predicted.gather(-1, target_ids.unsqueeze(-1)).sum().item())
        top1.append(bool(predicted[0].argmax().item() == target[0]))
    return scores, top1


def paired_mean_difference(a: List[float], b: List[float], n_boot: int = 2000, seed: int = 0):
    """Mean of (a - b) over paired items, with a 95% bootstrap interval."""
    diffs = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    if len(diffs) == 0:
        return 0.0, (0.0, 0.0)
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(diffs), (n_boot, len(diffs)), generator=generator)
    means = diffs[idx].mean(dim=1)
    return diffs.mean().item(), (means.quantile(0.025).item(), means.quantile(0.975).item())
