"""A small, self-contained LatentMAS pipeline shared by every experiment.

It follows the repo's sequential setup (Planner -> Critic -> Refiner think in latent
space, the Judger writes the answer) and reuses the repo's prompts and scorers, but
fixes the batching bugs found in models.py:

  * prompts are padded on the LEFT, so the last column is always a real token and
    latent steps start from a real hidden state, not a pad;
  * the real attention mask is carried across agents instead of an all-ones mask,
    so pad positions from one agent are never visible to the next;
  * RoPE positions count only real tokens, so a padded row sees exactly the same
    positions it would see on its own (batch size no longer changes results).

Everything an agent leaves behind lives in a SharedMemory: the KV cache, its mask,
and a list of named segments so experiments can drop parts of it.
"""

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from prompts import build_agent_message_sequential_latent_mas

LATENT_AGENTS = ["planner", "critic", "refiner"]

# should_stop(step, previous_hidden, hidden) -> bool tensor [batch]; True = this row is done.
StopRule = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- model


def pick_device(device: str = "auto") -> str:
    """'auto' picks a CUDA GPU, then an Apple-silicon GPU (mps), then the CPU."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    """bf16 on GPUs that support it, fp16 on older ones (e.g. Colab's T4), fp32 elsewhere."""
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(model_name: str, device: str = "auto"):
    """Load a causal LM and its tokenizer, set up for left padding."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = pick_device(device)
    dtype = pick_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    return model, tokenizer


# --------------------------------------------------------------------------- memory


@dataclass
class Segment:
    """A named slice [start, end) of the shared KV cache."""
    agent: str
    kind: str  # "prompt" or "latent"
    start: int
    end: int


@dataclass
class SharedMemory:
    """The KV cache that agents pass along, plus what is needed to read it correctly."""
    cache: DynamicCache
    attention_mask: torch.Tensor  # [batch, cache_len]; 1 = real entry, 0 = padding
    next_position: torch.Tensor   # [batch]; RoPE position of the row's next real token
    segments: List[Segment] = field(default_factory=list)

    @classmethod
    def empty(cls, batch_size: int, device) -> "SharedMemory":
        return cls(
            cache=DynamicCache(),
            attention_mask=torch.zeros(batch_size, 0, dtype=torch.long, device=device),
            next_position=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    @property
    def length(self) -> int:
        return self.attention_mask.shape[1]

    def copy(self) -> "SharedMemory":
        """Deep copy, so decoding on top of it doesn't change the original."""
        return copy.deepcopy(self)

    def select_rows(self, rows: List[int]) -> "SharedMemory":
        """Return a new memory holding only some batch rows (e.g. the questions to retry)."""
        index = torch.tensor(rows, dtype=torch.long, device=self.attention_mask.device)
        selected_cache = DynamicCache()
        for layer_idx, layer in enumerate(self.cache.layers):
            selected_cache.update(layer.keys[index], layer.values[index], layer_idx)
        return SharedMemory(selected_cache, self.attention_mask[index], self.next_position[index],
                            list(self.segments))

    def keep_only(self, keep_segment: Callable[[Segment], bool]) -> "SharedMemory":
        """Return a new memory holding only the segments for which keep_segment is True.

        next_position is NOT reset: new tokens continue from the original length, so
        they never reuse a RoPE position that a kept key already has.
        """
        kept = [s for s in self.segments if keep_segment(s)]
        index = torch.tensor([i for s in kept for i in range(s.start, s.end)],
                             dtype=torch.long, device=self.attention_mask.device)

        trimmed_cache = DynamicCache()
        for layer_idx, layer in enumerate(self.cache.layers):
            trimmed_cache.update(layer.keys[:, :, index], layer.values[:, :, index], layer_idx)

        new_segments, cursor = [], 0
        for s in kept:
            size = s.end - s.start
            new_segments.append(Segment(s.agent, s.kind, cursor, cursor + size))
            cursor += size

        return SharedMemory(trimmed_cache, self.attention_mask[:, index], self.next_position.clone(), new_segments)


# --------------------------------------------------------------------------- forward passes


def tokenize_left_padded(tokenizer, texts: List[str], device):
    encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


@torch.no_grad()
def append_to_memory(model, memory: SharedMemory, new_mask: torch.Tensor, *,
                     input_ids=None, inputs_embeds=None):
    """Run new tokens (ids or embeddings) on top of the memory and grow it in place.

    Returns the model outputs; outputs.hidden_states[-1][:, -1] is the hidden state
    of each row's newest token (always real, thanks to left padding).
    """
    positions = memory.next_position.unsqueeze(1) + new_mask.cumsum(dim=1) - 1
    positions = positions.clamp_min(0)

    outputs = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=torch.cat([memory.attention_mask, new_mask], dim=1),
        position_ids=positions,
        past_key_values=memory.cache,
        use_cache=True,
        output_hidden_states=True,
    )
    memory.cache = outputs.past_key_values
    memory.attention_mask = torch.cat([memory.attention_mask, new_mask], dim=1)
    memory.next_position = memory.next_position + new_mask.sum(dim=1)
    return outputs


def render_prompts(tokenizer, messages_batch, think: bool) -> List[str]:
    prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
               for m in messages_batch]
    return [p + "<think>" for p in prompts] if think else prompts


def agent_messages(model_name: str, task: str, role: str, questions: List[str]) -> List[List[dict]]:
    """The repo's sequential LatentMAS chat messages for one role."""
    prompt_args = SimpleNamespace(model_name=model_name, task=task)
    return [build_agent_message_sequential_latent_mas(role, q, method="latent_mas", args=prompt_args)
            for q in questions]


def agent_prompts(tokenizer, model_name: str, task: str, role: str, questions: List[str], think: bool):
    """The same messages rendered with the chat template, ready to tokenize."""
    return render_prompts(tokenizer, agent_messages(model_name, task, role, questions), think)


# --------------------------------------------------------------------------- latent agents


@torch.no_grad()
def run_latent_agent(model, tokenizer, memory: SharedMemory, prompts: List[str], *,
                     agent: str, latent_steps: int, mapping,
                     should_stop: Optional[StopRule] = None) -> dict:
    """One latent agent: read its prompt, then think for up to `latent_steps` steps.

    Rows that `should_stop` marks as done keep running (to keep the batch aligned),
    but their extra steps are masked out, so later agents never see them.
    Returns per-row stats: steps used and the latent vectors fed back.
    """
    input_ids, input_mask = tokenize_left_padded(tokenizer, prompts, model.device)
    start = memory.length
    outputs = append_to_memory(model, memory, input_mask, input_ids=input_ids)
    memory.segments.append(Segment(agent, "prompt", start, memory.length))

    batch_size = input_ids.shape[0]
    hidden = outputs.hidden_states[-1][:, -1]
    active = torch.ones(batch_size, dtype=torch.long, device=model.device)
    steps_used = torch.zeros(batch_size, dtype=torch.long, device=model.device)
    latent_vectors = []

    start = memory.length
    for step in range(latent_steps):
        embedding = mapping(hidden)
        latent_vectors.append(embedding)
        outputs = append_to_memory(model, memory, active.unsqueeze(1), inputs_embeds=embedding.unsqueeze(1))
        steps_used += active

        previous_hidden, hidden = hidden, outputs.hidden_states[-1][:, -1]
        if should_stop is not None:
            active = active * (~should_stop(step, previous_hidden, hidden)).long()
            if active.sum() == 0:
                break
    if memory.length > start:
        memory.segments.append(Segment(agent, "latent", start, memory.length))

    return {"steps_used": steps_used.tolist(), "latent_vectors": latent_vectors}


def build_latent_memory(model, tokenizer, questions: List[str], *, model_name: str, task: str,
                        latent_steps: int, mapping, think: bool = False,
                        should_stop: Optional[StopRule] = None):
    """Run Planner, Critic and Refiner in latent space. Returns (memory, per-agent stats)."""
    memory = SharedMemory.empty(len(questions), model.device)
    stats = {}
    for role in LATENT_AGENTS:
        prompts = agent_prompts(tokenizer, model_name, task, role, questions, think)
        stats[role] = run_latent_agent(model, tokenizer, memory, prompts, agent=role,
                                       latent_steps=latent_steps, mapping=mapping, should_stop=should_stop)
    return memory, stats


# --------------------------------------------------------------------------- text generation


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    outside_top_p = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
    sorted_probs[outside_top_p] = 0
    choice = torch.multinomial(sorted_probs / sorted_probs.sum(dim=-1, keepdim=True), 1)
    return sorted_ids.gather(-1, choice).squeeze(-1)


@torch.no_grad()
def generate_text(model, tokenizer, prompts: List[str], memory: Optional[SharedMemory] = None, *,
                  max_new_tokens: int = 512, temperature: float = 0.6, top_p: float = 0.95) -> List[str]:
    """Decode an answer for each prompt on top of `memory` (None = no memory at all).

    The memory is copied first, so the same memory can be reused for another call.
    """
    batch_size = len(prompts)
    memory = memory.copy() if memory is not None else SharedMemory.empty(batch_size, model.device)

    input_ids, input_mask = tokenize_left_padded(tokenizer, prompts, model.device)
    outputs = append_to_memory(model, memory, input_mask, input_ids=input_ids)

    generated = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=model.device)
    for _ in range(max_new_tokens):
        next_tokens = sample_next_token(outputs.logits[:, -1], temperature, top_p)
        for row in range(batch_size):
            if not finished[row]:
                generated[row].append(next_tokens[row].item())
        finished |= next_tokens == tokenizer.eos_token_id
        if finished.all():
            break
        still_writing = (~finished).long().unsqueeze(1)
        outputs = append_to_memory(model, memory, still_writing, input_ids=next_tokens.unsqueeze(1))

    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]


def judger_prompts(tokenizer, model_name: str, task: str, questions: List[str], think: bool) -> List[str]:
    return agent_prompts(tokenizer, model_name, task, "judger", questions, think)
