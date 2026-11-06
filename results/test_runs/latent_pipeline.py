"""Run one small LatentMAS pipeline, so that every experiment compares conditions on correct batching.

LatentMAS (latent multi-agent system) has four agents. The Planner, the Critic and the Refiner
write no text. Each one reads its prompt on top of a shared KV cache (key/value cache), then takes
latent steps. A latent step feeds the last-layer hidden state back as the next input.
The Judger then writes the answer on top of the whole cache.

This module follows the repo's sequential layout and reuses the repo's prompts.
It also fixes three batching bugs in models.py:

  * The tokenizer pads prompts on the left. The last column is then always a real token,
    so latent steps start from a real hidden state and not from a pad.
  * The pipeline carries the real attention mask across agents, in place of an all-ones mask.
    The next agent therefore never sees the pad positions of an earlier agent.
  * RoPE (rotary position embedding) positions count only real tokens. A padded row then sees
    the same positions that it sees alone, so the batch size does not change the results.

A SharedMemory object holds what the agents leave behind: the KV cache, its mask, and a list of
named segments. Experiments use the segments to drop parts of the cache.
"""

import argparse
import copy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel,
                          PreTrainedTokenizerBase)

from results.test_runs.latent_mapping import LatentMapping
from prompts import build_agent_message_sequential_latent_mas

LATENT_AGENTS = ["planner", "critic", "refiner"]

# should_stop(step, previous_hidden, hidden) -> bool tensor [batch]. True means that the row is done.
StopRule = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]

# One entry per latent agent: {"steps_used": [per row], "latent_vectors": [per step, [batch, hidden_dim]]}.
AgentStats = Dict[str, Dict[str, Any]]

ChatMessages = List[Dict[str, str]]

PerAgentValue = TypeVar("PerAgentValue")


# --------------------------------------------------------------------------- model


def pick_device(device: str = "auto") -> str:
    """Return the device to use: for 'auto', a CUDA GPU, then an Apple-silicon GPU (mps), then the CPU."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    """Return bf16 on GPUs that support it, fp16 on older GPUs (for example Colab's T4), and fp32 elsewhere."""
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(model_name: str, device: str = "auto") -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal language model and its tokenizer, with the tokenizer set to pad on the left."""
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
    """Name the slice [start, end) of the shared KV cache, so that experiments can keep or drop it."""
    agent: str
    kind: str  # "prompt" or "latent"
    start: int
    end: int


@dataclass
class SharedMemory:
    """Hold the KV cache that the agents pass along, and the data needed to read it correctly."""
    cache: DynamicCache
    attention_mask: torch.Tensor  # [batch, cache_len]. 1 = real entry, 0 = padding.
    next_position: torch.Tensor   # [batch]. RoPE position of the row's next real token.
    segments: List[Segment] = field(default_factory=list)

    @classmethod
    def empty(cls, batch_size: int, device: Union[str, torch.device]) -> "SharedMemory":
        """Return a memory with no cached positions, which is the state before the first agent."""
        return cls(
            cache=DynamicCache(),
            attention_mask=torch.zeros(batch_size, 0, dtype=torch.long, device=device),
            next_position=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    @property
    def length(self) -> int:
        """Return the number of cache columns, including padding."""
        return self.attention_mask.shape[1]

    def copy(self) -> "SharedMemory":
        """Return a deep copy, because decoding on top of a memory grows it in place."""
        return copy.deepcopy(self)

    def select_rows(self, rows: List[int]) -> "SharedMemory":
        """Return a new memory that holds only some batch rows, for example the questions to retry."""
        index = torch.tensor(rows, dtype=torch.long, device=self.attention_mask.device)
        selected_cache = DynamicCache()
        for layer_idx, layer in enumerate(self.cache.layers):
            selected_cache.update(layer.keys[index], layer.values[index], layer_idx)
        return SharedMemory(selected_cache, self.attention_mask[index], self.next_position[index],
                            list(self.segments))

    def keep_only(self, keep_segment: Callable[[Segment], bool]) -> "SharedMemory":
        """Return a new memory that holds only the segments for which keep_segment is True.

        This method does not reset next_position. New tokens continue from the original length,
        so they never reuse a RoPE position that a kept key already has.
        """
        kept_segments = [segment for segment in self.segments if keep_segment(segment)]
        index = torch.tensor([i for segment in kept_segments for i in range(segment.start, segment.end)],
                             dtype=torch.long, device=self.attention_mask.device)

        trimmed_cache = DynamicCache()
        for layer_idx, layer in enumerate(self.cache.layers):
            # keys and values have the shape [batch, heads, positions, head_dim].
            trimmed_cache.update(layer.keys[:, :, index], layer.values[:, :, index], layer_idx)

        new_segments, cursor = [], 0
        for segment in kept_segments:
            size = segment.end - segment.start
            new_segments.append(Segment(segment.agent, segment.kind, cursor, cursor + size))
            cursor += size

        return SharedMemory(trimmed_cache, self.attention_mask[:, index], self.next_position.clone(), new_segments)


# --------------------------------------------------------------------------- forward passes


def tokenize_left_padded(tokenizer: PreTrainedTokenizerBase, texts: List[str],
                         device: Union[str, torch.device]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return input ids and attention mask, both [batch, width]. load_model set the padding side to left."""
    encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


@torch.no_grad()
def append_to_memory(model: PreTrainedModel, memory: SharedMemory, new_mask: torch.Tensor, *,
                     input_ids: Optional[torch.Tensor] = None,
                     inputs_embeds: Optional[torch.Tensor] = None) -> Any:
    """Run new tokens (ids or embeddings) on top of the memory and grow the memory in place.

    The function returns the model outputs. outputs.hidden_states[-1][:, -1] is the hidden state
    of the newest token of each row. Left padding makes that token a real token.
    """
    positions = memory.next_position.unsqueeze(1) + new_mask.cumsum(dim=1) - 1  # [batch, new_tokens]
    # Leading pads of an empty memory get -1. The mask hides those columns, so 0 is safe.
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


def render_prompts(tokenizer: PreTrainedTokenizerBase, messages_batch: List[ChatMessages],
                   think: bool) -> List[str]:
    """Render chat messages to prompt strings, and append "<think>" when `think` is True."""
    prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
               for messages in messages_batch]
    return [prompt + "<think>" for prompt in prompts] if think else prompts


def build_agent_messages(model_name: str, task: str, role: str, questions: List[str]) -> List[ChatMessages]:
    """Return the repo's sequential LatentMAS chat messages for one role, one list per question."""
    prompt_args = SimpleNamespace(model_name=model_name, task=task)
    return [build_agent_message_sequential_latent_mas(role, question, method="latent_mas", args=prompt_args)
            for question in questions]


def build_agent_prompts(tokenizer: PreTrainedTokenizerBase, model_name: str, task: str, role: str,
                        questions: List[str], think: bool) -> List[str]:
    """Return one role's messages rendered with the chat template, ready to tokenize."""
    return render_prompts(tokenizer, build_agent_messages(model_name, task, role, questions), think)


# --------------------------------------------------------------------------- latent agents


@torch.no_grad()
def run_latent_agent(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, memory: SharedMemory,
                     prompts: List[str], *, agent: str, latent_steps: int, mapping: LatentMapping,
                     should_stop: Optional[StopRule] = None) -> Dict[str, Any]:
    """Run one latent agent: it reads its prompt, then takes up to `latent_steps` latent steps.

    Rows that `should_stop` marks as done keep running, because the batch must stay aligned.
    The mask hides their extra steps, so later agents never see those steps.
    The function returns the steps used per row and the latent vectors that it fed back.
    """
    input_ids, input_mask = tokenize_left_padded(tokenizer, prompts, model.device)
    start = memory.length
    outputs = append_to_memory(model, memory, input_mask, input_ids=input_ids)
    memory.segments.append(Segment(agent, "prompt", start, memory.length))

    batch_size = input_ids.shape[0]
    hidden = outputs.hidden_states[-1][:, -1]  # [batch, hidden_dim]
    active = torch.ones(batch_size, dtype=torch.long, device=model.device)  # [batch]. 1 = still thinking.
    steps_used = torch.zeros(batch_size, dtype=torch.long, device=model.device)
    latent_vectors = []

    start = memory.length
    for step in range(latent_steps):
        embedding = mapping(hidden)  # [batch, hidden_dim]
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


def repeat_for_each_agent(value: PerAgentValue) -> Dict[str, PerAgentValue]:
    """Return {agent: value} for the three latent agents, for callers that treat all agents alike."""
    return {agent: value for agent in LATENT_AGENTS}


def build_memory_per_agent(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, questions: List[str], *,
                           model_name: str, task: str, latent_steps: int, mappings: Dict[str, LatentMapping],
                           think: bool = False, stop_rules: Optional[Dict[str, Optional[StopRule]]] = None
                           ) -> Tuple[SharedMemory, AgentStats]:
    """Run Planner, Critic and Refiner, each with its own mapping and stop rule, and return (memory, stats)."""
    stop_rules = stop_rules or {}
    memory = SharedMemory.empty(len(questions), model.device)
    stats = {}
    for role in LATENT_AGENTS:
        prompts = build_agent_prompts(tokenizer, model_name, task, role, questions, think)
        stats[role] = run_latent_agent(model, tokenizer, memory, prompts, agent=role, latent_steps=latent_steps,
                                       mapping=mappings[role], should_stop=stop_rules.get(role))
    return memory, stats


def build_latent_memory(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, questions: List[str], *,
                        model_name: str, task: str, latent_steps: int, mapping: LatentMapping,
                        think: bool = False, should_stop: Optional[StopRule] = None
                        ) -> Tuple[SharedMemory, AgentStats]:
    """Run Planner, Critic and Refiner with one shared mapping and stop rule, and return (memory, stats)."""
    return build_memory_per_agent(model, tokenizer, questions, model_name=model_name, task=task,
                                  latent_steps=latent_steps, mappings=repeat_for_each_agent(mapping),
                                  think=think, stop_rules=repeat_for_each_agent(should_stop))


# --------------------------------------------------------------------------- text generation


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Return one token id per row from [batch, vocab] logits, by greedy or top-p (nucleus) sampling."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    outside_top_p = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
    sorted_probs[outside_top_p] = 0
    choice = torch.multinomial(sorted_probs / sorted_probs.sum(dim=-1, keepdim=True), 1)  # [batch, 1]
    return sorted_ids.gather(-1, choice).squeeze(-1)


@torch.no_grad()
def generate_text(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: List[str],
                  memory: Optional[SharedMemory] = None, *, max_new_tokens: int = 512,
                  temperature: float = 0.6, top_p: float = 0.95) -> List[str]:
    """Decode an answer for each prompt on top of `memory`. None means no memory at all.

    The function copies the memory first, so a caller can reuse the same memory for another call.
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
        # Finished rows stay in the batch, and a 0 in the mask hides their new columns.
        still_writing = (~finished).long().unsqueeze(1)  # [batch, 1]
        outputs = append_to_memory(model, memory, still_writing, input_ids=next_tokens.unsqueeze(1))

    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]


def generate_with_args(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: List[str],
                       memory: Optional[SharedMemory], args: argparse.Namespace) -> List[str]:
    """Call generate_text with the sampling settings from the common command-line flags."""
    return generate_text(model, tokenizer, prompts, memory, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_p=args.top_p)


def build_judger_prompts(tokenizer: PreTrainedTokenizerBase, model_name: str, task: str, questions: List[str],
                         think: bool) -> List[str]:
    """Return the Judger's rendered prompts, one per question."""
    return build_agent_prompts(tokenizer, model_name, task, "judger", questions, think)
