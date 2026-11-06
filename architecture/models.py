"""Wrap the language model, so that the methods can generate text and take latent steps through one interface.

A latent step feeds the last-layer hidden state h back as the next input after e = rescale(h W_a).
W_a is a ridge-regression map from output to input embeddings. The KV cache (key/value cache)
of all agents grows on the way. The Judger then generates text on top of that cache.

By default this code keeps the padding behaviour of the released code, which is the reproduction
path. --pad_fix turns on the corrected behaviour.
"""

import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    """Give the tokenizer a pad token, because batched tokenization needs one."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _apply_pad_fix(tokenizer: AutoTokenizer, enabled: bool) -> None:
    """Set left padding when the pad fix is on, because decoder-only generation needs left padding.

    Under right padding, every sequence shorter than the batch maximum continues from a pad
    token and not from its last real token. transformers then prints "right-padding was
    detected!" on every batch. The fix is off by default. The authors ran with right padding,
    so the unfixed path stays the reproduction path.
    """
    if enabled:
        tokenizer.padding_side = "left"


def _past_length(past_key_values: Optional[Tuple]) -> int:
    """Return the number of positions in the cache. Keys have the shape [batch, heads, positions, head_dim]."""
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


class ModelWrapper:
    """Hold the tokenizer and the model (Hugging Face, vLLM or both) and provide all generation calls."""

    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        # --pad_fix turns on all three parts of the right-padding fix together: left padding,
        # the corrected generation slice, and real attention masks through the latent loop.
        # Any one part alone is wrong. See the help text in run.py.
        self.pad_fix = bool(getattr(args, "pad_fix", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

        # Ablation code can read the last aligned vector before rescaling from here.
        self.pre_aligned = None

        if self.use_vllm:
            
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                ).to(args.device2).eval() 
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            _apply_pad_fix(self.tokenizer, self.pad_fix)
            return  # The vLLM path does not load a second copy on self.model.

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        _apply_pad_fix(self.tokenizer, self.pad_fix)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        """Return the chat as one prompt string, with a plain layout if the tokenizer has no chat template."""
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        """Return (prompt, input ids [1, seq_len], attention mask [1, seq_len], tokens) for one chat."""
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        """Return (prompts, input ids [batch, seq_len], attention mask [batch, seq_len], tokens) for a batch."""
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        """Return one generated text per prompt from the vLLM engine."""
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (W_a [hidden, hidden], target norm), which map a hidden state to an input embedding.

        W_a = (W_out^T W_out + lambda I)^-1 W_out^T W_in, with lambda = 1e-5.
        The target norm is the mean norm of the input token embeddings.
        Without --latent_space_realign, W_a is the identity and only the rescale remains.
        """
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if self.args.latent_space_realign:
            pass
        else:
            # The identity keeps the rescale in _apply_latent_realignment and drops the realignment.
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the cached (W_a, target norm) for this model on `device`, and build them on first use."""
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """Return e = rescale(h W_a), as [batch, hidden], in the dtype of `hidden`."""
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
        past_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        """Generate text on top of an optional cache, and return (texts, cache).

        Without --pad_fix, the mask of the past cache is all ones, as in the released code.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                if (
                    self.pad_fix
                    and past_attention_mask is not None
                    and past_attention_mask.shape[-1] == past_len
                ):
                    # This is the real mask from the latent loop. The torch.ones() below makes every
                    # pad position in the inherited cache visible again, which is the second
                    # half of the padding bug.
                    past_mask = past_attention_mask.to(
                        dtype=attention_mask.dtype, device=attention_mask.device
                    )
                else:
                    past_mask = torch.ones(
                        (attention_mask.shape[0], past_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            # Right padding puts the real tokens at [0, length), so generation starts at `length`.
            # Left padding puts them at the end of the padded block, so generation starts after
            # the full padded width. The right-padding slice on a left-padded batch gives pad
            # tokens and a truncated answer. skip_special_tokens hides the pads, so nothing
            # warns about this. That is why padding_side cannot change alone.
            start = input_ids.shape[1] if self.pad_fix else length
            generated_ids = sequences[idx, start:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        """Return the token ids of `text`, as [1, seq_len]."""
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        past_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Run one latent agent on the Hugging Face model, and return (past_key_values, running mask).

        The agent reads its prompt on top of the cache, then takes `latent_steps` latent steps.
        The caller passes the mask to the next agent and finally to the Judger, so that they can
        tell which cache positions are pads. Under --pad_fix the mask is real. Without that
        flag, the code keeps the old all-ones behaviour, because right padding is the
        reproduction path.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                if (
                    self.pad_fix
                    and past_attention_mask is not None
                    and past_attention_mask.shape[-1] == past_len
                ):
                    past_mask = past_attention_mask.to(
                        dtype=attention_mask.dtype, device=attention_mask.device
                    )
                else:
                    past_mask = torch.ones(
                        (attention_mask.shape[0], past_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        # A direct model(...) call does not derive position_ids from the mask, unlike generate().
        # It uses sequential cache positions. That is correct under right padding, where real
        # tokens start at 0. Under left padding, it shifts every padded row by its pad width.
        # This code therefore computes the positions when the fix is on.
        position_ids = None
        if self.pad_fix:
            position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
            position_ids = position_ids[:, -input_ids.shape[1]:]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        # `attention_mask` now covers the positions in `past`.
        running_mask = attention_mask

        # Position -1 is the last position of the row. Under left padding, that is the last real
        # prompt token, so the first latent step starts from the right state. Under right padding,
        # it is a pad for every sequence shorter than the batch maximum, which was 93.3% of
        # sequences when measured. This is the first half of the padding bug. Left padding makes
        # this index correct, so --pad_fix changes padding_side and does not gather at sum(mask)-1.
        #
        # Here B = batch and D = hidden size. e_t is the input embedding of the last position.
        # TODO: e_t, h_t, e_t_plus_1 and latent_vecs_all never leave this function, and nothing uses them.
        e_t = outputs.hidden_states[0][:, -1, :]          # [B, D]
        last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D]
        h_t = last_hidden.detach().clone()

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())

        for step in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)

            latent_vecs_all.append(latent_vec.detach().clone())

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()
            
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            if self.pad_fix:
                # This extends the real mask by one position for the new latent entry.
                # The torch.ones(past_len + 1) below makes every pad in the cache visible again.
                # Pads were 35.8% of KV positions when measured.
                latent_mask = torch.cat(
                    [
                        running_mask,
                        torch.ones(
                            (latent_embed.shape[0], 1),
                            dtype=running_mask.dtype,
                            device=running_mask.device,
                        ),
                    ],
                    dim=-1,
                )
            else:
                latent_mask = torch.ones(
                    (latent_embed.shape[0], past_len + 1),
                    dtype=torch.long,
                    device=self.device,
                )
            latent_pos = None
            if self.pad_fix:
                # The new position number is the count of real positions before it.
                latent_pos = (latent_mask.long().cumsum(dim=-1) - 1).clamp_min(0)[:, -1:]

            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                position_ids=latent_pos,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            running_mask = latent_mask
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past, running_mask
    
    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        """Run one latent agent on the second Hugging Face model for the vLLM path.

        The function returns (past_key_values, embeddings). The embeddings have the shape
        [batch, seq_len + latent_steps, hidden]. They hold the prompt's input embeddings and the
        latent embeddings, because vLLM reads embeddings and cannot read a Hugging Face KV cache.
        This path has no --pad_fix handling.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])  # [batch, seq_len, hidden]
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1)

