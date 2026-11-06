"""Turn a last-layer hidden state into the next input embedding, because a latent step feeds the model its own output.

The hidden state h lives in the model's output space. The model reads input embeddings e.
A mapping bridges the two spaces:

    identity  e = rescale(h). This is the repo's default.
    ridge     e = rescale(h W_a). W_a is the paper's ridge-regression map (repo flag --latent_space_realign).
    soft      e = the probability-weighted average of real token embeddings.
    mix       e = alpha * ridge + (1 - alpha) * soft.

rescale() sets each vector to the mean norm of the token embeddings.
Every mapping is a callable: mapping(hidden [batch, hidden_dim]) -> embedding [batch, hidden_dim].
"""

from typing import Callable

import torch
from transformers import PreTrainedModel

LatentMapping = Callable[[torch.Tensor], torch.Tensor]


def get_input_embedding_matrix(model: PreTrainedModel) -> torch.Tensor:
    """Return W_in, the token embedding table, as a tensor of shape [vocab, hidden_dim]."""
    return model.get_input_embeddings().weight.detach()


def get_output_embedding_matrix(model: PreTrainedModel) -> torch.Tensor:
    """Return W_out, the weights of the language-model head, as a tensor of shape [vocab, hidden_dim]."""
    return model.get_output_embeddings().weight.detach()


def compute_mean_token_norm(token_embeddings: torch.Tensor) -> torch.Tensor:
    """Return the mean norm of the rows of [vocab, hidden_dim], which is the target length of a latent vector."""
    return token_embeddings.norm(dim=-1).mean()


def rescale_to_norm(vectors: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    """Scale each vector to target_norm, so that the mappings differ only in direction."""
    return vectors * (target_norm / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-6))


class IdentityMapping:
    """Feed back the hidden state with only its length changed, which is the repo's default."""

    def __init__(self, model: PreTrainedModel) -> None:
        self.target_norm = compute_mean_token_norm(get_input_embedding_matrix(model).float())

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return rescale(h), with the same shape and dtype as `hidden`."""
        return rescale_to_norm(hidden.float(), self.target_norm).to(hidden.dtype)


class RidgeMapping:
    """Apply the paper's W_a = (W_out^T W_out + lambda I)^-1 W_out^T W_in, then rescale.

    W_a is the ridge-regression map from output embeddings to input embeddings.
    Models with tied embeddings (Qwen3-0.6B and Qwen3-4B) have W_out == W_in.
    W_a is then almost the identity, so this class behaves like IdentityMapping.
    """

    def __init__(self, model: PreTrainedModel, ridge_lambda: float = 1e-5) -> None:
        w_in = get_input_embedding_matrix(model).float()    # [vocab, hidden_dim]
        w_out = get_output_embedding_matrix(model).float()  # [vocab, hidden_dim]
        gram = w_out.T @ w_out + ridge_lambda * torch.eye(w_out.shape[1], device=w_out.device)
        self.matrix = torch.linalg.solve(gram, w_out.T @ w_in)  # [hidden_dim, hidden_dim]
        self.target_norm = compute_mean_token_norm(w_in)

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return rescale(h W_a), with the same shape and dtype as `hidden`."""
        aligned = hidden.float() @ self.matrix
        return rescale_to_norm(aligned, self.target_norm).to(hidden.dtype)


class SoftTokenMapping:
    """Compute e = softmax(top_k(h W_out^T) / tau) W_in, a weighted average of real token embeddings.

    The result is a convex combination of token embeddings.
    It therefore stays inside the region of real embeddings, and a raw hidden state does not.
    """

    def __init__(self, model: PreTrainedModel, temperature: float = 1.0, top_k: int = 50) -> None:
        self.w_in = get_input_embedding_matrix(model)
        self.lm_head = model.get_output_embeddings()
        self.temperature = temperature
        self.top_k = top_k

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the soft embedding. This mapping does not rescale its result."""
        logits = self.lm_head(hidden).float()                                 # [batch, vocab]
        top_logits, top_ids = logits.topk(min(self.top_k, logits.shape[-1]), dim=-1)
        weights = torch.softmax(top_logits / self.temperature, dim=-1)        # [batch, k]
        top_embeddings = self.w_in[top_ids].float()                           # [batch, k, hidden_dim]
        soft_embedding = (weights.unsqueeze(-1) * top_embeddings).sum(dim=1)  # [batch, hidden_dim]
        return soft_embedding.to(hidden.dtype)


class MixMapping:
    """Blend the ridge and soft mappings as alpha * ridge + (1 - alpha) * soft."""

    def __init__(self, model: PreTrainedModel, alpha: float = 0.5, temperature: float = 1.0,
                 top_k: int = 50) -> None:
        self.ridge = RidgeMapping(model)
        self.soft = SoftTokenMapping(model, temperature, top_k)
        self.alpha = alpha

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the blended embedding. This mapping does not rescale the sum."""
        return self.alpha * self.ridge(hidden) + (1 - self.alpha) * self.soft(hidden)


def build_mapping(name: str, model: PreTrainedModel, temperature: float = 1.0, top_k: int = 50,
                  alpha: float = 0.5) -> LatentMapping:
    """Return the mapping called `name`, so that scripts can select a mapping by a string."""
    if name == "identity":
        return IdentityMapping(model)
    if name == "ridge":
        return RidgeMapping(model)
    if name == "soft":
        return SoftTokenMapping(model, temperature, top_k)
    if name == "mix":
        return MixMapping(model, alpha, temperature, top_k)
    raise ValueError(f"Unknown latent mapping: {name}")
