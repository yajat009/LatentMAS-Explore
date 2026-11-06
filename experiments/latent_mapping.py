"""Ways to turn a last-layer hidden state into the next input embedding.

A latent step feeds the model's own last hidden state back in as if it were the
next token. The hidden state lives in "output space", the model expects
"input-embedding space", so something has to bridge the two:

    identity  the repo's default: use h as-is, rescaled to the average token-embedding norm
    ridge     the paper's W_a (repo flag --latent_space_realign), then rescale
    soft      direction 1: the probability-weighted average of real token embeddings
    mix       alpha * ridge + (1 - alpha) * soft

Every mapping is a callable: mapping(hidden [batch, dim]) -> embedding [batch, dim].
"""

import torch


def input_embedding_matrix(model) -> torch.Tensor:
    return model.get_input_embeddings().weight.detach()


def output_embedding_matrix(model) -> torch.Tensor:
    return model.get_output_embeddings().weight.detach()


def rescale_to_norm(vectors: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    return vectors * (target_norm / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-6))


class IdentityMapping:
    """Feed the hidden state back unchanged, only rescaled (repo default)."""

    def __init__(self, model):
        self.target_norm = input_embedding_matrix(model).float().norm(dim=-1).mean()

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return rescale_to_norm(hidden.float(), self.target_norm).to(hidden.dtype)


class RidgeMapping:
    """The paper's W_a = (W_out^T W_out + lambda I)^-1 W_out^T W_in, then rescale.

    With tied embeddings (Qwen3-0.6B/4B) W_out == W_in, so W_a is almost the
    identity and this behaves like IdentityMapping.
    """

    def __init__(self, model, ridge_lambda: float = 1e-5):
        w_in = input_embedding_matrix(model).float()
        w_out = output_embedding_matrix(model).float()
        gram = w_out.T @ w_out + ridge_lambda * torch.eye(w_out.shape[1], device=w_out.device)
        self.matrix = torch.linalg.solve(gram, w_out.T @ w_in)
        self.target_norm = w_in.norm(dim=-1).mean()

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        aligned = hidden.float() @ self.matrix
        return rescale_to_norm(aligned, self.target_norm).to(hidden.dtype)


class SoftTokenMapping:
    """Direction 1: e = softmax(top_k(h W_out^T) / tau) W_in.

    The result is a convex combination of real token embeddings, so it cannot
    drift off the embedding manifold the way a raw hidden state can.
    """

    def __init__(self, model, temperature: float = 1.0, top_k: int = 50):
        self.w_in = input_embedding_matrix(model)
        self.lm_head = model.get_output_embeddings()
        self.temperature = temperature
        self.top_k = top_k

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden).float()
        top_logits, top_ids = logits.topk(min(self.top_k, logits.shape[-1]), dim=-1)
        weights = torch.softmax(top_logits / self.temperature, dim=-1)        # [batch, k]
        top_embeddings = self.w_in[top_ids].float()                           # [batch, k, dim]
        soft_embedding = (weights.unsqueeze(-1) * top_embeddings).sum(dim=1)  # [batch, dim]
        return soft_embedding.to(hidden.dtype)


class MixMapping:
    """alpha * ridge + (1 - alpha) * soft."""

    def __init__(self, model, alpha: float = 0.5, temperature: float = 1.0, top_k: int = 50):
        self.ridge = RidgeMapping(model)
        self.soft = SoftTokenMapping(model, temperature, top_k)
        self.alpha = alpha

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ridge(hidden) + (1 - self.alpha) * self.soft(hidden)


def build_mapping(name: str, model, temperature: float = 1.0, top_k: int = 50, alpha: float = 0.5):
    if name == "identity":
        return IdentityMapping(model)
    if name == "ridge":
        return RidgeMapping(model)
    if name == "soft":
        return SoftTokenMapping(model, temperature, top_k)
    if name == "mix":
        return MixMapping(model, alpha, temperature, top_k)
    raise ValueError(f"Unknown latent mapping: {name}")
