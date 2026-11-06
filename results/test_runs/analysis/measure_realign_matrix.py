#!/usr/bin/env python
"""Measure how far the realignment matrix is from the identity, to show whether --latent_space_realign has any effect.

The realign arm of the latent_steps sweep is useful only if the flag changes the computation.
models.py solves W_out @ M ~= W_in by least squares. Without the flag it discards M and uses
the identity:

    if self.args.latent_space_realign:  pass
    else:                               realign_matrix = torch.eye(...)

This script measures how far M is from the identity, and what M does to realistic vectors.
If M is near the identity, the flag does nothing and the sweep has no value.
If M is far from the identity, every run without the flag fed output-space vectors into the
input-embedding slot. That is the main candidate explanation for the collapse at ls>0
(ls = latent steps).

The script reads only the weights. It needs no GPU and generates no text.

    python results/test_runs/analysis/measure_realign_matrix.py --model Qwen/Qwen3-4B
"""
import argparse
import json
import os
import sys
from typing import Tuple

# The upstream modules import each other by bare name, so architecture/ must be on the path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "architecture"))

import torch


def _load_embeddings(model_name: str, tied: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (W_in, W_out), both [vocab, hidden], read from the weight files without building the model."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = snapshot_download(model_name, allow_patterns=["*.safetensors*", "*.json"])
    want = {"model.embed_tokens.weight", "lm_head.weight"}
    found = {}
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(root, fn), framework="pt") as f:
            for k in f.keys():
                if k in want and k not in found:
                    found[k] = f.get_tensor(k).float()
        if len(found) == len(want):
            break
    W_in = found["model.embed_tokens.weight"]
    # A tied model has no separate lm_head shard, because one matrix serves both roles.
    # W_out @ M ~= W_in then becomes W @ M ~= W, and the least-squares solve returns I.
    W_out = W_in if tied else found.get("lm_head.weight")
    if W_out is None:
        raise SystemExit("lm_head.weight not found and the config says untied")
    return W_in, W_out


def main() -> None:
    """Solve for M, print how far it is from the identity, and write the JSON result."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="results/raw/sweeps/realign_check.json")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", f"/pub/{os.environ.get('USER','')}/hf")
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model)
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    print(f"{args.model}: tie_word_embeddings={tied}, hidden={cfg.hidden_size}")

    # The script needs only two tensors. It reads them from the safetensors shards,
    # because loading a whole 14B model in fp32 costs too much memory.
    W_in, W_out = _load_embeddings(args.model, tied)
    print(f"W_in {tuple(W_in.shape)}  W_out {tuple(W_out.shape)}  tied={tied}")

    # This is the computation of _build_latent_realign_matrix in models.py.
    gram = W_out.T @ W_out
    gram = gram + 1e-5 * torch.eye(gram.shape[0], dtype=gram.dtype)
    rhs = W_out.T @ W_in
    M = torch.linalg.solve(gram, rhs)
    target_norm = W_in.norm(dim=1).mean()

    D = M.shape[0]
    I = torch.eye(D, dtype=M.dtype)
    res = {
        "model": args.model,
        "hidden_dim": D,
        "weights_tied": tied,
        "target_norm": round(float(target_norm), 4),
        "M_vs_I_relative_fro": round(float((M - I).norm() / I.norm()), 4),
        "M_diag_mean": round(float(M.diagonal().mean()), 4),
        "M_offdiag_absmean": round(float((M - torch.diag(M.diagonal())).abs().mean()), 6),
    }

    # The cosine between h and h@M shows what M does to realistic vectors.
    torch.manual_seed(0)
    h = W_out[torch.randint(0, W_out.shape[0], (256,))]  # [256, hidden]. Real rows have a realistic scale.
    hm = h @ M
    cos = torch.nn.functional.cosine_similarity(h, hm, dim=-1)
    res["cosine_h_vs_hM_mean"] = round(float(cos.mean()), 4)
    res["cosine_h_vs_hM_min"] = round(float(cos.min()), 4)

    print(json.dumps(res, indent=2))
    if tied:
        verdict = (
            "TIED embeddings -- W_out IS W_in, so solving W_out @ M ~= W_in returns "
            "the identity and --latent_space_realign is a mathematical no-op on this "
            "model. The flag cannot be tested here at all."
        )
    elif res["M_vs_I_relative_fro"] > 0.5:
        verdict = (
            "UNTIED and M is FAR from identity -- the flag is a real transform, and "
            "every run without it feeds output-space vectors into the input-embedding "
            "slot with only a norm rescale."
        )
    else:
        verdict = (
            "UNTIED but M is close to identity -- the flag changes little in practice."
        )
    print("\nverdict:", verdict)
    res["verdict"] = verdict
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
