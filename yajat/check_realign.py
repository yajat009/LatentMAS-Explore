#!/usr/bin/env python
"""Is --latent_space_realign actually doing anything?

The ls-sweep's realign arm is only worth running if the flag changes the
computation. models.py builds M by least-squares-solving W_out @ M ~= W_in, then
throws it away and substitutes the identity unless the flag is set:

    if self.args.latent_space_realign:  pass
    else:                               realign_matrix = torch.eye(...)

So this measures how far the solved M is from the identity it replaces, and what
it does to a real hidden state. If M were near-identity the flag would be a no-op
and the sweep would be pointless; if it is far from identity, then every run ever
made without the flag has been feeding output-space vectors into an
input-embedding slot, which is the leading explanation for the ls>0 collapse.

CPU-only and weights-only -- no GPU, no generation.

    python yajat/check_realign.py --model Qwen/Qwen3-4B
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _load_embeddings(model_name, tied):
    """Read embed_tokens.weight and lm_head.weight without building the model."""
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
    # A tied model has no separate lm_head shard: the same matrix serves both, so
    # W_out @ M ~= W_in becomes W @ M ~= W and the least-squares solve returns I.
    W_out = W_in if tied else found.get("lm_head.weight")
    if W_out is None:
        raise SystemExit("lm_head.weight not found and the config says untied")
    return W_in, W_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="yajat/results/realign_check.json")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", f"/pub/{os.environ.get('USER','')}/hf")
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model)
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    print(f"{args.model}: tie_word_embeddings={tied}, hidden={cfg.hidden_size}")

    # Only two tensors are needed, so pull them straight out of the safetensors
    # shards instead of materialising a whole 14B model in fp32.
    W_in, W_out = _load_embeddings(args.model, tied)
    print(f"W_in {tuple(W_in.shape)}  W_out {tuple(W_out.shape)}  tied={tied}")

    # Exactly models.py:_build_latent_realign_matrix
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

    # What it does to realistic vectors: cosine between h and h@M.
    torch.manual_seed(0)
    h = W_out[torch.randint(0, W_out.shape[0], (256,))]  # real rows, realistic scale
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
