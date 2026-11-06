# 05 Realignment matrix W_a

**Status: CONFIRMED** by computation from the model weights.

`W_a` maps a hidden state from output space to input space. The code uses it only with `--latent_space_realign`.

| Model | Tied embeddings | Distance of `W_a` from identity | Mean cosine between `h` and `h · W_a` |
|---|---|---|---|
| Qwen3-4B | Yes | 0.0 | 1.0000 |
| Qwen3-8B | No | not computed from weights | 0.0274 |
| Qwen3-14B | No | 1.05 | 0.0042 |

What this means:

- **Qwen3-4B.** The input and output embedding matrices are the same, so `W_a` is the identity. The flag cannot change the method. The small difference between runs with and without it (23/45 against 25/45) is sampling noise.
- **Qwen3-8B and 14B.** `W_a` turns vectors to nearly a right angle. A run without the flag feeds output-space vectors into the input slot. It is therefore not the method in the paper, and all our 14B runs are of this kind.

The 8B cosine comes from the content audit, which measures hidden states on real questions. See `../08_content_audit/`.

Not tested: any accuracy run on 8B or 14B with the flag on. Also not tested: the paper scales by one global constant, while the code rescales each vector to the mean embedding length.

## Evidence

- `realign_matrix_Qwen3-4B.json`, `realign_matrix_Qwen3-14B.json`: output of `results/test_runs/analysis/measure_realign_matrix.py`
