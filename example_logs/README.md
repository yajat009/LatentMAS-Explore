# `example_logs/` — the authors' two released run logs

Upstream artifacts, unmodified. They matter more than "examples" suggests:
**each one is exactly one cell of the paper**, confirmed to the token by
`repro/analyze.py` re-tokenizing every `[Output]` block.

| file | accuracy (log / paper) | gen tokens/problem (log / paper) | paper cell |
|---|---|---|---|
| `qwen3_14b_mbppplus_sequential.txt` | 76.19 / 75.7 | **1621.2 / 1621** | Table 1, 14B, MBPP+, sequential |
| `qwen3_14b_humanevalplus_hierarchical.txt` | 84.76 / 86.6 | **1512.3 / 1512** | Table 2, 14B, HumanEval+, hierarchical |

Two things they pin down that the paper does not state:

1. The MBPP+ log has `latent_steps: 0` in all 1134 trace entries — the latent
   channel was **off** for that published cell.
2. The paper's Token column counts **generated** text only. Prompt/prefill is
   785 (MBPP+) and 1183 (HumanEval+) tokens per problem, and LatentMAS pays it
   four times over.

Both logs print `method: muscle` and carry no `latent_steps` field in their arg
namespace, so they came from an **earlier code revision** than the released
`run.py`. Full analysis in `repro/README.md`.
