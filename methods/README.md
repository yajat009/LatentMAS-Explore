# `methods/` — the three arms being compared (upstream code)

All three run the same four agents — Planner → Critic → Refiner → Judger
(`__init__.py:default_agents`). Only the handoff between them differs.

| file | what it does |
|---|---|
| `baseline.py` | One prompt, one generation. No agents. The single-agent control. |
| `text_mas.py` | Each agent generates **text**; it is appended to the next agent's prompt. The token-space multi-agent control, and the arm the paper's ×3.7 speedup is measured against. |
| `latent_mas.py` | The contribution. Planner/Critic/Refiner **emit no tokens** — their prompts are forward-passed onto a shared `past_key_values`, then `latent_steps` steps run in embedding space. The KV cache is the entire inter-agent channel; only the judger calls `generate()`. |

Two lines worth knowing before you run anything:

- `latent_mas.py` — `past_for_decoding = past_kv if self.latent_steps > 0 else None`.
  At `--latent_steps 0` (**the argparse default**) the judger gets `None`, the whole
  accumulated cache is discarded, and the method degenerates to a single-agent call
  preceded by three wasted forward passes. Always pass `--latent_steps` explicitly.
- `latent_only` / `sequential_info_only` are read here via
  `getattr(args, ..., False)` but **`run.py` never defines them**, so they are
  permanently False. They are the paper's ablations, currently unreachable.

Unmodified from upstream except for the vLLM import guard applied by
`repro/02_patch_blockers.sh`.
