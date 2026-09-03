# `repro/logs/` — Slurm and tooling logs

Gitignored by default (`latentmas-<jobid>.out/.err` from `run_latentmas.sbatch`,
env-build and asset-fetch logs). Three are tracked because they are evidence for
claims in `repro/README.md` rather than run noise:

| tracked file | what it shows |
|---|---|
| `trace_latent_ls0.log` | `trace_pipeline.py` at `--latent_steps 0`: judger inherits **0** KV positions — the 536 built by the first three agents are discarded |
| `trace_latent_ls10.log` | the same at `--latent_steps 10`: judger inherits 566 — the latent channel actually connected |
| `analyze_reference.log` | `analyze.py` reproducing the paper's Token column exactly from the two upstream released logs |
