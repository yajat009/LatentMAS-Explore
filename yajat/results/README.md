# `yajat/results/` — run outputs

Mostly gitignored: raw stdout logs are multi-MB (`run.py` prints every full
prompt) and the `--checkpoint` JSONL carries the entire agent trace per problem.
Only the distilled summaries are tracked:

| tracked file | what it is |
|---|---|
| `arm_comparison.json` | `compare_arms.py` output — every arm on the common prefix they have all finished |
| `padding_probe.json` | the bs=1 vs bs=15 right-padding probe |
| `summary.json` | `analyze.py` output for this fork's runs |
| `reference_summary.json` | `analyze.py` output for the two upstream released logs |

Untracked, produced locally by a run:

- `<arm>.ckpt.jsonl` — `run.py --checkpoint`. One JSON record per problem,
  appended after every batch, replayed on startup as a prefix skip. This is the
  real data source: `compare_arms.py` reads these, not the stdout logs, because
  preemption means a *finished* log is rare and a partial one is normal.
- `<arm>_<jobid>.log` — teed stdout.

Arm naming: `{method}_{model}_{task}_{prompt}_ls{latent_steps}_bs{batch}_think{0,1}`.
Keyed on the config, not the job id, so a resubmitted job resumes the same work.
