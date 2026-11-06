# Cluster scripts

These scripts run the tests on UCI's HPC3 cluster, which uses the Slurm job scheduler. Run them from the repo root.

| Script | What it does |
|---|---|
| `setup_environment.sh` | Builds the conda environment in `/pub/$USER/envs/latentmas` |
| `download_models_and_datasets.sh` | Downloads Qwen3-4B, 8B and 14B and the datasets into `$HF_HOME`. Run it on a login node. |
| `download_qwen3_4b_only.sh` | Downloads Qwen3-4B only, for a quick start |
| `run_one_configuration.sbatch` | Runs one `architecture/run.py` configuration, with checkpoints and automatic resubmission after preemption |
| `submit_method_comparison.sh` | Submits Single, TextMAS and LatentMAS for one task |
| `submit_mbppplus_4b_runs.sh` | Submits the Qwen3-4B MBPP+ runs |
| `submit_latent_steps_sweep.sh` | Submits the latent-steps sweep on the first 45 MBPP+ questions |
| `submit_padding_test.sh` | Submits batch size 1 against batch size 15 on the same questions |
| `reproduce_paper_tables.sh` | Runs every paper configuration, and skips finished ones. `MODE=list` only prints the commands. |
| `run_reduced_reproduction.sh` | Runs a subset of the reproduction that fits 4 GPU-hours |
| `run_audit_test.sh` | Runs one audit test as its own job |
| `run_on_free_l40s.sh` | Wraps any command in a job on one free L40S GPU |

## Cluster facts that affect runs

- **Accounts.** `EECS195221_CLASS_GPU` can use the `gpu` partition (A30, A100). The personal account `ynagaraj` can use only `free-gpu` and `free-gpu32`. Neither account can use `gpu32`.
- **Preemption.** A paying job cancels a job on a free partition. Slurm does not requeue it, so `--requeue` does not help. `run_one_configuration.sbatch` therefore resubmits itself, and it stops if the checkpoint did not grow.
- **GPUs.** `models.py` uses bf16, so do not use V100 or RTX6000 GPUs. Qwen3-4B and 8B fit on an A30 (24 GB). Qwen3-14B needs an L40S or A100.
- **Memory.** LatentMAS ran out of memory on an A30 at batch size 15, because the Judger holds the cache of all earlier agents. TextMAS ran out of memory at batch size 15 on Qwen3-4B on an L40S, so the TextMAS run used batch size 8.
- **Python.** `~/.local` leaks into every Python 3.10 environment on this account and breaks NCCL. Every script sets `PYTHONNOUSERSITE=1`.
- **Downloads.** Set `HF_HOME` under `/pub/$USER`, because the home quota is small.
- **Time.** `run_reduced_reproduction.sh` runs overran their `timeout` limits by 6 to 10 minutes. The cause is unknown.
