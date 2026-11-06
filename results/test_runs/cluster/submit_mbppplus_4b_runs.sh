#!/bin/bash
# Submit the three-arm MBPP+ comparison: Qwen3-4B, bs=15, think=1, max_new_tokens=4096.
#
# The partition is free-gpu32, because this account cannot use the billed gpu and gpu32
# partitions. slurm_job_submit rejects them with "without specifying a GPU Account", and
# `sacctmgr show assoc user=ynagaraj` lists only the plain `ynagaraj` account.
# Preemption is therefore unavoidable. run.py --checkpoint and #SBATCH --requeue let a run
# survive it. Jobs 55644121 and 55644122 lost 2.5 GPU-hours to preemption before that.
#
# All arms run on gpu:L40S, so that wall-clock times are comparable. The paper's speed claim
# is half of its thesis, so the arms must not use different GPU models.
set -euo pipefail
cd "$(dirname "$0")/../../.."

COMMON=(--partition=free-gpu32 --gres=gpu:L40S:1 --mem=32G --time=12:00:00)
BASE="MODEL=Qwen/Qwen3-4B,TASK=mbppplus,PROMPT=sequential,NSAMP=-1,BS=15"

# baseline: a single agent, which is the arm that LatentMAS must beat on cost.
sbatch "${COMMON[@]}" --export=ALL,$BASE,METHOD=baseline,LSTEPS=0    results/test_runs/cluster/run_one_configuration.sbatch
# latent_mas ls=0: the configuration of the released logs. The Judger gets past_kv=None, so no channel exists.
sbatch "${COMMON[@]}" --export=ALL,$BASE,METHOD=latent_mas,LSTEPS=0  results/test_runs/cluster/run_one_configuration.sbatch
# latent_mas ls=10: the latent channel is on.
sbatch "${COMMON[@]}" --export=ALL,$BASE,METHOD=latent_mas,LSTEPS=10 results/test_runs/cluster/run_one_configuration.sbatch
