#!/bin/bash
#SBATCH -p gpu
#SBATCH -A EECS195221_CLASS_GPU
#SBATCH --gres=gpu:A100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:14:00
#SBATCH -o logs/%x.%j.out
# Runs one README command as its own short job, so that the total GPU time stays under a fixed budget.
#
# The first argument names the run. The rest is the command. The time limit above is the
# budget guard: Slurm ends the job at the limit, and the checkpoint keeps every finished batch.
#
# Usage, from the repo root:  sbatch -J <name> hpc3/run_readme_repro.sh <name> <command...>
set -u
export PATH=/pub/$USER/envs/latentmas/bin:$PATH
export PYTHONNOUSERSITE=1                 # ~/.local otherwise leaks into this env
export HF_HOME=/pub/$USER/hf_cache
export HF_DATASETS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/readme_repro
mkdir -p "$OUT"
name=$1; shift
echo "start $(date +%T) on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1): $*"
"$@" >> "$OUT/$name.log" 2>&1
status=$?
echo "end $(date +%T) status $status"
