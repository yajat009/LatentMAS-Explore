#!/bin/bash
#SBATCH -p gpu
#SBATCH -A EECS195221_CLASS_GPU
#SBATCH --gres=gpu:A30:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x.%j.out
# Runs one audit test as its own job. The first argument names the run. The rest is the
# command to run. Checkpoints keep finished questions if the time limit ends the job.
#
# Usage, from the repo root:  sbatch -J <name> results/test_runs/cluster/run_audit_test.sh <name> <command...>
set -u
export PATH=/pub/$USER/envs/latentmas/bin:$PATH
export PYTHONNOUSERSITE=1                 # ~/.local otherwise leaks into this env
export HF_HOME=/pub/$USER/hf_cache
export HF_DATASETS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
name=$1; shift
echo "start $(date +%T) on $(hostname): $*"
"$@" >> "results/raw/audit_tests/$name.log" 2>&1
status=$?
echo "end $(date +%T) status $status"
