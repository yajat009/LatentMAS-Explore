#!/bin/bash
#SBATCH -J latentmas-4h
#SBATCH -p gpu
#SBATCH -A EECS195221_CLASS_GPU
#SBATCH --gres=gpu:A30:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:40:00
#SBATCH -o logs/%x.%j.out
# Runs a reduced reproduction that fits a 4 GPU-hour budget, on subsets of the test sets.
#
# The runs go in priority order on one A30. Each run has its own time limit, so one slow
# run cannot use the time of the runs after it. A run that hits its limit keeps every
# finished question in its checkpoint, and its log then has no final JSON summary.
# Rerunning this script resumes each unfinished run and skips the finished ones.
#
# Usage, from the repo root:  mkdir -p logs && sbatch hpc3/run_4h_plan.sh
set -u

export PATH=/pub/$USER/envs/latentmas/bin:$PATH
export PYTHONNOUSERSITE=1                 # ~/.local otherwise leaks into this env
export HF_HOME=/pub/$USER/hf_cache
export HF_DATASETS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=results/repro_4h
mkdir -p "$OUT"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1

# Reuse the 45 questions already finished with these exact settings. The copy leaves
# the original checkpoint unchanged, and run.py resumes from it as a prefix skip.
SEED_CHECKPOINT=yajat/results/latent_mas_Qwen3-4B_mbppplus_sequential_ls20_bs1_think1.ckpt.jsonl
RESUMED_CHECKPOINT=$OUT/Qwen3-4B_mbppplus_latent_mas_m20.ckpt.jsonl
[ -f "$RESUMED_CHECKPOINT" ] || cp "$SEED_CHECKPOINT" "$RESUMED_CHECKPOINT"

run_arm() {  # $1 = run name, $2 = time limit, rest = run.py arguments
    local name=$1 limit=$2; shift 2
    local log=$OUT/$name.log
    if [ -f "$log" ] && tail -n 3 "$log" | grep -q '"accuracy"'; then
        echo "skip (done): $name"
        return
    fi
    echo "=== $(date +%T) start $name (limit $limit)"
    # Append, so a resumed run never overwrites the log of an earlier attempt.
    timeout "$limit" python -u run.py "$@" --max_new_tokens 4096 --checkpoint "$OUT/$name.ckpt.jsonl" >> "$log" 2>&1
    echo "=== $(date +%T) end $name (exit $?)"
}

# These strings stay unquoted below, so that the shell splits them into separate run.py arguments.
SEQ="--prompt sequential"
LATENT="--method latent_mas --latent_steps 20 --think --generate_bs 1 $SEQ"

run_arm Qwen3-4B_mbppplus_latent_mas_m20 40m $LATENT --model_name Qwen/Qwen3-4B --task mbppplus --max_samples 120

run_arm Qwen3-4B_gsm8k_baseline         12m --method baseline --generate_bs 8 --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 100
run_arm Qwen3-4B_gsm8k_text_mas         20m --method text_mas --generate_bs 8 $SEQ --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 100
run_arm Qwen3-4B_gsm8k_latent_mas_m20   30m $LATENT --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 100

run_arm Qwen3-8B_mbppplus_baseline         13m --method baseline --generate_bs 4 --model_name Qwen/Qwen3-8B --task mbppplus --max_samples 60
run_arm Qwen3-8B_mbppplus_latent_mas_m20   20m $LATENT --model_name Qwen/Qwen3-8B --task mbppplus --max_samples 60
run_arm Qwen3-8B_mbppplus_latent_mas_m20_realign 20m $LATENT --latent_space_realign --model_name Qwen/Qwen3-8B --task mbppplus --max_samples 60
