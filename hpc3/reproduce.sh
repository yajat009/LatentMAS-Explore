#!/bin/bash
# Reproduce the paper's Single, TextMAS and LatentMAS numbers with the released run.py.
# LatentMAS runs use latent steps > 0, so the method is on. They use batch size 1, so padding cannot leak.
#
# Each configuration writes one log: results/repro/<model>_<task>_<method>_<layout>[_m<steps>][_realign].log
# The script skips a configuration whose log already ends with the JSON summary.
# You can therefore run it again after a crash or a preemption, and it redoes only what is missing.
#
# Usage (from the repo root, on a GPU node or inside a batch job):
#   MODELS="Qwen/Qwen3-4B" TASKS="mbppplus" bash hpc3/reproduce.sh           # run here, one after another
#   MODE=list bash hpc3/reproduce.sh                                           # only print the commands
#   MODE=submit SBATCH_ARGS="-p gpu -A EECS195221_CLASS_GPU --gres=gpu:A30:1" bash hpc3/reproduce.sh
#                                                                              # one Slurm job per configuration
# Settings (environment variables, defaults in brackets):
#   MODELS        [Qwen/Qwen3-4B Qwen/Qwen3-8B]
#   TASKS         [mbppplus gsm8k humanevalplus medqa arc_challenge arc_easy]
#   LAYOUTS       [sequential hierarchical]
#   STEPS         [20 40]           latent steps to try for LatentMAS
#   MAX_NEW       [4096]            max new tokens (the shipped example log used 4096)
#   TEXT_BS       [8]               batch size for baseline and TextMAS (no shared cache, so batching is safe)
#   MAX_SAMPLES   [-1]              -1 = full test set
#   OUT           [results/repro]   folder for the logs
set -u
MODELS=${MODELS:-"Qwen/Qwen3-4B Qwen/Qwen3-8B"}
TASKS=${TASKS:-"mbppplus gsm8k humanevalplus medqa arc_challenge arc_easy"}
LAYOUTS=${LAYOUTS:-"sequential hierarchical"}
STEPS=${STEPS:-"20 40"}
MAX_NEW=${MAX_NEW:-4096}
TEXT_BS=${TEXT_BS:-8}
MAX_SAMPLES=${MAX_SAMPLES:--1}
MODE=${MODE:-run}
SBATCH_ARGS=${SBATCH_ARGS:-}
OUT=${OUT:-results/repro}
mkdir -p "$OUT" logs

log_is_finished() { [ -f "$1" ] && tail -n 3 "$1" | grep -q '"accuracy"'; }

launch_run() {  # $1 = log file, rest = run.py arguments
    local log=$1; shift
    if log_is_finished "$log"; then echo "skip (done): $log"; return; fi
    local cmd="python -u run.py $* --max_samples $MAX_SAMPLES --max_new_tokens $MAX_NEW --checkpoint ${log%.log}.ckpt.jsonl"
    case $MODE in
        list)   echo "$cmd > $log" ;;
        submit) sbatch $SBATCH_ARGS -J "$(basename "$log" .log)" -o "logs/%x.%j.out" --wrap "$cmd >> $log 2>&1" ;;
        *)      echo "=== $log"; $cmd >> "$log" 2>&1 || echo "FAILED (exit $?): $log  -- see the end of the log" ;;
    esac
}

for model in $MODELS; do
    model_tag=$(basename "$model")
    for task in $TASKS; do
        # The single agent has no layout.
        launch_run "$OUT/${model_tag}_${task}_baseline.log" --method baseline --model_name "$model" --task "$task" \
            --generate_bs "$TEXT_BS"
        for layout in $LAYOUTS; do
            launch_run "$OUT/${model_tag}_${task}_text_mas_${layout}.log" --method text_mas --model_name "$model" \
                --task "$task" --prompt "$layout" --generate_bs "$TEXT_BS"
            for steps in $STEPS; do
                # generate_bs is 1 because batched latent runs leak padding into the shared cache.
                launch_run "$OUT/${model_tag}_${task}_latent_mas_${layout}_m${steps}.log" --method latent_mas \
                    --model_name "$model" --task "$task" --prompt "$layout" --latent_steps "$steps" --think \
                    --generate_bs 1
                launch_run "$OUT/${model_tag}_${task}_latent_mas_${layout}_m${steps}_realign.log" --method latent_mas \
                    --model_name "$model" --task "$task" --prompt "$layout" --latent_steps "$steps" --think \
                    --generate_bs 1 --latent_space_realign
            done
        done
    done
done
