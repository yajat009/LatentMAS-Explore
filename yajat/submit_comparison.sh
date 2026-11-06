#!/bin/bash
# Submit all three arms on the same data, because the paper's claim is a comparison:
# the same accuracy with fewer tokens and less time. LatentMAS alone proves nothing.
#
# The defaults reproduce the hyperparameters of the released logs (think=1, max_new_tokens=4096,
# temperature 0.6, top_p 0.95, seed 42) on a GSM8K subset with Qwen3-4B.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-4B}
TASK=${TASK:-gsm8k}
PROMPT=${PROMPT:-sequential}
NSAMP=${NSAMP:-120}
BS=${BS:-15}
LSTEPS=${LSTEPS:-10}
PART=${PART:-free-gpu}
GRES=${GRES:-gpu:A30:1}

sub () {  # sub <method> <lsteps> <timelimit>
  sbatch --partition="$PART" --gres="$GRES" --time="$3" \
    --export=ALL,MODEL=$MODEL,TASK=$TASK,METHOD=$1,PROMPT=$PROMPT,LSTEPS=$2,NSAMP=$NSAMP,BS=$BS,MAXNEW=4096,THINK=1 \
    yajat/run_latentmas.sbatch
}

# latent_mas is the fastest arm, because only the Judger generates text.
sub latent_mas "$LSTEPS" 4:00:00
# baseline runs one generation and no agents.
sub baseline   0          4:00:00
# text_mas takes about 4 times the wall-clock time, because all four agents generate text.
sub text_mas   0          8:00:00

echo
squeue -u "$USER" -o "%.10i %.12P %.24j %.8T %.10M %.6D %R"
