#!/bin/bash
# Submit a padding probe, to measure the right-padding artifact (finding 5) before spending GPU-hours.
#
# Qwen3's tokenizer_config.json sets no padding_side, so Hugging Face defaults to right padding,
# and models.py does not override it. Under right padding, every sequence shorter than the
# batch maximum continues from a pad token and not from its last real prompt token.
#
# The failure is silent. generate_text_batch decodes with skip_special_tokens=True, so the pads
# never appear in the output text.
#
# Do not fix this by setting padding_side="left" alone. generate_text_batch slices
#   sequences[idx, prompt_lengths[idx]:]
# where prompt_lengths = attention_mask.sum(dim=1). That slice is correct only under right
# padding. Under left padding, generation starts at input_ids.shape[1] for every row, so the
# slice would return the end of the prompt and not the generated text.
#
# The probe runs the same 30 samples with the same seed at bs=1 (no padding) and at bs=20
# (heavy padding). A large gap means that batched numbers are not reliable, and every arm must
# run at bs=1.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-4B}
TASK=${TASK:-gsm8k}
NSAMP=${NSAMP:-30}
LSTEPS=${LSTEPS:-10}

for BS in 1 20; do
  sbatch --partition=free-gpu --gres=gpu:A100:1 --time=3:00:00 \
    --export=ALL,MODEL=$MODEL,TASK=$TASK,METHOD=latent_mas,PROMPT=sequential,\
LSTEPS=$LSTEPS,NSAMP=$NSAMP,BS=$BS,MAXNEW=4096,THINK=1 \
    yajat/run_latentmas.sbatch
done
squeue -u "$USER" -o "%.10i %.12P %.26j %.8T %.10M %R"
