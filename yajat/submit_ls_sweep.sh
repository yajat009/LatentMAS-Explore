#!/bin/bash
# Submit the latent_steps sweep, which yajat/README.md calls the most informative experiment.
# Earlier runs covered only ls=0 and ls=10 (ls = latent steps), and accuracy falls from 0.756 to
# 0.378 between them. The sweep shows whether the fall is monotonic (the latent channel harms
# accuracy at 4B) or a cliff (something breaks at a specific length).
#
# REALIGN=1 adds --latent_space_realign, which no earlier run used.
# Without that flag, models.py sets the output-to-input map to the identity. The model then reads
# an output-space vector as an input embedding. This is the main candidate explanation for the
# collapse at ls>0.
#
# The batch size is 1, because the right-padding artifact costs about 18 points at bs=15 and would
# mix with the effect of latent_steps. This sweep measures the channel, not the padding.
#
# Every job runs on L40S, because every earlier arm ran on L40S.
# free-gpu32 allows at most 4 GPUs, so the jobs queue and finish a few at a time.
#
# Usage:  ./yajat/submit_ls_sweep.sh          # both arms
#         REALIGN_ONLY=1 ./yajat/submit_ls_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

LS_VALUES=${LS_VALUES:-"0 1 2 5 10 20 40"}
NSAMP=${NSAMP:-45}
BS=${BS:-1}
TASK=${TASK:-mbppplus}
MODEL=${MODEL:-Qwen/Qwen3-4B}
JOBTIME=${JOBTIME:-3:00:00}
JOBMEM=${JOBMEM:-32G}
GPUMODEL=${GPUMODEL:-L40S}
PART=${PART:-free-gpu32}

REALIGNS="0 1"
[ "${REALIGN_ONLY:-0}" = "1" ] && REALIGNS="1"
[ "${NOREALIGN_ONLY:-0}" = "1" ] && REALIGNS="0"

for RA in $REALIGNS; do
  for LS in $LS_VALUES; do
    SUF=""; [ "$RA" = "1" ] && SUF="_realign"
    TAG="latent_mas_$(basename "$MODEL")_${TASK}_sequential_ls${LS}_bs${BS}_think1${SUF}"
    CKPT="yajat/results/${TAG}.ckpt.jsonl"
    have=0; [ -f "$CKPT" ] && have=$(wc -l < "$CKPT")
    if [ "$have" -ge "$NSAMP" ]; then
      echo "skip  ls=$LS realign=$RA -- already have $have/$NSAMP"
      continue
    fi
    echo "submit ls=$LS realign=$RA (have $have/$NSAMP)"
    sbatch --partition="$PART" --gres=gpu:${GPUMODEL}:1 \
      --mem="$JOBMEM" --time="$JOBTIME" \
      --job-name="lsweep-ls${LS}ra${RA}" \
      --export=ALL,METHOD=latent_mas,MODEL=$MODEL,TASK=$TASK,PROMPT=sequential,\
LSTEPS=$LS,NSAMP=$NSAMP,BS=$BS,MAXNEW=4096,REALIGN=$RA,THINK=1,\
GPUMODEL=$GPUMODEL,JOBMEM=$JOBMEM,JOBTIME=$JOBTIME,CHAIN=1,CHAIN_N=0,CHAIN_MAX=20 \
      yajat/run_latentmas.sbatch
  done
done
