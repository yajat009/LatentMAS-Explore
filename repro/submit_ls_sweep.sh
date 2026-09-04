#!/bin/bash
# latent_steps sweep -- the experiment repro/README.md calls the most informative
# one available. Only ls=0 and ls=10 had been run, and accuracy falls 0.756 -> 0.378
# between them. The sweep says whether that is monotonic (the latent channel is
# actively harmful at 4B) or a cliff (something breaks at a specific length).
#
# REALIGN=1 crosses it with --latent_space_realign, which had never been run at all.
# Without that flag models.py sets the output->input map to identity, so an
# output-space vector is fed back into an input-embedding slot; it is the leading
# candidate explanation for the ls>0 collapse.
#
# bs=1 deliberately: the right-padding artifact costs ~18 points at bs=15 and would
# otherwise be confounded with latent_steps. This measures the channel, not the pad.
#
# Every job is pinned to L40S because every previously-measured arm was on L40S.
# free-gpu32 caps at 4 GPUs, so these queue and drain a few at a time.
#
# Usage:  ./repro/submit_ls_sweep.sh          # both arms
#         REALIGN_ONLY=1 ./repro/submit_ls_sweep.sh
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
    CKPT="repro/results/${TAG}.ckpt.jsonl"
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
      repro/run_latentmas.sbatch
  done
done
