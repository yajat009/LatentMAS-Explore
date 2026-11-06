#!/bin/bash
#SBATCH --job-name=latentmas
#SBATCH --account=ynagaraj
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=yajat/logs/%x-%j.out
#SBATCH --error=yajat/logs/%x-%j.err
# Slurm preempts jobs on free-* partitions. --requeue puts the job back in the queue, and
# --checkpoint (below) makes the restart resume and not repeat 3 hours of work.
# Append mode stops a requeued attempt from truncating the log of the previous attempt.
#SBATCH --requeue
#SBATCH --open-mode=append
#
# Usage:
#   sbatch --export=ALL,METHOD=latent_mas,MODEL=Qwen/Qwen3-14B,TASK=mbppplus,\
# PROMPT=sequential,LSTEPS=0,NSAMP=-1,BS=20 yajat/run_latentmas.sbatch
#
# GPU selection (models.py always uses bf16, so never use V100 or sm_70):
#   Qwen3-4B  (~8 GB)  : --partition=free-gpu   --gres=gpu:A30:1
#   Qwen3-8B  (~16 GB) : --partition=free-gpu   --gres=gpu:A30:1     (tight) or L40S
#   Qwen3-14B (~29 GB) : --partition=free-gpu32 --gres=gpu:L40S:1 | gpu:RTX6000:1
#                        or --partition=free-gpu --gres=gpu:A100:1
# free-* partitions are preemptible.

set -euo pipefail

METHOD=${METHOD:-latent_mas}
MODEL=${MODEL:-Qwen/Qwen3-14B}
TASK=${TASK:-mbppplus}
PROMPT=${PROMPT:-sequential}
LSTEPS=${LSTEPS:-0}
NSAMP=${NSAMP:--1}
BS=${BS:-15}
MAXNEW=${MAXNEW:-4096}
REALIGN=${REALIGN:-0}
# --think appends a literal <think> to the assistant turn. Both released reference logs
# used think:True, so this setting is on by default to match the paper.
THINK=${THINK:-1}
# Preemption on this cluster cancels the job and does not requeue it. The free-gpu32 L40S nodes
# (k54-06 and k54-07) are also in the billed gpu32 partition. A paying job evicts this job with
# "CANCELLED ... DUE to SIGNAL Terminated", so --requeue never runs. Four jobs died that way on
# 2026-08-28. This script therefore resubmits itself on TERM and on any non-zero exit.
# --checkpoint makes each new attempt resume where the last one stopped.
CHAIN=${CHAIN:-1}
CHAIN_N=${CHAIN_N:-0}
CHAIN_MAX=${CHAIN_MAX:-20}

export HF_HOME=/pub/$USER/hf
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}   # The cache still serves hits. With 1, any id mismatch is fatal.
export HF_DATASETS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
# ~/.local/lib/python3.10/site-packages leaks into every python3.10 environment on this
# account and shadows torch's CUDA libraries, so it must stay off sys.path.
export PYTHONNOUSERSITE=1
# latent_mas ran out of memory on a 24 GB A30 at bs=15, with 4.33 GiB reserved but unallocated.
# The Judger inherits the KV (key/value) cache of all earlier agents, so its peak memory is
# much higher than that of baseline and text_mas at the same batch size.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# The script calls the environment's interpreter directly, because `conda activate` is fragile
# here. Five jobs died at once with "EnvironmentLocationNotFound" while something rewrote the
# environment folder. An absolute path does not have this race.
PY_BIN=/pub/$USER/envs/latentmas/bin/python
[ -x "$PY_BIN" ] || { echo "FATAL: $PY_BIN missing"; exit 1; }
export PATH=/pub/$USER/envs/latentmas/bin:$PATH

cd "$SLURM_SUBMIT_DIR"
mkdir -p yajat/logs yajat/results

# REALIGN must be in the tag, because it is a different arm and needs its own checkpoint.
# Otherwise a REALIGN=1 job resumes from the REALIGN=0 checkpoint of the same ls and bs,
# and it reports a mixture of the two without a warning.
RA_SUFFIX=""
[ "$REALIGN" = "1" ] && RA_SUFFIX="_realign"
TAG="${METHOD}_$(basename "$MODEL")_${TASK}_${PROMPT}_ls${LSTEPS}_bs${BS}_think${THINK}${RA_SUFFIX}"
OUT="yajat/results/${TAG}_${SLURM_JOB_ID}.log"
# The checkpoint path depends on the configuration and not on the job id. A requeued attempt
# (which keeps its job id) and a job resubmitted by hand therefore both continue the work.
CKPT="yajat/results/${TAG}.ckpt.jsonl"

ckpt_lines() { [ -f "$CKPT" ] && wc -l < "$CKPT" || echo 0; }

resubmit() {
  local why="$1"
  if [ "$CHAIN" != "1" ]; then echo "[chain] $why; chaining disabled"; return; fi
  if [ "$CHAIN_N" -ge "$CHAIN_MAX" ]; then echo "[chain] $why; hit CHAIN_MAX=$CHAIN_MAX"; return; fi
  # An attempt that made no progress will crash again. The script chains only after progress,
  # because otherwise 20 attempts repeat the same traceback.
  if [ "$(ckpt_lines)" -le "$START_LINES" ]; then
    echo "[chain] $why; but checkpoint still at $START_LINES problems -- no progress, not chaining"
    return
  fi
  local next=$((CHAIN_N + 1))
  echo "[chain] $why; resubmitting as attempt $next/$CHAIN_MAX"
  sbatch --partition="$SLURM_JOB_PARTITION" --gres=gpu:${GPUMODEL}:1 \
    --mem="${JOBMEM}" --time="${JOBTIME}" \
    --export=ALL,METHOD=$METHOD,MODEL=$MODEL,TASK=$TASK,PROMPT=$PROMPT,LSTEPS=$LSTEPS,\
NSAMP=$NSAMP,BS=$BS,MAXNEW=$MAXNEW,REALIGN=$REALIGN,THINK=$THINK,\
CHAIN=$CHAIN,CHAIN_N=$next,CHAIN_MAX=$CHAIN_MAX,GPUMODEL=$GPUMODEL,JOBMEM=$JOBMEM,JOBTIME=$JOBTIME \
    yajat/run_latentmas.sbatch
}

# Every arm uses the same GPU model, because wall-clock time is half of the paper's claim.
# A comparison across L40S and A30 measures the cluster and not the method.
GPUMODEL=${GPUMODEL:-L40S}
JOBMEM=${JOBMEM:-32G}
JOBTIME=${JOBTIME:-12:00:00}

# Preemption arrives as SIGTERM to this script. The trap resubmits first, then exits.
trap 'resubmit "caught SIGTERM (preemption)"; exit 143' TERM

EXTRA=()
[ "$REALIGN" = "1" ] && EXTRA+=(--latent_space_realign)
[ "$THINK" = "1" ] && EXTRA+=(--think)

START_LINES=$(ckpt_lines)

nvidia-smi
echo "=== $TAG (attempt ${SLURM_RESTART_COUNT:-0}) ==="
if [ -f "$CKPT" ]; then echo "resuming from $(wc -l < "$CKPT") completed problems in $CKPT"; fi

set +e
"$PY_BIN" -u architecture/run.py \
  --method "$METHOD" \
  --model_name "$MODEL" \
  --task "$TASK" \
  --prompt "$PROMPT" \
  --latent_steps "$LSTEPS" \
  --max_samples "$NSAMP" \
  --max_new_tokens "$MAXNEW" \
  --generate_bs "$BS" \
  --device cuda:0 \
  --checkpoint "$CKPT" \
  "${EXTRA[@]}" 2>&1 | tee -a "$OUT"
rc=${PIPESTATUS[0]}
set -e

echo "--- final line ---"
tail -1 "$OUT"
if [ "$rc" -ne 0 ]; then
  resubmit "run.py exited $rc"
else
  echo "[chain] completed cleanly after $CHAIN_N resubmission(s)"
fi
exit "$rc"
