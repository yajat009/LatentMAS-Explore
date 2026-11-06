#!/bin/bash
# Finishes the runs that the batch time limits cut short, on the interactive session's GPU.
# Each run resumes from its checkpoint, so only the missing questions are generated.
export PATH=/pub/ynagaraj/envs/latentmas/bin:$PATH PYTHONNOUSERSITE=1 HF_HOME=/pub/ynagaraj/hf_cache HF_DATASETS_CACHE=/pub/ynagaraj/hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=results/readme_repro
R="--task gsm8k --max_samples 100 --max_new_tokens 2048 --subset_seed 0"
while pgrep -u $USER -f 'run.py --method baseline --model_name Qwen/Qwen3-4B' >/dev/null; do sleep 5; done
echo "$(date +%T) 4B baseline finished: $(wc -l < $O/4b_gsm8k_baseline.ckpt.jsonl) questions"
python -u run.py --method latent_mas --model_name Qwen/Qwen3-4B --prompt sequential $R --checkpoint $O/4b_gsm8k_latent_mas.ckpt.jsonl >> $O/4b_gsm8k_latent_mas.log 2>&1
echo "$(date +%T) 4B latent_mas exit $?: $(wc -l < $O/4b_gsm8k_latent_mas.ckpt.jsonl) questions"
python -u run.py --method baseline --model_name Qwen/Qwen3-8B $R --checkpoint $O/8b_gsm8k_baseline.ckpt.jsonl >> $O/8b_gsm8k_baseline.log 2>&1
echo "$(date +%T) 8B baseline exit $?: $(wc -l < $O/8b_gsm8k_baseline.ckpt.jsonl) questions"
echo ALL_DONE
