#!/bin/bash
# Run any command on one free L40S GPU on HPC3, so that a long job does not need its own batch script.
# Submit from the repo folder, for example ~/LatentMAS-Explore:   sbatch -J <name> results/test_runs/cluster/run_on_free_l40s.sh <command...>
#SBATCH -p free-gpu32               # free, preemptible partition
#SBATCH --gres=gpu:L40S:1           # one 48 GB L40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G                   # 8 GB per core, under the 9 GB cap; enough to load Qwen3-14B
#SBATCH -t 1-00:00:00               # 1 day. The free partition allows up to 3 days.
#SBATCH -o logs/%x.%j.out           # logs/ must exist before you submit
#SBATCH --requeue

module purge
module load miniconda3/23.5.2
eval "$(conda shell.bash hook)"
conda activate latentmas

export HF_HOME=/pub/$USER/hf_cache
cd "$SLURM_SUBMIT_DIR"            # the folder you ran sbatch from

nvidia-smi
echo "Running: $*"
"$@"
