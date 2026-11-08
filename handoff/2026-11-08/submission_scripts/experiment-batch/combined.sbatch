#!/bin/bash
#SBATCH --job-name=genimage-combined
#SBATCH --account=mscaiehpc42026
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/combined-%j.log

set -eo pipefail
module load anaconda3/2023.09-0-biybti3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/genimage"
set -u
cd "/scratch/$USER/AI-Image-Detection"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python -u scripts/local/completion_batch/run_batch.py run
