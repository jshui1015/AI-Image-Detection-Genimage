#!/bin/bash -l
#SBATCH --job-name=resolution
#SBATCH --account=mscaiehpc42026
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/resolution-%A_%a.out
#SBATCH --error=logs/resolution-%A_%a.err
set -eo pipefail
module load anaconda3/2023.09-0-biybti3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
models=(resnet-20261031 v2-20261031 v2-20261101 v2-20261102)
python scripts/evaluate_resolution_control.py --model "${models[$SLURM_ARRAY_TASK_ID]}" --output-dir "outputs/resolution_control/$SLURM_ARRAY_JOB_ID" --threads 6 --workers 2
