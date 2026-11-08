#!/bin/bash
set -eo pipefail
module load anaconda3/2023.09-0-biybti3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/genimage"
set -u
cd "/scratch/$USER/AI-Image-Detection"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python -u "$2/phase2.py" "$1" "$2"
