#!/bin/bash
set -eo pipefail
module load anaconda3/2023.09-0-biybti3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/genimage"
set -u
cd "/scratch/$USER/AI-Image-Detection"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
if [[ "$1" == "prepare" ]]; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
else
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
python -m pip freeze > "$2/environment-$1.txt"
python -u "$2/followup.py" "$1" "$2"
