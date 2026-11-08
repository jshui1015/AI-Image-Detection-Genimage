#!/bin/bash
set -eo pipefail
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "/scratch/$USER/AI-Image-Detection"
# New directory for every run; never overwrite past experimental outputs.
TASK_DIR=$(mktemp -d "$PWD/outputs/followup-20261108-XXXXXX")
printf 'Run directory: %s\n' "$TASK_DIR"
cp "$SOURCE_DIR/"*.py "$SOURCE_DIR/"*.json "$SOURCE_DIR/"*.sh "$SOURCE_DIR/"*.md "$TASK_DIR/"
for name in train.py clip_detector.py evaluate.py evaluate_robustness_holdout.py; do
  test -f "$name"
done
test -f outputs/cifake-external-20261108-154531/status.json
test -f outputs/phase2-20261108-144532/status.json
# Prevent a stale module path or missing dependency from consuming GPU queue time.
module load anaconda3/2023.09-0-biybti3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/genimage"
set -u
python - "$TASK_DIR" <<'PY'
import sys,json,hashlib,importlib.util
from pathlib import Path
for name in ['numpy','torch','torchvision','PIL','huggingface_hub','transformers']:
 assert importlib.util.find_spec(name),('Missing package',name)
p=Path(sys.argv[1])
for name,h in json.loads((p/'source_hashes.json').read_text()).items():
 assert hashlib.sha256((Path(name).read_text().rstrip()+'\n').encode()).hexdigest()==h,('Source changed',name)
print('PREFLIGHT OK')
PY
CI=$(sbatch --parsable --account=mscaiehpc42026 --partition=amd --nodes=1 --ntasks=1 --cpus-per-task=2 --time=02:00:00 --job-name=followup-ci --output="$TASK_DIR/ci-%j.log" "$TASK_DIR/stage.sh" confidence "$TASK_DIR")
printf '%s\n' "ci=$CI" "output=$TASK_DIR" > "$TASK_DIR/jobs.txt"
PREP=$(sbatch --parsable --account=mscaiehpc42026 --partition=amd --nodes=1 --ntasks=1 --cpus-per-task=2 --time=04:00:00 --job-name=followup-data --output="$TASK_DIR/prepare-%j.log" "$TASK_DIR/stage.sh" prepare "$TASK_DIR")
printf '%s\n' "prepare=$PREP" >> "$TASK_DIR/jobs.txt"
GPU=$(sbatch --parsable --dependency="afterok:${PREP%%;*}" --kill-on-invalid-dep=yes --account=mscaiehpc42026 --partition=gpu-l20 --nodes=1 --gpus-per-node=1 --time=01:00:00 --job-name=followup-nine --output="$TASK_DIR/gpu-%j.log" "$TASK_DIR/stage.sh" gpu "$TASK_DIR")
printf '%s\n' "gpu=$GPU" >> "$TASK_DIR/jobs.txt"
POST=$(sbatch --parsable --dependency="afterok:${GPU%%;*}:${CI%%;*}" --kill-on-invalid-dep=yes --account=mscaiehpc42026 --partition=amd --nodes=1 --ntasks=1 --cpus-per-task=2 --time=02:00:00 --job-name=followup-report --output="$TASK_DIR/analyze-%j.log" "$TASK_DIR/stage.sh" analyze "$TASK_DIR")
printf '%s\n' "analysis=$POST" >> "$TASK_DIR/jobs.txt"
mkdir -p "$HOME/genimage-setup"
printf '%s\n' "$TASK_DIR" > "$HOME/genimage-setup/followup-latest.txt"
cat "$TASK_DIR/jobs.txt"
