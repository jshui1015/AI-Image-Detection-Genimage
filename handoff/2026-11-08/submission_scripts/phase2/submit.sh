#!/bin/bash
set -euo pipefail
cd "/scratch/$USER/AI-Image-Detection"
TASK_DIR="$PWD/outputs/phase2-$(date +%Y%m%d-%H%M%S)"
mkdir "$TASK_DIR"
cp scripts/local/phase2/phase2.py scripts/local/phase2/stage.sh "$TASK_DIR/"
PREP=$(sbatch --parsable --account=mscaiehpc42026 --partition=amd --nodes=1 --ntasks=1 --cpus-per-task=2 --time=01:00:00 --job-name=genimage-prepare --output="$TASK_DIR/prepare-%j.log" "$TASK_DIR/stage.sh" prepare "$TASK_DIR")
echo "CPU preparation: $PREP"
GPU=$(sbatch --parsable --dependency="afterok:${PREP%%;*}" --account=mscaiehpc42026 --partition=gpu-l20 --nodes=1 --gpus-per-node=1 --time=02:00:00 --job-name=genimage-phase2 --output="$TASK_DIR/gpu-%j.log" "$TASK_DIR/stage.sh" gpu "$TASK_DIR")
echo "GPU job: $GPU"
POST=$(sbatch --parsable --dependency="afterok:${GPU%%;*}" --account=mscaiehpc42026 --partition=amd --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:30:00 --job-name=genimage-analysis --output="$TASK_DIR/analyze-%j.log" "$TASK_DIR/stage.sh" analyze "$TASK_DIR")
printf '%s\n' "prepare=$PREP" "gpu=$GPU" "analysis=$POST" "output=$TASK_DIR" | tee "$TASK_DIR/jobs.txt"
