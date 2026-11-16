#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train_transformer
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/train.out
#BSUB -e hpc/logs/train.err

# Submit your training run to a GPU queue:   bsub < hpc/train.sh
# Run this from proj root

set -eu

cd "src/protgen/models"
# source hpc/env.sh   # uv on the PATH, caches off the home quota
echo "starting"
python3 ./1_2_Transformer_train.py
