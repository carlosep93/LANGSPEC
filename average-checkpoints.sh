#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=average-checkpoints.log


DEST_DIR="/scratch/carlos/add-taen-mono-250-indic-avg"
CP_DIR="/scratch/carlos/add-taen-mono-250-indic"

mkdir -p $DEST_DIR

stdbuf -i0 -e0 -o0 python scripts/average_checkpoints.py  --output  $DEST_DIR/checkpoint_avg.pt --input $CP_DIR --num-epoch-checkpoints 2

