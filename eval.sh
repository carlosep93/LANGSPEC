#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=eval-fairseq.log


stdbuf -i0 -e0 -o0 python ../fairseq/generate.py  data-bin/wmt17.tokenized.en-tr \
    --path checkpoints/trans-entr/checkpoint_best.pt \
    --batch-size 128 --beam 5
