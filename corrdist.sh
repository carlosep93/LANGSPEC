#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:2
#SBATCH --mem=60G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=train-esen-interlingua-corrdir.log


mkdir -p checkpoints/esen-interlingua

stdbuf -i0 -e0 -o0 python train.py  data-bin/wmt13.tokenized.32k.en-es/ \
 --arch interlingua_transformer --save-dir checkpoints/esen-interlingua-corrdir --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0 --criterion interlingua_label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 1500 --update-freq 16  --save-interval-updates 5000  --task interlingua_translation   --lang-pairs es-en,en-es --corrdist True --lbd 1


