#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=train-tren-comparative.log



stdbuf -i0 -e0 -o0 python train.py  data-bin/wmt17.tokenized.16k.tr-en/ \
 --arch interlingua_transformer --save-dir checkpoints/tren-comparative --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0 --criterion dist_label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 1000 --update-freq 24  --save-interval-updates 5000  --task dist_interlingua_translation   --lang-pairs en-tr,tr-en


