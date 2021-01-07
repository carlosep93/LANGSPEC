#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=../logs/unsup-train-wmt20-taen.log


FAIRSEQ_DIR="/home/usuaris/veu/cescola/interlingua-nodistance"


stdbuf -i0 -e0 -o0 python $FAIRSEQ_DIR/unsup_train.py $FAIRSEQ_DIR/data-bin/wmt20-mono/ \
 --arch unsupervised_multilingual_transformer --save-dir ../add-taen-mono --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 2500 --update-freq 16  --save-interval-updates 10000  --task unsupervised_multilingual_translation   --keys ta-en --pivotkeys en-de --newkey en-ta  --prev-enc-model ../add-taen-basic-tied/ --prev-dec-model ../add-enta-basic-tied/ --prev-pivot-model  ../europarl-basic-tied/ --reset-optimizer --lang-pairs en-ta --restore-file checkpoint_best.pt


