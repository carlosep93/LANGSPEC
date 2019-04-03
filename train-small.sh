#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=train-fairseq.log


mkdir -p checkpoints/trans-entr


stdbuf -i0 -e0 -o0 python train.py  ../entr-test/wmt17.tokenized.8k.tr-en \
 --arch interlingua_transformer --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0 \
 --criterion interlingua_label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 20 --update-freq 1 \
 --save-dir checkpoints/trans-entr --encoder-ffn-embed-dim 64 --encoder-layers 1 --encoder-attention-heads 1 \
 --decoder-layers 1 --decoder-attention-heads 1 --encoder-embed-dim 32 --task interlingua_translation \
 --lang-pairs en-tr

