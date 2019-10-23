#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=add-lang-big3-csen2.log



stdbuf -i0 -e0 -o0 python add_lang.py  data-bin/europarl/ \
 --arch multilingual_transformer_big --save-dir checkpoints/add-lang-big3-csen2 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 2500 --update-freq 16  --save-interval-updates 30000  --task multilingual_translation   --key de-en --newkey cs-en --reuse decoder --prev-model   checkpoints/europarl-big3 --reset-optimizer --lang-pairs cs-en --restore-file checkpoint_best.pt


