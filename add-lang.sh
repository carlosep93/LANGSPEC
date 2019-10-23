#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=add-lang-big3-ruen-ft.log



stdbuf -i0 -e0 -o0 python add_lang.py  data-bin/europarl/ \
 --arch multilingual_transformer_big --save-dir checkpoints/add-lang-big3-ruen-ft --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler reduce_lr_on_plateau --lr 0.001 --min-lr 1e-09 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 2500 --update-freq 16  --save-interval-updates 30000  --task multilingual_translation   --key de-en --newkey ru-en --reuse decoder --prev-model   checkpoints/europarl-big3 --reset-optimizer --lang-pairs ru-en --restore-file checkpoint_best.pt --finetune True



