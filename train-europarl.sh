#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=50G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=train-europarl-big-fr-all.log



stdbuf -i0 -e0 -o0 python train.py  data-bin/europarl/ \
 --arch interlingua_transformer_big --save-dir checkpoints/europarl-big-fr-all --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 2000 --update-freq 16  --save-interval-updates 30000  --task interlingua_nodistance_translation   --lang-pairs de-en,de-es,de-fr,en-de,en-es,en-fr,es-de,es-en,es-fr,fr-de,fr-en,fr-es  --freeze-schedule n-n,f-n,n-f,n-n,n-f,f-n,n-f,f-n,n-n,f-n,n-f,n-n


