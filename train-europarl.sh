#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=train-europarl-nograd.log



stdbuf -i0 -e0 -o0 python train.py  data-bin/europarl/ \
 --arch interlingua_transformer --save-dir checkpoints/europarl-nograd --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 3584 --update-freq 16  --save-interval-updates 30000  --task interlingua_nodistance_translation   --lang-pairs en-de,fr-de,es-de,de-en,fr-en,es-en,en-fr,es-fr,de-fr,de-es,en-es,fr-es --freeze-schedule n-f,n-n,f-n,f-n,n-f,n-n,f-n,n-f,n-n,n-f,n-n,f-n 

