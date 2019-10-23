#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=train-asr-callhome.log



#stdbuf -i0 -e0 -o0 python train.py  data-bin/callhome/ \
# --arch speech_transformer --save-dir checkpoints/asr-callhome --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 10.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 1.0 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 200 --update-freq 16  --save-interval-updates 30000 --task speech_translation --target-lang es --source-lang scp --raw-text --kernels 10 14 18 --tds-layers 2 3 6 --input-channels 80 --decoder-embed-dim 512 --W 80 --F 27 --T 100 --p 1.0 --mf 1 --mt 1 --max-source-positions 1000 --mask-speech False 

#stdbuf -i0 -e0 -o0 python train.py  data-bin/callhome/ \
# --arch speechconvtransformer  --save-dir checkpoints/asr-callhome-fbk --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 350 --update-freq 16  --save-interval-updates 30000 --task speech_translation --target-lang es --source-lang scp --raw-text --kernels 10 14 18 --tds-layers 2 3 6 --input-channels 40 --decoder-embed-dim 512 --W 80 --F 27 --T 100 --p 1.0 --mf 1 --mt 1 --max-source-positions 1000 --mask-speech False --distance-penalty log 

stdbuf -i0 -e0 -o0 python train.py  data-bin/callhome/ \
     --arch vggtransformer_base --optimizer adadelta --clip-norm 10.0  --lr 1.0  --mask-speech False --adadelta-eps 1e-8 --adadelta-rho 0.95 --task speech_translation --mask-speech False --target-lang es --source-lang scp --raw-text --relu-dropout 0.1 --task speech_translation --lr 1.0 --clip-norm 10.0  --max-tokens 200 --update-freq 8 --save-dir checkpoints/trans-speech --criterion cross_entropy_acc
