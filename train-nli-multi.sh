#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=multinli-basic.log



stdbuf -i0 -e0 -o0 python train.py  data-bin/multinli/ \
 --task nli --arch nli --optimizer adam --clip-norm 0.0 --lr-scheduler reduce_lr_on_plateau  --lr 5e-06 --min-lr 1e-09 --dropout 0.0 --weight-decay 0.0   --criterion classification_loss --save-dir checkpoints/multinli-basic --raw-text --enc-path checkpoints/europarl-big-fr-no/checkpoint_best.pt --enc-key en-es --ref en --hyp en --class-hidden-size 128 --class-dropout 0.1 --max-tokens 500 


#stdbuf -i0 -e0 -o0 python train.py  /home/usuaris/veu/cescola/fairseq/data-bin/multinli-joint-no-ru/ \
# --task nli --arch nli_lstm --optimizer adam --clip-norm 0.0 --lr-scheduler fixed --lr 0.0003 --min-lr 1e-09 --dropout 0.0 --weight-decay 0.0   --criterion classification_loss --save-dir checkpoints/multinli-baseline-lstm --raw-text --enc-path /scratch/carlos/europarl-multi-lstm-4layers/checkpoint_best.pt --ref src --hyp src --class-hidden-size 384 --class-dropout 0.1 --encoder-layers 4  --valid-subset test
