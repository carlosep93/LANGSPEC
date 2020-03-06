#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=multinli-baseline-tied.log



#stdbuf -i0 -e0 -o0 python train.py  data-bin/multinli/ \
# --task nli --arch nli --optimizer adam --clip-norm 0.0 --lr-scheduler reduce_lr_on_plateau  --lr 0.0001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0   --criterion classification_loss --save-dir checkpoints/multinli-basic-tied --raw-text --enc-path /scratch/carlos/europarl-basic-tied/checkpoint_best.pt --enc-key en-de --ref en --hyp en --class-hidden-size 128 --class-dropout 0.1 


stdbuf -i0 -e0 -o0 python train.py  /home/usuaris/veu/cescola/fairseq/data-bin/multinli-joint-no-ru/ \
 --task nli --arch nli --optimizer adam --clip-norm 0.0 --lr-scheduler reduce_lr_on_plateau --lr 0.0001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0   --criterion classification_loss --save-dir checkpoints/multinli-baseline-tied --raw-text --enc-path /scratch/carlos/baseline-share-embeddings/checkpoint_best.pt --ref src --hyp src --class-hidden-size 128 --class-dropout 0.1
