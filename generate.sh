#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH -w veuc08                                                            
#SBATCH --output=generate-enfr-europarl-nograd.log


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="en"
TGT="fr"
DEST_DIR="data-bin/europarl/"
CP_DIR="checkpoints/europarl"
CP="checkpoint_best.pt"

CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python generate.py $DEST_DIR --path $CP_DIR/$CP \
    --beam 5 --batch-size 5 --source-lang ${SRC} --target-lang ${TGT} --task interlingua_nodistance_translation --freeze-schedule n-n  --remove-bpe

