#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=generate-int-esen.log


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="es"
TGT="en"
DEST_DIR="data-bin/wmt13.tokenized.32k.en-es/"
CP_DIR="checkpoints/esen-interlingua-finetuning"
CP="checkpoint_best.pt"

stdbuf -i0 -e0 -o0 python generate.py $DEST_DIR --path $CP_DIR/$CP \
    --beam 5 --batch-size 5 --source-lang ${SRC} --target-lang ${TGT} --task dist_interlingua_translation --remove-bpe

