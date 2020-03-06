#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=basic-tied-logs/generate-rufr-basic-tied-ru.log


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="ru"
TGT="fr"
DEST_DIR="data-bin/europarl/"
CP_DIR="/scratch/carlos/europarl-basic-tied-ru/"
CP="checkpoint_best.pt"

stdbuf -i0 -e0 -o0 python generate.py $DEST_DIR --path $CP_DIR/$CP \
    --beam 5 --batch-size 5 --source-lang ${SRC} --target-lang ${TGT} --task interlingua_nodistance_translation --freeze-schedule n-n  --remove-bpe

