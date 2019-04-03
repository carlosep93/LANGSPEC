#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=generate-add-fres.log


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="fr"
TGT="es"
DEST_DIR="data-bin/wmt13.tokenized.32k.fr-en/"
CP_DIR="checkpoints/add-lang-fren"
CP="checkpoint_best.pt"

stdbuf -i0 -e0 -o0 python generate-add-lang.py $DEST_DIR --path $CP_DIR/$CP \
    --beam 5 --batch-size 5 --source-lang ${SRC} --target-lang ${TGT} --task multilingual_translation \
    --enc-model $CP_DIR/$CP --enc-key fr-en \
    --dec-model checkpoints/esen-interlingua-finetuning/$CP --dec-key es-es \
    --newkey fr-es --newarch multilingual_transformer_iwslt_de_en \
    --newtask multilingual_translation --remove-bpe 



