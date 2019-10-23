#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=logs-add-big3/generate-add-csfr-big3.log


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="cs"
TGT="en"
DEST_DIR="data-bin/europarl/"
CP_DIR="checkpoints/add-lang-big3-csen"
CP="checkpoint_best.pt"

#CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python generate-add-lang.py $DEST_DIR --path $CP_DIR/$CP \
#    --beam 5 --batch-size 5 --source-lang ${SRC} --target-lang ${TGT} --task multilingual_translation \
#    --enc-model $CP_DIR/$CP --enc-key ru-en \
#    --dec-model checkpoints/europarl/$CP --dec-key es-en \
#    --newkey ru-en --newarch multilingual_transformer_iwslt_de_en \
#    --newtask multilingual_translation --remove-bpe 

#CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python generate-add-lang.py $DEST_DIR --path $CP_DIR/$CP \
#    --beam 5 --batch-size 5 --source-lang ${SRC} --target-lang ${TGT} --task multilingual_translation \
#    --enc-model checkpoints/europarl/$CP  --enc-key en-es \
#    --dec-model $CP_DIR/$CP --dec-key en-ru \
#    --newkey en-ru --newarch multilingual_transformer_iwslt_de_en \
#    --newtask multilingual_translation --remove-bpe 

CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python generate-add-lang.py $DEST_DIR --path $CP_DIR/$CP \
    --beam 5 --batch-size 5 --source-lang ${SRC} --target-lang ${TGT} --task multilingual_translation \
    --enc-model $CP_DIR/$CP   --enc-key cs-en \
    --dec-model $CP_DIR/$CP  --dec-key cs-en \
    --newkey cs-en --newarch multilingual_transformer_big \
    --newtask multilingual_translation --remove-bpe 





