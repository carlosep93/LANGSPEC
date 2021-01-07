#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=wmt20-logs/generate-enta-mono.log


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="en"
TGT="ta"
DEST_DIR="data-bin/wmt20"
DEC_CP_DIR="/scratch/carlos/add-taen-mono-250-small"
ENC_CP_DIR="/scratch/carlos/europarl-basic-tied"
CP="checkpoint_best.pt"

# stdbuf -i0 -e0 -o0 python generate-add-lang.py $DEST_DIR --path $CP_DIR/$CP \
#    --beam 5 --batch-size 5 --source-lang ${SRC} --target-lang ${TGT} --task multilingual_translation \
#    --enc-model $CP_DIR/$CP --enc-key fr-en \
#    --dec-model checkpoints/esen-interlingua-finetuning/$CP --dec-key es-es \
#    --newkey fr-es --newarch multilingual_transformer_iwslt_de_en \
#    --newtask multilingual_translation --remove-bpe 

stdbuf -i0 -e0 -o0 python generate-add-lang.py $DEST_DIR --path $CP_DIR/$CP \
    --beam 5 --batch-size 5 --source-lang ${SRC} --target-lang ${TGT} --task multilingual_translation \
    --enc-model $ENC_CP_DIR/$CP   --enc-key en-es \
    --dec-model $DEC_CP_DIR/$CP  --dec-key ta-en \
    --newkey en-ta  --newarch multilingual_transformer \
    --newtask multilingual_translation --remove-bpe 


