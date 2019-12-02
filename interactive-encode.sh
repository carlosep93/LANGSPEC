#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=interactive-encode-acl-es.log

WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="es"
TGT="en"
DEST_DIR="data-bin/europarl"
CP_DIR="checkpoints/europarl-big3"
CP="checkpoint_best.pt"
PYTHON="/home/usuaris/veu/cescola/virtualenv-16.0.0/torch/bin/python"
FAIRSEQ_DIR="/home/usuaris/veu/cescola/fairseq"
INPUT_DATA="../europarl/en-es/test.es"
OUTPUT="encodings-acl-es.json"

mkdir tmp

cp $INPUT_DATA tmp/tmp.src

subword-nmt apply-bpe -c $DEST_DIR/codes/es.codes < tmp/tmp.src > tmp/tmp.bpe

CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points 500 \
     --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task interlingua_nodistance_translation \
    --enc-model $CP_DIR/$CP --enc-key ${SRC}-${TGT} \
    --dec-model $CP_DIR/$CP --dec-key ${SRC}-${TGT} \
    --newkey $SRC-$TGT --newarch multilingual_transformer_big  --output-file $OUTPUT \
    --newtask multilingual_translation --remove-bpe < tmp/tmp.bpe 


