#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=interactive-encode-distance-es.log

WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="es"
TGT="en"
DEST_DIR="../interlingua-fairseq/data-bin/wmt13.tokenized.32k.en-es"
CP_DIR="../interlingua-fairseq/checkpoints/esen-interlingua-finetuning"
CP="checkpoint_best.pt"
PYTHON="/home/usuaris/veu/cescola/virtualenv-16.0.0/torch/bin/python"
FAIRSEQ_DIR="/home/usuaris/veu/cescola/fairseq"
INPUT_DATA="../europarl/en-es/test.es"
OUTPUT="encodings-distance-es.json"

mkdir tmp

cp $INPUT_DATA tmp/tmp-es.src

#sed 's/^/<2en> /' tmp/tmp3.src > tmp/tmp.pref3.src

subword-nmt apply-bpe -c $DEST_DIR/general.es-en.tc.clean.codes.es < tmp/tmp-es.src > tmp/tmp-es.bpe

CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points 3000 \
     --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task multilingual_translation --maxlength 220 \
    --enc-model $CP_DIR/$CP --enc-key ${SRC}-${TGT} \
    --dec-model $CP_DIR/$CP --dec-key ${SRC}-${TGT} \
    --newkey $SRC-$TGT --newarch multilingual_transformer_iwslt_de_en  --output-file $OUTPUT \
    --newtask multilingual_translation --remove-bpe < tmp/tmp-es.bpe 


