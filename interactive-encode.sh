#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=interactive-encode-multi-es.log

WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="src"
TGT="tgt"
DEST_DIR="../fairseq/data-bin/multi-europarl"
CP_DIR="../fairseq/checkpoint/multi-europarl"
CP="checkpoint_best.pt"
PYTHON="/home/usuaris/veu/cescola/virtualenv-16.0.0/torch/bin/python"
FAIRSEQ_DIR="/home/usuaris/veu/cescola/fairseq"
INPUT_DATA="../europarl/en-es/test.es"
OUTPUT="encodings-multi-es.json"

mkdir tmp

cp $INPUT_DATA tmp/tmp.src

sed  -i 's/^/<2en> /' tmp/tmp.src

subword-nmt apply-bpe -c $DEST_DIR/corpus.tc.shuf.codes.src < tmp/tmp.src > tmp/tmp.bpe

CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points 200 \
     --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
    --enc-model $CP_DIR/$CP --enc-key ${SRC}-${TGT} \
    --dec-model $CP_DIR/$CP --dec-key ${SRC}-${TGT} \
    --newkey $SRC-$TGT --newarch transformer_iwslt_de_en --output-file $OUTPUT \
    --newtask translation --remove-bpe < tmp/tmp.bpe 


