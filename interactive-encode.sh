#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=interactive-encode-kk.log

WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="kk"
TGT="tr"
DEST_DIR="data-bin/wmt19.tokenized.10k.kk-en"
CP_DIR="checkpoints/add-kken-tren-int-auto"
CP="checkpoint_best.pt"
PYTHON="/home/usuaris/veu/cescola/virtualenv-16.0.0/torch/bin/python"
FAIRSEQ_DIR="/home/usuaris/veu/cescola/fairseq"
INPUT_DATA="data-bin/tr-kk/test.bpe.kk"
OUTPUT="encodings-kk.json"


CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points 100 \
     --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task multilingual_translation \
    --enc-model $CP_DIR/$CP --enc-key kk-en \
    --dec-model checkpoints/tren-int-auto/checkpoint_best.pt --dec-key en-tr \
    --newkey kk-tr --newarch multilingual_transformer_iwslt_de_en \
    --newtask multilingual_translation --remove-bpe < $INPUT_DATA > $OUTPUT



