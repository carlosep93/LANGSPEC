#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH -x veuc02                                                       
#SBATCH --output=generate-asr-callhome.log


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="scp"
TGT="es"
DEST_DIR="data-bin/callhome/"
CP_DIR="checkpoints/asr-callhome"
CP="checkpoint_best.pt"

stdbuf -i0 -e0 -o0 python generate-asr.py $DEST_DIR --path $CP_DIR/$CP \
    --beam 5 --batch-size 5 --source-lang ${SRC} --target-lang ${TGT} --task speech_translation --remove-bpe \
    --raw-text --kernels 10 14 18 --tds-layers 2 3 6 --input-channels 80 --W 80 --F 27 \
    --T 100  --mf 1 --mt 1 --max-source-positions 1000 \
    --mask-speech False


