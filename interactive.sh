#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=interactive.log

SRC="en"
TGT="ru"
DEST_DIR="data-bin/europarl"
ENC_CP_DIR="/scratch/carlos/europarl-basic-tied-ru"
DEC_CP_DIR="/scratch/carlos/europarl-basic-tied-ru"
CP="checkpoint_best.pt"
PYTHON="/home/usuaris/veu/cescola/virtualenv-16.0.0/torch/bin/python"
FAIRSEQ_DIR="/home/usuaris/veu/cescola/interlingua-nodistance"
INPUT_DATA="/scratch/carlos/mt_gender/pr_data/interlingua/en-pro-inter.bpe.txt"
OUTPUT="/scratch/carlos/mt_gender/translations/inter-ru/interru-en-pro.ru"







$PYTHON interactive-add-lang.py $DEST_DIR --path $CP_DIR/$CP \
    --beam 5 --buffer-size 20  --batch-size 20 --source-lang ${SRC} --target-lang ${TGT} --task interlingua_nodistance_translation  \
    --enc-model $ENC_CP_DIR/$CP   --enc-key $SRC-$TGT \
    --dec-model $DEC_CP_DIR/$CP  --dec-key $SRC-$TGT \
    --newkey $SRC-$TGT  --newarch interlingua_transformer --freeze-schedule n-n \
    --newtask interlingua_nodistance_translation --remove-bpe \
    < $INPUT_DATA > $OUTPUT 


python ../fairseq/clean-output.py < $OUTPUT > $OUTPUT.cl



