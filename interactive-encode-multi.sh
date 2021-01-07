#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=encodings-tied/shared/encodings-multi.log

WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
DEST_DIR="../fairseq/data-bin/multi-europarl-ru-joint"
CP_DIR="/scratch/carlos/baseline-ru-hare-embeddings"
CP="checkpoint_best.pt"
PYTHON="/home/usuaris/veu/cescola/virtualenv-16.0.0/torch/bin/python"
FAIRSEQ_DIR="/home/usuaris/veu/cescola/fairseq"
N=100
mkdir tmp




encode() {
    INPUT_DATA=$1
    OUTPUT=$2
    SRC=$3
    TGT=$4
    
    head --l $N  $INPUT_DATA > tmp/tmp.$SRC 
    
    sed  -i 's/^/<2en> /' tmp/tmp.$SRC

    subword-nmt apply-bpe -c $DEST_DIR/corpus.tc.shuf.codes < tmp/tmp.$SRC > tmp/tmp.bpe.$SRC

    
    CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points 500 \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
        --enc-model $CP_DIR/$CP --enc-key ${SRC}-${TGT} \
        --dec-model $CP_DIR/$CP --dec-key ${SRC}-${TGT} \
        --newkey $SRC-$TGT --newarch transformer_big --output-file $OUTPUT \
        --newtask translation --remove-bpe < tmp/tmp.bpe.$SRC 
}

mkdir -p encodings-tied/shared

encode ../europarl/de-en/test.de encodings-tied/shared/encodings-tied-de.json src tgt
encode ../europarl/de-en/test.en encodings-tied/shared/encodings-tied-en.json src tgt
encode ../europarl/en-es/test.es encodings-tied/shared/encodings-tied-es.json src tgt
encode ../europarl/en-fr/test.fr encodings-tied/shared/encodings-tied-fr.json src tgt
encode ../europarl/ru-en/test.ru encodings-tied/shared/encodings-tied-ru.json src tgt
