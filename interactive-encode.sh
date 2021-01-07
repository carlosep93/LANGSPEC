#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=interactive-encoder.log 

WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
CP="checkpoint_best.pt"
PYTHON="/home/usuaris/veu/cescola/virtualenv-16.0.0/torch/bin/python"
FAIRSEQ_DIR="/home/usuaris/veu/cescola/fairseq"
N=3888
mkdir tmp



encode-lstm() {
    INPUT_DATA=$1
    OUTPUT=$2
    SRC=$3
    TGT=$4
    
    head --l $N  $INPUT_DATA > tmp/tmp.lstm.$SRC 
    
    sed  -i 's/^/<2es> /' tmp/tmp.lstm.$SRC

    #subword-nmt apply-bpe -c $DEST_DIR/corpus.tc.shuf.codes < tmp/tmp.$SRC > tmp/tmp.bpe.$SRC

    
    CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points 500 \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
        --enc-model $CP_DIR/$CP --enc-key ${SRC}-${TGT} \
        --dec-model $CP_DIR/$CP --dec-key ${SRC}-${TGT} \
        --newkey $SRC-$TGT --newarch lstm --output-file $OUTPUT \
        --newtask translation --remove-bpe < tmp/tmp.lstm.$SRC 
}

encode() {
    INPUT_DATA=$1
    OUTPUT=$2
    SRC=$3
    TGT=$4
    
    head --l $N  $INPUT_DATA > tmp/tmp.$SRC 
    
    sed  -i 's/^/<2es> /' tmp/tmp.$SRC

    #subword-nmt apply-bpe -c $DEST_DIR/corpus.tc.shuf.codes < tmp/tmp.$src > tmp/tmp.bpe.$src

    
    cuda_visible_devices="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
        --enc-model $CP_DIR/$CP --enc-key ${SRC}-${TGT} \
        --dec-model $CP_DIR/$CP --dec-key ${SRC}-${TGT} \
        --newkey $SRC-$TGT --newarch transformer_big --output-file $OUTPUT \
        --newtask translation --remove-bpe < tmp/tmp.$SRC 
}

encode_notag() {
    INPUT_DATA=$1
    OUTPUT=$2
    SRC=$3
    TGT=$4
    
    head --l $N  $INPUT_DATA > tmp/tmp.$SRC 
    

    #subword-nmt apply-bpe -c $dest_dir/corpus.tc.shuf.codes < tmp/tmp.$src > tmp/tmp.bpe.$src

    
    cuda_visible_devices="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
         --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation \
        --enc-model $CP_DIR/$CP --enc-key ${SRC}-${TGT} \
        --dec-model $CP_DIR/$CP --dec-key ${SRC}-${TGT} \
        --newkey $SRC-$TGT --newarch transformer_big --output-file $OUTPUT --newtask translation --remove-bpe < tmp/tmp.$SRC 
}



encode-interlingua() {
    INPUT_DATA=$1
    OUTPUT=$2
    SRC=$3
    TGT=$4

    head --l $N  $INPUT_DATA > tmp/tmp.$SRC 

    #subword-nmt apply-bpe -c $DEST_DIR/codes/$SRC.codes < tmp/tmp.$SRC > tmp/tmp.bpe.$SRC


    CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python interactive-encode.py $DEST_DIR --path $CP_DIR/$CP --n-points $N \
     --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task multilingual_translation \
     --enc-model $CP_DIR/$CP --enc-key ${SRC}-${TGT} \
     --dec-model $CP_DIR/$CP --dec-key ${SRC}-${TGT} \
     --newkey $SRC-$TGT --newarch multilingual_transformer  --output-file $OUTPUT  \
     --newtask multilingual_translation --remove-bpe < tmp/tmp.$SRC 

}

mkdir -p  /scratch/carlos/mt_gender/encodings

#Interlingua encodings
#CP_DIR="/scratch/carlos/europarl-basic-tied"
#DEST_DIR="data-bin/europarl"
#encode-interlingua /scratch/carlos/mt_gender/pr_data/interlingua/en-inter.bpe.txt /scratch/carlos/mt_gender/encodings/encodings-inter.json en es

#Shared encodings
#CP_DIR="/scratch/carlos/baseline-ru-hare-embeddings"
#DEST_DIR="/home/usuaris/veu/cescola/fairseq/data-bin/multi-europarl-ru-joint/"
#encode /scratch/carlos/mt_gender/pr_data/tr_shared/en-shared.bpe.txt /scratch/carlos/mt_gender/encodings/encodings-shared.json src tgt

#Shared encodings no russian
#CP_DIR="/scratch/carlos/baseline-share-embeddings"
#DEST_DIR="/home/usuaris/veu/cescola/fairseq/data-bin/multi-europarl-noru-joint/"
#encode /scratch/carlos/mt_gender/pr_data/tr_shared_noru/en-shared.bpe.txt /scratch/carlos/mt_gender/encodings/encodings-shared_noru.json src tgt


#LSTM encodings
#CP_DIR="/scratch/carlos/europarl-multi-lstm-4layers"
#DEST_DIR="/home/usuaris/veu/cescola/fairseq/data-bin/multi-europarl-no-ru/"
#encode-lstm /scratch/carlos/mt_gender/pr_data/lstm_shared/en-lstm.bpe.txt /scratch/carlos/mt_gender/encodings/encodings-lstm.json src tgt

#Bilingual FR
CP_DIR="/scratch/carlos/baseline-enfr"
DEST_DIR="/home/usuaris/veu/cescola/interlingua-nodistance/data-bin/europarl"
encode_notag /scratch/carlos/mt_gender/pr_data/interlingua/en-inter.bpe.txt /scratch/carlos/mt_gender/encodings/encodings-enfr2.json en fr


#Interlingua + Russian encodings
#CP_DIR="/scratch/carlos/europarl-basic-tied-ru"
#DEST_DIR="data-bin/europarl"
#encode-interlingua /scratch/carlos/mt_gender/pr_data/interlingua/en-inter.bpe.txt /scratch/carlos/mt_gender/encodings/encodings-inter-ru.json en es


