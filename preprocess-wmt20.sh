#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=preprocess-wmt20-taen.log


learn_bpe() {
    out=$1
    shift
    n_op=$1
    shift
    arr=("${@}")
    cat ${arr[@]} | subword-nmt learn-bpe -s $n_op -o $out
}


apply_bpe() {
    codes=$1
    shift
    lang=$1
    shift
    outdir=$1
    shift
    paths=("${@}")
    for path in  "${paths[@]}"; do
        echo $path
        for f in corpus.tc.$lang dev.$lang test.$lang; do
            new_f="${f/.$lang/.bpe.$lang}"
            echo $f
            echo $new_f
            echo "************************"
            subword-nmt apply-bpe -c  $codes < $path/$f >  $path/$new_f
        done
    done
}

get_dictionary() {
    o=$1
    shift
    n=$1
    shift
    arr=("${@}")
    python get_dictionary.py -c ${arr[@]} -o $o -n $n
}


preprocess() {
   
    WORK_DIR=$1 
    DEST_DIR=$2
    SRC=$3
    TGT=$4
    TRN_PREF=$5
    VAL_PREF=$6
    TEST_PREF=$7
    N_OP=$8
    SRC_DICT=$9
    TGT_DICT=${10}
    
    python preprocess.py --source-lang $SRC --target-lang $TGT \
        --trainpref $WORK_DIR/${TRN_PREF}.bpe --validpref $WORK_DIR/${VAL_PREF}.bpe --testpref $WORK_DIR/${TEST_PREF}.bpe \
        --destdir $DEST_DIR  --srcdict $SRC_DICT --tgtdict $TGT_DICT
}



dest_dir="data-bin/wmt20-indic"
n_op=16000

ta_train="/home/usuaris/veu/cescola/wmt20/ta-en/par/processed-indic-nlp/train.tc.ta"
en_train="/home/usuaris/veu/cescola/wmt20/ta-en/par/processed-indic-nlp/train.tc.en"

ta_valid="/home/usuaris/veu/cescola/wmt20/ta-en/par/processed-indic-nlp/valid.tc.ta"
en_valid="/home/usuaris/veu/cescola/wmt20/ta-en/par/processed-indic-nlp/valid.tc.en"

ta_test="/home/usuaris/veu/cescola/wmt20/ta-en/par/processed-indic-nlp/test.tc.ta"
en_test="/home/usuaris/veu/cescola/wmt20/ta-en/par/processed-indic-nlp/test.tc.en"

mkdir $dest_dir

#LEARN TA BPE

subword-nmt learn-bpe -s $n_op -o $dest_dir/ta.codes < $ta_train

#COPY EN BPE

en_codes="/home/usuaris/veu/cescola/interlingua-nodistance/data-bin/europarl/codes/en.codes"

cp $en_codes $dest_dir

# APPLY BPE TO TA DATA

subword-nmt apply-bpe -c  $dest_dir/ta.codes < $ta_train  >  $dest_dir/train.bpe.ta
subword-nmt apply-bpe -c  $dest_dir/en.codes < $en_train  >  $dest_dir/train.bpe.en

subword-nmt apply-bpe -c  $dest_dir/ta.codes < $ta_valid  >  $dest_dir/valid.bpe.ta
subword-nmt apply-bpe -c  $dest_dir/en.codes < $en_valid  >  $dest_dir/valid.bpe.en

subword-nmt apply-bpe -c  $dest_dir/ta.codes < $ta_test  >  $dest_dir/test.bpe.ta
subword-nmt apply-bpe -c  $dest_dir/en.codes < $en_test  >  $dest_dir/test.bpe.en


# GET TA DICTIONARY

ta_array=("data-bin/wmt20-indic/train.bpe.ta")

get_dictionary $dest_dir/dict.ta.txt $n_op "${ta_array[@]}"

# COPY EN DICT FROM PREVIOUS MODEL

cp data-bin/europarl/dict.en.txt $dest_dir 

# BINARIZE DATA

preprocess  $dest_dir $dest_dir ta en train valid test $n_op $dest_dir/dict.ta.txt $dest_dir/dict.en.txt
