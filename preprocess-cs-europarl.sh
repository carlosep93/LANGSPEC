#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=preprocess-cs-europarl.log


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
        for f in corpus.tc.$lang dev.$lang test.$lang dev.tc.$lang test.tc.$lang; do
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


#de-en  de-es  de-fr  en-es  en-fr  es-fr 

dest_dir="data-bin/europarl"
n_op=32000

cs_array=("../europarl/cs-en/corpus.tc.cs")

mkdir $dest_dir

#learn_bpe $dest_dir/codes/cs.codes $n_op "${cs_array[@]}"


cs_dir_array=("../europarl/cs-en")

#apply_bpe $dest_dir/codes/cs.codes cs $dest_dir  "${cs_dir_array[@]}"
#apply_bpe $dest_dir/codes/en.codes en $dest_dir  "${cs_dir_array[@]}"

cs_array=("../europarl/cs-en/corpus.tc.bpe.cs")

#get_dictionary $dest_dir/dict.cs.txt $n_op "${cs_array[@]}"

preprocess  ../europarl/cs-en/ $dest_dir cs en corpus.tc dev.tc test.tc $n_op $dest_dir/dict.cs.txt $dest_dir/dict.en.txt
