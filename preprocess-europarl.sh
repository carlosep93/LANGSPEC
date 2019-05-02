#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=preprocess-europarl.log


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


#de-en  de-es  de-fr  en-es  en-fr  es-fr 

dest_dir="data-bin/europarl"
n_op=32000

de_array=("../europarl/de-en/corpus.tc.de" "../europarl/de-es/corpus.tc.de"  "../europarl/de-fr/corpus.tc.de")
en_array=("../europarl/de-en/corpus.tc.en" "../europarl/en-es/corpus.tc.en"  "../europarl/en-fr/corpus.tc.en")
fr_array=("../europarl/de-fr/corpus.tc.fr" "../europarl/en-fr/corpus.tc.fr"  "../europarl/es-fr/corpus.tc.fr")
es_array=("../europarl/de-es/corpus.tc.es" "../europarl/en-es/corpus.tc.es"  "../europarl/es-fr/corpus.tc.es")

mkdir $dest_dir

learn_bpe $dest_dir/de.codes $n_op "${de_array[@]}"
learn_bpe $dest_dir/en.codes $n_op "${en_array[@]}"
learn_bpe $dest_dir/fr.codes $n_op "${fr_array[@]}"
learn_bpe $dest_dir/es.codes $n_op "${es_array[@]}"


de_dir_array=("../europarl/de-en" "../europarl/de-es"  "../europarl/de-fr")
en_dir_array=("../europarl/de-en" "../europarl/en-es"  "../europarl/en-fr")
fr_dir_array=("../europarl/de-fr" "../europarl/en-fr"  "../europarl/es-fr")
es_dir_array=("../europarl/de-es" "../europarl/en-es"  "../europarl/es-fr")

apply_bpe $dest_dir/codes/de.codes de $destdir "${de_dir_array[@]}"
apply_bpe $dest_dir/codes/en.codes en $destdir "${en_dir_array[@]}"
apply_bpe $dest_dir/codes/fr.codes fr $destdir "${fr_dir_array[@]}"
apply_bpe $dest_dir/codes/es.codes es $destdir "${es_dir_array[@]}"

de_array=("../europarl/de-en/corpus.tc.bpe.de" "../europarl/de-es/corpus.tc.bpe.de"  "../europarl/de-fr/corpus.tc.bpe.de")
en_array=("../europarl/de-en/corpus.tc.bpe.en" "../europarl/en-es/corpus.tc.bpe.en"  "../europarl/en-fr/corpus.tc.bpe.en")
fr_array=("../europarl/de-fr/corpus.tc.bpe.fr" "../europarl/en-fr/corpus.tc.bpe.fr"  "../europarl/es-fr/corpus.tc.bpe.fr")
es_array=("../europarl/de-es/corpus.tc.bpe.es" "../europarl/en-es/corpus.tc.bpe.es"  "../europarl/es-fr/corpus.tc.bpe.es")

get_dictionary $dest_dir/dict.de.txt $n_op "${de_array[@]}"
get_dictionary $dest_dir/dict.en.txt $n_op "${en_array[@]}"
get_dictionary $dest_dir/dict.fr.txt $n_op "${fr_array[@]}"
get_dictionary $dest_dir/dict.es.txt $n_op "${es_array[@]}"

preprocess  ../europarl/de-en/ $dest_dir de en corpus.tc dev test $n_op $dest_dir/dict.de.txt $dest_dir/dict.en.txt
preprocess  ../europarl/de-es/ $dest_dir de es corpus.tc dev test $n_op $dest_dir/dict.de.txt $dest_dir/dict.es.txt 
preprocess  ../europarl/de-fr/ $dest_dir de fr corpus.tc dev test $n_op $dest_dir/dict.de.txt $dest_dir/dict.fr.txt
preprocess  ../europarl/en-es/ $dest_dir en es corpus.tc dev test $n_op $dest_dir/dict.en.txt $dest_dir/dict.es.txt
preprocess  ../europarl/en-fr/ $dest_dir en fr corpus.tc dev test $n_op $dest_dir/dict.en.txt $dest_dir/dict.fr.txt 
preprocess  ../europarl/es-fr/ $dest_dir es fr corpus.tc dev test $n_op $dest_dir/dict.es.txt $dest_dir/dict.fr.txt 
