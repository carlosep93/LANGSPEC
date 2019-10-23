#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=preprocess-fren500.log


WORKING_DIR="data-bin/corpus-fren-500k"
SRC="fr"
TGT="en"

TRN_PREF="train"
VAL_PREF="newstest2012.tc"
TES_PREF="newstest2013.tc"

DEST_DIR="data-bin/wmt13.tokenized.32k.500k.fr-en"

mkdir $DEST_DIR


N_OP=32000

CORPUS="~/europarl/de-en/corpus.tc.de ~/europarl/de-en/corpus.tc.en ~/europarl/de-es/corpus.tc.de ~/europarl/de-es/corpus.tc.es ~/europarl/de-fr/corpus.tc.de ~/europarl/de-fr/corpus.tc.fr ~/europarl/en-es/corpus.tc.en ~/europarl/en-es/corpus.tc.es ~/europarl/en-fr/corpus.tc.en ~/europarl/en-fr/corpus.tc.fr ~/europarl/es-fr/corpus.tc.fr ~/europarl/es-fr/corpus.tc.fr"


learn-bpe() {
    CORPUS=$1
    CODES_PATH=$2
    cat ${CORPUS} | subword-nmt learn-bpe -s $N_OP -o ${CODES_PATH}/${TRN_PREF}.codes
}


apply-bpe() {
    LAN=$1
    WORKING_DIR=$2
    DEST_DIR=$3

    echo "apply bpe to " $LAN

    subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes < ${WORKING_DIR}/${TRN_PREF}.${LAN} >  ${DEST_DIR}/${TRN_PREF}.bpe.${LAN}
    subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes < ${WORKING_DIR}/${VAL_PREF}.${LAN} >  ${DEST_DIR}/${VAL_PREF}.bpe.${LAN}
    subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes < ${WORKING_DIR}/${TES_PREF}.${LAN} >  ${DEST_DIR}/${TES_PREF}.bpe.${LAN}
}

python preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $DEST_DIR/${TRN_PREF}.bpe --validpref $DEST_DIR/${VAL_PREF}.bpe --testpref $DEST_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --nwordstgt $N_OP --nwordssrc $N_OP --tgtdict data-bin/wmt19.tokenized.32k.tr-en/dict.en.txt 

#Copy test for autoencoder translation

cp $DEST_DIR/test.${SRC}-${TGT}.${SRC}.bin  $DEST_DIR/test.${SRC}-${SRC}.${SRC}.bin
cp $DEST_DIR/test.${SRC}-${TGT}.${SRC}.idx  $DEST_DIR/test.${SRC}-${SRC}.${SRC}.idx
cp $DEST_DIR/test.${SRC}-${TGT}.${TGT}.bin  $DEST_DIR/test.${TGT}-${TGT}.${TGT}.bin
cp $DEST_DIR/test.${SRC}-${TGT}.${TGT}.idx  $DEST_DIR/test.${TGT}-${TGT}.${TGT}.idx


