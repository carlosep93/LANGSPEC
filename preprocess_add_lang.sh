#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=preprocess-add-fren.log


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="en"
TGT="fr"

TRN_PREF="general.fr-en.tc.clean"
VAL_PREF="newstest2012.tc"
TES_PREF="newstest2013.tc"
SRC_CODES="/home/usuaris/veu/cescola/interlingua-fairseq/data-bin/wmt13.tokenized.32k.en-es/general.es-en.tc.clean.codes.en"
TGT_CODES="data-bin/wmt13.tokenized.32k.fr-en/fr-en.codes.fr"
DEST_DIR="data-bin/wmt13.tokenized.32k.fr-en"

mkdir $DEST_DIR


N_OP=32000


#subword-nmt learn-bpe -s $N_OP  < ${WORKING_DIR}/${TRN_PREF}.${TGT} > ${TGT_CODES}

echo "apply bpe to " $SRC

#subword-nmt apply-bpe -c  ${SRC_CODES} < ${WORKING_DIR}/${TRN_PREF}.${SRC} >  ${DEST_DIR}/${TRN_PREF}.bpe.${SRC}
#subword-nmt apply-bpe -c  ${SRC_CODES} < ${WORKING_DIR}/${VAL_PREF}.${SRC} >  ${DEST_DIR}/${VAL_PREF}.bpe.${SRC}
#subword-nmt apply-bpe -c  ${SRC_CODES} < ${WORKING_DIR}/${TES_PREF}.${SRC} >  ${DEST_DIR}/${TES_PREF}.bpe.${SRC}

echo "apply bpe to " $TGT

#subword-nmt apply-bpe -c  ${TGT_CODES} < ${WORKING_DIR}/${TRN_PREF}.${TGT} >  ${DEST_DIR}/${TRN_PREF}.bpe.${TGT}
#subword-nmt apply-bpe -c  ${TGT_CODES} < ${WORKING_DIR}/${VAL_PREF}.${TGT} >  ${DEST_DIR}/${VAL_PREF}.bpe.${TGT}
#subword-nmt apply-bpe -c  ${TGT_CODES} < ${WORKING_DIR}/${TES_PREF}.${TGT} >  ${DEST_DIR}/${TES_PREF}.bpe.${TGT}

python preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $DEST_DIR/${TRN_PREF}.bpe --validpref $DEST_DIR/${VAL_PREF}.bpe --testpref $DEST_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --nwordstgt $N_OP --nwordssrc $N_OP --srcdict data-bin/wmt13.tokenized.32k.en-es/dict.en.txt 

#Copy test for autoencoder translation

cp $DEST_DIR/test.${SRC}-${TGT}.${SRC}.bin  $DEST_DIR/test.${SRC}-${SRC}.${SRC}.bin
cp $DEST_DIR/test.${SRC}-${TGT}.${SRC}.idx  $DEST_DIR/test.${SRC}-${SRC}.${SRC}.idx
cp $DEST_DIR/test.${SRC}-${TGT}.${TGT}.bin  $DEST_DIR/test.${TGT}-${TGT}.${TGT}.bin
cp $DEST_DIR/test.${SRC}-${TGT}.${TGT}.idx  $DEST_DIR/test.${TGT}-${TGT}.${TGT}.idx


