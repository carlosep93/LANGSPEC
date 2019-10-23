#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=preprocess-asr-callhome.log


WORKING_DIR="/home/usuaris/veu/cescola/callhome/"
SRC="es"
TGT="scp"

TRN_PREF="train"
VAL_PREF="dev"
TES_PREF="test"

DEST_DIR="data-bin/callhome"

mkdir $DEST_DIR


N_OP=4000


subword-nmt learn-bpe -s $N_OP -o ${DEST_DIR}/${TRN_PREF}.codes <  ${WORKING_DIR}/train/${TRN_PREF}.tc.${SRC} 

echo "apply bpe to " $SRC

subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes < ${WORKING_DIR}/train/${TRN_PREF}.tc.${SRC} >  ${DEST_DIR}/${TRN_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes < ${WORKING_DIR}/dev/${VAL_PREF}.tc.${SRC} >  ${DEST_DIR}/${VAL_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes < ${WORKING_DIR}/test/${TES_PREF}.tc.${SRC} >  ${DEST_DIR}/${TES_PREF}.bpe.${SRC}


python preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $DEST_DIR/${TRN_PREF}.bpe --validpref $DEST_DIR/${VAL_PREF}.bpe --testpref $DEST_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR --output-format raw --only-source

