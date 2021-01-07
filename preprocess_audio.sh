#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=preprocess_audio_en_enfr.log



DEST_DIR='data-bin/audio-en-enfr'
WORKING_DIR='/scratch/carlos/must-c/en-fr'
CODES_DIR='/home/usuaris/veu/cescola/interlingua-nodistance/data-bin/europarl'
SPEECH_DIR='/scratch/gerard.ion.gallego/MUSTC_v1.0/en-fr/data'
SRC='ens'
TGT='en'

mkdir -p $DEST_DIR

cp $CODES_DIR/dict.$TGT.txt $DEST_DIR/


subword-nmt apply-bpe -c  ${CODES_DIR}/codes/$TGT.codes < $WORKING_DIR/train.tc.$TGT >  ${DEST_DIR}/train.$TGT 
subword-nmt apply-bpe -c  ${CODES_DIR}/codes/$TGT.codes < $WORKING_DIR/dev.tc.$TGT >  ${DEST_DIR}/dev.$TGT 
subword-nmt apply-bpe -c  ${CODES_DIR}/codes/$TGT.codes < $WORKING_DIR/test.tc.$TGT >  ${DEST_DIR}/test.$TGT

cp $SPEECH_DIR/train/h5/train.h5 $DEST_DIR/train.h5
cp $SPEECH_DIR/dev/h5/dev.h5 $DEST_DIR/dev.h5
cp $SPEECH_DIR/tst-COMMON/h5/tst-COMMON.h5 $DEST_DIR/test.h5

python preprocess_audio.py -s $SRC  -t $TGT --format h5 --inputtype audio \
    --trainpref $DEST_DIR/train --validpref $DEST_DIR/dev \
    --testpref $DEST_DIR/test --destdir $DEST_DIR --tgtdict $DEST_DIR/dict.$TGT.txt

rm $DEST_DIR/train.h5 $DEST_DIR/dev.h5 $DEST_DIR/test.h5
