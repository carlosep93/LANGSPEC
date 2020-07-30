
DEST_DIR='/home/cep/interlingua-nodistance/data-bin/audio-enen'
WORKING_DIR='/home/cep/must-c/en-de/processed'
SRC='en'
TGT='en'

mkdir -p $DEST_DIR




subword-nmt apply-bpe -c  ${DEST_DIR}/codes/$TGT.codes < $WORKING_DIR/train/train.$TGT >  ${DEST_DIR}/train.$TGT 
subword-nmt apply-bpe -c  ${DEST_DIR}/codes/$TGT.codes < $WORKING_DIR/dev/dev.$TGT >  ${DEST_DIR}/dev.$TGT 
subword-nmt apply-bpe -c  ${DEST_DIR}/codes/$TGT.codes < $WORKING_DIR/tst-COMMON/tst-COMMON.$TGT >  ${DEST_DIR}/tst-COMMON.$TGT

cp $WORKING_DIR/train/train.h5 $DEST_DIR/train.h5
cp $WORKING_DIR/dev/dev.h5 $DEST_DIR/dev.h5
cp $WORKING_DIR/tst-COMMON/tst-COMMON.h5 $DEST_DIR/tst-COMMON.h5

python preprocess_audio.py -s $SRC  -t $TGT --format h5 --inputtype audio \
    --trainpref $DEST_DIR/train --validpref $DEST_DIR/dev \
    --testpref $DEST_DIR/tst-COMMON --destdir $DEST_DIR --tgtdict $DEST_DIR/dict.$TGT.txt
