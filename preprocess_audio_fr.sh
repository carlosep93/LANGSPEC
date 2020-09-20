
DEST_DIR='/home/cep/interlingua-nodistance/data-bin/audio-enfr-melspec-test'
WORKING_DIR='/home/cep/must-c/en-fr/processed'
SRC='ens'
TGT='fr'

mkdir -p $DEST_DIR




subword-nmt apply-bpe -c  ${DEST_DIR}/codes/$TGT.codes < $WORKING_DIR/test.$TGT >  ${DEST_DIR}/test.$TGT

WORKING_DIR='/home/cep/must-c/en-fr/processed/'

cp $WORKING_DIR/test.h5 $DEST_DIR/test.h5

python preprocess_audio.py -s $SRC  -t $TGT --format h5 --inputtype audio \
    --trainpref $DEST_DIR/test --validpref $DEST_DIR/test \
    --testpref $DEST_DIR/test --destdir $DEST_DIR --tgtdict $DEST_DIR/dict.$TGT.txt

