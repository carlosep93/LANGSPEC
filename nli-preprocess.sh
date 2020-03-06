#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=preprocess-nli.log


DEST_DIR='data-bin/multinli'


WORKING_DIR='/home/usuaris/veu/cescola/MultiNLI/multinli_1.0/'




#Train split

#subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/en.codes < ${WORKING_DIR}/snli_1.0/ref.train.en  >  ${DEST_DIR}/ref.train.en
#subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/en.codes < ${WORKING_DIR}/snli_1.0/hyp.train.en  >  ${DEST_DIR}/hyp.train.en
#cp  ${WORKING_DIR}/snli_1.0/lab.train.en $DEST_DIR/


#Valid split

#subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/en.codes < ${WORKING_DIR}/snli_1.0/ref.valid.en  >  ${DEST_DIR}/ref.valid.en
#subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/en.codes < ${WORKING_DIR}/snli_1.0/hyp.valid.en  >  ${DEST_DIR}/hyp.valid.en
#cp  ${WORKING_DIR}/snli_1.0/lab.valid.en $DEST_DIR/

#Test split

#for lang in de en es fr ru
#do
#    subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/$lang.codes < ${WORKING_DIR}/../xnli/procs/ref.test.tok.tok.$lang  >  ${DEST_DIR}/ref.test.$lang
#    subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/$lang.codes < ${WORKING_DIR}/../xnli/procs/hyp.test.tok.tok.$lang  >  ${DEST_DIR}/hyp.test.$lang
#    cp  ${WORKING_DIR}/../xnli/lab.test.$lang $DEST_DIR/
#done

#subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/en.codes < ${WORKING_DIR}/ref.test.tok.en  >  ${DEST_DIR}/ref.test.en
#subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/en.codes < ${WORKING_DIR}/hyp.test.tok.en  >  ${DEST_DIR}/hyp.test.en
    
#cp  ${WORKING_DIR}/lab.test.en $DEST_DIR/


# MultiNli

mkdir -p $DEST_DIR

#Train split

#subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/en.codes < ${WORKING_DIR}/ref.train.en  >  ${DEST_DIR}/ref.train.en
#subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/en.codes < ${WORKING_DIR}/hyp.train.en  >  ${DEST_DIR}/hyp.train.en
#cp  ${WORKING_DIR}/lab.train.en $DEST_DIR/


#Valid split

#subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/en.codes < ${WORKING_DIR}/ref.dev.en  >  ${DEST_DIR}/ref.valid.en
#subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/en.codes < ${WORKING_DIR}/hyp.dev.en  >  ${DEST_DIR}/hyp.valid.en
#cp  ${WORKING_DIR}/lab.dev.en $DEST_DIR/lab.valid.en

#Test split

WORKING_DIR='/home/usuaris/veu/cescola/XNLI/XNLI-1.0/procs/'

for lang in de en es fr ru
do
   subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/$lang.codes < ${WORKING_DIR}/ref.dev.tok.$lang  >  ${DEST_DIR}/ref.valid.$lang
   subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/$lang.codes < ${WORKING_DIR}/hyp.dev.tok.$lang  >  ${DEST_DIR}/hyp.valid.$lang
   cp  ${WORKING_DIR}/lab.dev.$lang $DEST_DIR/lab.valid.$lang
   
   subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/$lang.codes < ${WORKING_DIR}/ref.test.tok.$lang  >  ${DEST_DIR}/ref.test.$lang
   subword-nmt apply-bpe -c  ${DEST_DIR}/../europarl/codes/$lang.codes < ${WORKING_DIR}/hyp.test.tok.$lang  >  ${DEST_DIR}/hyp.test.$lang
   cp  ${WORKING_DIR}/lab.test.$lang $DEST_DIR/lab.test.$lang
   cp  ${DEST_DIR}/../europarl/dict.${lang}.txt ${DEST_DIR}/dict.${lang}.txt

done

