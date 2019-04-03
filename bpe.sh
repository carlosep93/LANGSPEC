#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=4G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=bpe.log


WORKING_DIR="/home/usuaris/veu/cescola/interlingua-tr/en-tr"
n_op=16000


echo "apply bpe to turkish"
subword-nmt learn-bpe -s $n_op < ${working_dir}/train/train.tr > ${working_dir}/train/train.codes.tr
subword-nmt apply-bpe -c  ${working_dir}/train/train.codes.tr < ${working_dir}/test/test.tr > ${working_dir}/test/test.tr.bpe 

echo "apply bpe to english"
subword-nmt learn-bpe -s $n_op < ${working_dir}/train/train.en > ${working_dir}/train/train.codes.en
subword-nmt apply-bpe -c  ${working_dir}/train/train.codes.en < ${working_dir}/test/test.en > ${working_dir}/test/test.en.bpe


