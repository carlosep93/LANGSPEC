#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=train_scratch_enen_adapt.log


DEST_DIR='data-bin/audio-enen-melspec'
CP_DIR='checkpoints/asr_en_enfr_scratch_melspec-3conv'

mkdir -p $CP_DIR

python train.py $DEST_DIR  \
    --clip-norm 20.0 \
    --max-sentences 8 \
    --max-tokens 12000 \
    --save-dir  $CP_DIR \
    --max-epoch 100 \
    --lr 5e-3 \
    --min-lr 1e-08 \
    --dropout 0.1 \
    --lr-schedule inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 3e-4 \
    --optimizer adam \
    --arch speech_interlingua_transformer_big_3conv \
    --task speech_interlingua_translation  \
    --audio-input \
    --max-source-positions 2000 --max-target-positions 300 \
    --update-freq 64 \
    --skip-invalid-size-inputs-valid-test \
    --sentence-avg \
    --distance-penalty log \
    --criterion label_smoothed_cross_entropy \
    --lang-pairs ens-en \
    --freeze-schedule n-n \
    --label-smoothing 0.1 \
    --no-cache-source \
    --audio-features 40 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --final-norm
    --adapter-network \
    --adapter-size 4096 \
    --adapter-attn-layers 0 \
