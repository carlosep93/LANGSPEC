#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=train_ptr_enen_adapt_8192.log


DEST_DIR='data-bin/audio-enen-melspec'
CP_DIR='checkpoints/asr-enen-ptr-melspec_adapt_8192'
PREV_ENC_DIR='checkpoints/asr_en_enfr_scratch_melspec-3conv'
PREV_DEC_DIR='/scratch/carlos/europarl-basic-tied-normalize'


mkdir -p $CP_DIR

#cp checkpoints/asr-en-scratch/checkpoint_best.py $CP_DIR


python add_audio.py $DEST_DIR \
    --clip-norm 20.0 \
    --max-sentences 8 \
    --max-tokens 12000 \
    --save-dir $CP_DIR \
    --max-epoch 100 \
    --lr 0.0001 \
    --min-lr 1e-08 \
    --dropout 0.1 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 3e-4 \
    --optimizer adam \
    --arch speech_interlingua_transformer_big_3conv \
    --task speech_interlingua_translation  \
    --audio-input \
    --max-source-positions 2000 --max-target-positions 300 \
    --update-freq 64  \
    --skip-invalid-size-inputs-valid-test \
    --sentence-avg \
    --distance-penalty log \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --enckeys ens-en \
    --deckeys de-en \
    --newkeys ens-en \
    --reuse both \
    --prev-enc-models $PREV_ENC_DIR \
    --prev-dec-models $PREV_DEC_DIR  \
     --reset-optimizer \
    --lang-pairs ens-en \
    --restore-file checkpoint_best.pt \
    --freeze-schedule n-f \
    --no-cache-source \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --final-norm \
    --adapter-network \
    --adapter-size 8192 \
    --adapter-attn-layers 0 \
