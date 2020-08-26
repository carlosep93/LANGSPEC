
DEST_DIR='data-bin/audio-enen-melspec'
CP_DIR='checkpoints/asr_en_melspec_ptr_3conv_adapt_3ttn'
PREV_ENC_DIR='checkpoints/asr_en_scratch_melspec-3conv'
PREV_DEC_DIR='checkpoints/europarl-basic-tied'


mkdir -p $CP_DIR

cp checkpoints/asr-en-scratch/checkpoint_best.py $CP_DIR

CUDA_VISIBLE_DEVICES=4 python add_audio.py $DEST_DIR \
    --clip-norm 20.0 \
    --max-sentences 8 \
    --max-tokens 12000 \
    --save-dir $CP_DIR \
    --max-epoch 100 \
    --lr 0.001 \
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
    --enckey ens-en \
    --deckey de-en \
    --newkey ens-en \
    --reuse both \
    --prev-enc-model $PREV_ENC_DIR \
    --prev-dec-model $PREV_DEC_DIR \
     --reset-optimizer \
    --lang-pairs ens-en \
    --restore-file checkpoint_best.pt \
    --freeze-schedule n-f \
    --no-cache-source \
    --encoder-normalize-before \
    --adapter-network \
    --adapter-size 256 \
    --adapter-attn-layers 3 
    #--decoder-normalize-before \
    #--final-norm \
