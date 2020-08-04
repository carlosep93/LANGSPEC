
DEST_DIR='data-bin/audio-enen'
CP_DIR='checkpoints/asr-en-scratch'

mkdir -p $CP_DIR

CUDA_VISIBLE_DEVICES=0 python train.py $DEST_DIR  \
    --clip-norm 20 \
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
    --arch speech_interlingua_transformer_big \
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
    --no-cache-source
