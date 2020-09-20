
SRC="ens"
TGT="fr"
DEST_DIR="data-bin/audio-enfr-melspec-test"
ENC_CP_DIR="checkpoints/asr_en_melspec_ptr_3conv_norm_adapt_4096"
DEC_CP_DIR="checkpoints/europarl-basic-tied-normalize"
CP="checkpoint_best.pt"

CUDA_VISIBLE_DEVICES=1 python generate-add-lang.py $DEST_DIR --path $ENC_CP_DIR/$CP --audio-input --no-cache-source \
    --beam 5 --batch 32 --source-lang ${SRC} --target-lang ${TGT} --task speech_interlingua_translation \
    --enc-model $ENC_CP_DIR/$CP   --enc-key ens-en \
    --dec-model $DEC_CP_DIR/$CP  --dec-key en-fr \
    --newkey ens-fr  --newarch speech_interlingua_transformer_big_3conv \
    --newtask speech_interlingua_translation --remove-bpe --freeze-schedule n-n \
    --skip-invalid-size-inputs-valid-test --max-source-positions 100000 --max-target-positions 5000 --audio-features 40 --final-norm


