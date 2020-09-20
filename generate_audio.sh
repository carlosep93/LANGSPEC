
DEST_DIR='data-bin/audio-ende-melspec'
CP_DIR='checkpoints/st_de_melspec_scratch_adapt'

CUDA_VISIBLE_DEVICES=1 python generate.py $DEST_DIR --path \
    $CP_DIR/checkpoint_best.pt --task speech_interlingua_translation --audio-input --no-cache-source \
    --source-lang ens --target-lang de --freeze-schedule n-n --remove-bpe \
    --skip-invalid-size-inputs-valid-test --max-source-positions 100000 --max-target-positions 5000 --audio-features 40 --final-norm
     
