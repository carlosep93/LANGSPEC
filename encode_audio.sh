
SRC="en"
TGT="es"
DEST_DIR="data-bin/audio-enen-melspec"
CP_DIR="checkpoints/europarl-basic-tied-normalize"
CP="checkpoint_best.pt"

CUDA_VISIBLE_DEVICES=2 stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP  --beam 5 --batch-size 10 --source-lang $SRC --target-lang $TGT --task interlingua_nodistance_translation --output-file encodings-text.json --n-points 1000 --freeze-schedule n-n

SRC="ens"
TGT="en"
DEST_DIR="data-bin/audio-enen-melspec"
CP_DIR="checkpoints/asr_en_scratch_melspec-3conv"
CP="checkpoint_best.pt"



CUDA_VISIBLE_DEVICES=2 stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP  --beam 1 --batch-size 10 --source-lang $SRC --target-lang $TGT --task speech_interlingua_translation --output-file encodings-speech.json --n-points 1000 --freeze-schedule n-n --audio-features 40 --audio-input --no-cache-source --skip-invalid-size-inputs-valid-test --gen-subset test 


