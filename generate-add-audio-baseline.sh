#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=results/st-enfr-adapt.log





SRC="ens"
TGT="fr"
DEST_DIR="data-bin/audio-enfr"
ENC_CP_DIR="checkpoints/st_enfr_ptr_3conv_adapt"
DEC_CP_DIR="/scratch/carlos/europarl-basic-tied-normalize"
CP="checkpoint_best.pt"

python generate-add-lang.py $DEST_DIR --path $ENC_CP_DIR/$CP --audio-input --no-cache-source \
    --beam 5 --batch 32 --source-lang ${SRC} --target-lang ${TGT} --task speech_interlingua_translation \
    --enc-model $ENC_CP_DIR/$CP   --enc-key ens-fr \
    --dec-model $DEC_CP_DIR/$CP  --dec-key en-fr \
    --newkey ens-fr  --newarch speech_interlingua_transformer_big_3conv \
    --newtask speech_interlingua_translation --remove-bpe --freeze-schedule n-n \
    --skip-invalid-size-inputs-valid-test --max-source-positions 100000 --max-target-positions 5000 --audio-features 40 --final-norm


