#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=30G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=results/asr-enen-scratch.log




DEST_DIR='data-bin/audio-multi'
CP_DIR='checkpoints/asr_en_enfr_scratch_melspec-3conv'


python generate.py $DEST_DIR --path \
    $CP_DIR/checkpoint_best.pt --task speech_interlingua_translation --audio-input --no-cache-source \
    --source-lang ens --target-lang en --freeze-schedule n-n  \
    --skip-invalid-size-inputs-valid-test --max-source-positions 100000 --max-target-positions 5000 --audio-features 40 --final-norm 
     
