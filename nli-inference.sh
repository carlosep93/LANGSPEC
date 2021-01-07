#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=multinli-results/multinli-basic-5layer.log


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
DEST_DIR="data-bin/multinli/"
CP_DIR="checkpoints/multinli-basic-tied-5layer"
CP="checkpoint_best.pt"
ENC_DIR="/scratch/carlos/europarl-basic-tied/"

inference() {

    echo "***************************************"
    echo $1
    echo "***************************************"
    
    key=$1-en
    if [ $1 == "en" ]; then
        key=$1-es
    fi

    stdbuf -i0 -e0 -o0 python nli-inference.py $DEST_DIR --path $CP_DIR/$CP  --task nli --ref-lang $1 --hyp-lang $1 --max-tokens 500 --raw-text  --ref-enc-path $ENC_DIR/checkpoint_best.pt --ref-enc-key $key --hyp-enc-path $ENC_DIR/checkpoint_best.pt --hyp-enc-key $key 


}

for l1 in de en es fr; do
    inference $l1 
done


ENC_DIR="/scratch/carlos/add-ruen-basic-tied"

inference ru
