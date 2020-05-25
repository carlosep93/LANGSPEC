#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=multinli-results/multinli-multi-lstm.log


DEST_DIR="../fairseq/data-bin/multinli-joint-no-ru/"
CP_DIR="checkpoints/multinli-baseline-lstm"
CP="checkpoint_best.pt"
ENC_DIR="/scratch/carlos/europarl-multi-lstm-4layers/"
key="src-tgt"


inference() {

    echo "***************************************"
    echo $1
    echo "***************************************"

    cp $DEST_DIR/ref.test.$1 $DEST_DIR/ref.test.src
    cp $DEST_DIR/hyp.test.$1 $DEST_DIR/hyp.test.src
    cp $DEST_DIR/lab.test.$1 $DEST_DIR/lab.test.src
    
    CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python nli-inference.py $DEST_DIR --path $CP_DIR/$CP  --task nli --ref src --hyp src --batch-size 20 --raw-text  --enc-path $ENC_DIR/checkpoint_best.pt

}

for l1 in de en es fr; do
    inference $l1 
done

