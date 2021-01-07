#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=multinli-results/multinli-multi-joint-noru.log


DEST_DIR="/home/usuaris/veu/cescola/fairseq/data-bin/nli-europarl-noru-joint/"
CP_DIR="checkpoints/multinli-multi-joint-noru"
CP="checkpoint_best.pt"
ENC_DIR="/scratch/carlos/baseline-share-embeddings"
key="src-tgt"


inference() {

    echo "***************************************"
    echo $1
    echo "***************************************"

    cp $DEST_DIR/ref.test.$1 $DEST_DIR/ref.test.src
    cp $DEST_DIR/hyp.test.$1 $DEST_DIR/hyp.test.src
    cp $DEST_DIR/lab.test.$1 $DEST_DIR/lab.test.src
    
    CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python nli-inference.py $DEST_DIR --path $CP_DIR/$CP  --task nli --ref-lang src --hyp-lang src --max-tokens 500 --raw-text  --ref-enc-path $ENC_DIR/checkpoint_best.pt --hyp-enc-path $ENC_DIR/checkpoint_best.pt

}

for l1 in de en es fr; do
    inference $l1 
done

