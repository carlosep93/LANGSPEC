#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=get-embedding-multi.log


SRC="src"
TGT="tgt"
DEST_DIR="../fairseq/data-bin/multi-europarl-ru-joint"
CP_DIR="/scratch/carlos/baseline-ru-hare-embeddings"
CP="checkpoint_best.pt"


CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python get_embedding_table.py $DEST_DIR --path $CP_DIR/$CP  --source-lang $SRC --target-lang $TGT --task translation --output-file encodings-tied/shared/embeddings-shared.json



