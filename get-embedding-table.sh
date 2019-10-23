#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=get-embedding-nodist.log


#SRC="es"
#TGT="en"
#DEST_DIR="data-bin/wmt13.tokenized.32k.en-es"
#CP_DIR="checkpoints/esen-interlingua-finetuning"
#CP="checkpoint_best.pt"

#CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python get_embedding_table.py $DEST_DIR --path $CP_DIR/$CP  --source-lang $SRC --target-lang $TGT --task interlingua_translation --output-file embedding-es.json 

#CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python get_embedding_table.py $DEST_DIR --path $CP_DIR/$CP  --source-lang $TGT --target-lang $SRC --task interlingua_translation --output-file embedding-en.json 

#SRC="fr"
#TGT="en"
#DEST_DIR="data-bin/wmt13.tokenized.32k.fr-en"
#CP_DIR="checkpoints/add-lang-fren"
#CP="checkpoint_best.pt"

#CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python get_embedding_table.py $DEST_DIR --path $CP_DIR/$CP  --source-lang $SRC --target-lang $TGT --task multilingual_translation --output-file embedding-fr.json 


#SRC="de"
#TGT="en"
#DEST_DIR="data-bin/wmt13.tokenized.32k.de-en"
#CP_DIR="checkpoints/add-lang-deen"
#CP="checkpoint_best.pt"

#CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python get_embedding_table.py $DEST_DIR --path $CP_DIR/$CP  --source-lang $SRC --target-lang $TGT --task multilingual_translation --output-file embedding-de.json 

#SRC="src"
#TGT="tgt"
#DEST_DIR="../fairseq/data-bin/multi-europarl"
#CP_DIR="../fairseq/checkpoint/multi-europarl"
#CP="checkpoint_best.pt"

#CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python get_embedding_table.py $DEST_DIR --path $CP_DIR/$CP  --source-lang $SRC --target-lang $TGT --task translation --output-file embedding-multi.json 

SRC="es"
TGT="en"
DEST_DIR="data-bin/europarl"
CP_DIR="checkpoints/europarl"
CP="checkpoint_best.pt"

CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python get_embedding_table.py $DEST_DIR --path $CP_DIR/$CP  --source-lang es --target-lang en --task interlingua_nodistance_translation --output-file embedding-nodist-es.json --freeze-schedule n-n

CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python get_embedding_table.py $DEST_DIR --path $CP_DIR/$CP  --source-lang en --target-lang es --task interlingua_nodistance_translation --output-file embedding-nodist-en.json --freeze-schedule n-n

CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python get_embedding_table.py $DEST_DIR --path $CP_DIR/$CP  --source-lang fr --target-lang en --task interlingua_nodistance_translation --output-file embedding-nodist-fr.json --freeze-schedule n-n


