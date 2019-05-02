#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=50G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH -w veuc08                                                       
#SBATCH --output=encode-base.log


#WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
#SRC="en"
#TGT="tr"
#DEST_DIR="data-bin/wmt17.tokenized.16k.tr-en"
#CP_DIR="checkpoints/entr-interlingua"
#CP="checkpoint62.pt"


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
DEST_DIR="data-bin/europarl"
CP_DIR="checkpoints/europarl"
CP="checkpoint_best.pt"

mkdir enc-europarl

stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP  --beam 5 --batch-size 10 --source-lang en --target-lang es --task interlingua_nodistance_translation --output-file enc-europarl/encodings-en.json --n-points 500 --freeze-schedule n-n 

stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP  --beam 5 --batch-size 10 --source-lang de --target-lang es --task interlingua_nodistance_translation --output-file enc-europarl/encodings-de.json --n-points 500 --freeze-schedule n-n

stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP  --beam 5 --batch-size 10 --source-lang es --target-lang en --task interlingua_nodistance_translation --output-file enc-europarl/encodings-es.json --n-points 500 --freeze-schedule n-n

stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP  --beam 5 --batch-size 10 --source-lang fr --target-lang en --task interlingua_nodistance_translation --output-file enc-europarl/encodings-fr.json --n-points 500 --freeze-schedule n-n

