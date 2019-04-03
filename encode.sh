#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=50G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=encode-base.log


#WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
#SRC="en"
#TGT="tr"
#DEST_DIR="data-bin/wmt17.tokenized.16k.tr-en"
#CP_DIR="checkpoints/entr-interlingua"
#CP="checkpoint62.pt"


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="de"
TGT="en"
DEST_DIR="data-bin/wmt13.tokenized.32k.de-en"
CP_DIR="checkpoints/add-lang-deen"
CP="checkpoint_best.pt"

stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP  --beam 5 --batch-size 10 --source-lang $SRC --target-lang $TGT --task multilingual_translation --output-file encodings-${SRC}.json --n-points 100 

#stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path $CP_DIR/$CP  --beam 5 --batch-size 10 --source-lang $TGT --target-lang $SRC --task multilingual_translation --output-file encodings-${TGT}.json --n-points 100

#stdbuf -i0 -e0 -o0 python encode.py $DEST_DIR --path ../fairseq/checkpoints/esen-es2en/$CP  --beam 1 --batch-size 1 --source-lang $SRC --target-lang $TGT --task translation --output-file encodings-${SRC}.json --n-points 100

stdbuf -i0 -e0 -o0 python encode.py data-bin/wmt13.tokenized.32k.en-es --path checkpoints/esen-interlingua-finetuning/$CP  --beam 1 --batch-size 1 --source-lang en --target-lang es --task interlingua_translation --output-file encodings-en.json --n-points 100


#stdbuf -i0 -e0 -o0 python encode.py data-bin/wmt13.tokenized.32k.en-es --path checkpoints/esen-interlingua-finetuning/$CP  --beam 1 --batch-size 1 --source-lang es --target-lang en --task interlingua_translation --output-file encodings-es100.json --n-points 100

#stdbuf -i0 -e0 -o0 python encode.py data-bin/wmt13.tokenized.32k.fr-en --path checkpoints/add-lang-fren/$CP  --beam 1 --batch-size 1 --source-lang fr --target-lang en --task multilingual_translation --output-file encodings-fr100.json --n-points 100


#stdbuf -i0 -e0 -o0 python encode.py data-bin/wmt13.tokenized.32k.en-es --path ../fairseq/checkpoints/esen-interlingua-finetuning/$CP  --beam 1 --batch-size 1 --source-lang en --target-lang es --task interlingua_translation --output-file encodings-en.json --n-points 3000

#stdbuf -i0 -e0 -o0 python encode.py data-bin/wmt13.tokenized.32k.fr-en/  --path ../fairseq/checkpoints/fren/$CP  --beam 1 --batch-size 1 --source-lang fr --target-lang en --task translation --output-file encodings-fr.json --n-points 100

stdbuf -i0 -e0 -o0 python visualization/preProcessData.py ${SRC}  encodings-${SRC}.json $DEST_DIR/newstest2013.tc.bpe.$SRC \
 ${TGT} encodings-${TGT}.json $DEST_DIR/newstest2013.tc.bpe.$TGT 100
