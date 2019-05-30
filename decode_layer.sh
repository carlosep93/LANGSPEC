#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=50G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=decode_layer0.log


#WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
#SRC="en"
#TGT="tr"
#DEST_DIR="data-bin/wmt17.tokenized.16k.tr-en"
#CP_DIR="checkpoints/entr-interlingua"
#CP="checkpoint62.pt"


#WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
#SRC="de"
#TGT="en"
#DEST_DIR="data-bin/wmt13.tokenized.32k.de-en"
#CP_DIR="checkpoints/add-lang-deen"
#CP="checkpoint_best.pt"
LAYER=0

#stdbuf -i0 -e0 -o0 python generate_decoder_layer.py  $DEST_DIR --path $CP_DIR/$CP  --beam 1 --batch-size 1 --source-lang $SRC --target-lang $TGT --task multilingual_translation --output-file ../vectors-elora/interlingua-distance/decodings-layer${LAYER}/decodings-${SRC}-${LAYER}.json --n-points 200 --layer $LAYER

WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="fr"
TGT="en"
DEST_DIR="data-bin/wmt13.tokenized.32k.fr-en"
CP_DIR="checkpoints/add-lang-fren"
CP="checkpoint_best.pt"

#CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python generate_decoder_layer.py  $DEST_DIR --path $CP_DIR/$CP  --beam 1 --batch-size 1 --source-lang $SRC --target-lang $TGT --task multilingual_translation --output-file ../vectors-elora/interlingua-distance/decodings-layer${LAYER}/decodings-${SRC}-${LAYER}.json --n-points 3000 --layer $LAYER 


#CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python generate_decoder_layer.py data-bin/wmt13.tokenized.32k.en-es --path checkpoints/esen-interlingua-finetuning/$CP  --beam 1 --batch-size 1 --source-lang es --target-lang en --task interlingua_translation --output-file ../vectors-elora/interlingua-distance/decodings-layer${LAYER}/decodings-esen-${LAYER}.json --n-points 3000 --layer $LAYER


#CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python generate_decoder_layer.py data-bin/wmt13.tokenized.32k.en-es --path checkpoints/esen-interlingua-finetuning/$CP  --beam 1 --batch-size 1 --source-lang en --target-lang es --task interlingua_translation --output-file ../vectors-elora/interlingua-distance/decodings-layer${LAYER}/decodings-enes-${LAYER}.json --n-points 3000 --layer $LAYER

CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python generate_decoder_layer.py ../fairseq/data-bin/multi-europarl --path checkpoints/esen-interlingua-finetuning/$CP  --beam 1 --batch-size 1 --source-lang es --target-lang es --task interlingua_translation --output-file ../vectors-elora/interlingua-distance/decodings-layer${LAYER}/decodings-eses-${LAYER}.json --n-points 3000 --layer $LAYER


CUDA_VISIBLE_DEVICES="" stdbuf -i0 -e0 -o0 python generate_decoder_layer.py data-bin/wmt13.tokenized.32k.en-es --path checkpoints/esen-interlingua-finetuning/$CP  --beam 1 --batch-size 1 --source-lang en --target-lang en --task interlingua_translation --output-file ../vectors-elora/interlingua-distance/decodings-layer${LAYER}/decodings-enen-${LAYER}.json --n-points 3000 --layer $LAYER




#stdbuf -i0 -e0 -o0 python visualization/preProcessData.py en  decodings-enen.json $WORKING_DIR/newstest2013.tc.en \
# fr decodings-fr.json $WORKING_DIR/newstest2013.tc.fr 200 data_fren.json

#stdbuf -i0 -e0 -o0 python visualization/preProcessData.py en  decodings-enen.json $WORKING_DIR/newstest2013.tc.en \
# de decodings-de.json $WORKING_DIR/newstest2013.tc.de 200 data_deen.json

#stdbuf -i0 -e0 -o0 python visualization/preProcessData.py es  decodings-eses.json $WORKING_DIR/newstest2013.tc.es \
# fr decodings-fr.json $WORKING_DIR/newstest2013.tc.fr 200 data_fres.json

#stdbuf -i0 -e0 -o0 python visualization/preProcessData.py es  decodings-eses.json $WORKING_DIR/newstest2013.tc.es \
# de decodings-de.json $WORKING_DIR/newstest2013.tc.de 200 data_dees.json


#stdbuf -i0 -e0 -o0 python visualization/preProcessData.py es  decodings-eses.json $WORKING_DIR/newstest2013.tc.es \
# es decodings-enes.json $WORKING_DIR/newstest2013.tc.es 200 data_enesauto.json

#stdbuf -i0 -e0 -o0 python visualization/preProcessData.py en  decodings-enen.json $WORKING_DIR/newstest2013.tc.en \
# en decodings-esen.json $WORKING_DIR/newstest2013.tc.en 200 data_esenauto.json

