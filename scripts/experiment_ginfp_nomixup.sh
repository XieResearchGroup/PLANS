#!/bin/bash

EPOCHS=1000
BATCH_SIZE=256
ES_PATIENCE=20
REPEAT=3

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiment_ginfp_mixup \
    -p ./data/cyp450_smiles_GINfp_labels.json \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ginfp/ns_only_no_mixup \
    --repeat $REPEAT \
    --rand-seed $i
done
