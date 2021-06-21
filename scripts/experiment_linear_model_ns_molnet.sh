#!/bin/bash

EPOCHS=1000
BATCH_SIZE=256
ES_PATIENCE=20
REPEAT=3
MIXUP=0.4
MIXUP_REPEAT=10
DATAPATH=./data/MolNet/tox21_ecfp.csv

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ecfp_linearModel_MLP_only.experiments_ecfp_linearModel_MLP_molnet_NS \
    -p ${DATAPATH} \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/linear/MLP_only/tox21_ns_no_mixup \
    --repeat 0 \
    --rand-seed $i
done

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ecfp_linearModel_MLP_only.experiments_ecfp_linearModel_MLP_molnet_NS \
    -p ${DATAPATH} \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/linear/MLP_only/tox21_ns_mixup \
    --repeat $REPEAT \
    --mixup $MIXUP \
    --mixup-repeat $MIXUP_REPEAT \
    --rand-seed $i
done