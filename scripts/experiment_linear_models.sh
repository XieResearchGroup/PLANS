#!/bin/bash

EPOCHS=1000
BATCH_SIZE=256
ES_PATIENCE=20
REPEAT=3
MIXUP=0.4
MIXUP_REPEAT=10

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiment_linear_models \
    -p ./data/fromraw_cid_inchi_smiles_fp_labels_onehots.csv \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/linear/linear_models \
    --repeat $REPEAT
done

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiment_linear_model_mixup \
    -p ./data/fromraw_cid_inchi_smiles_fp_labels_onehots.csv \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/linear/mixup \
    --repeat $REPEAT \
    --mixup $MIXUP \
    --mixup-repeat $MIXUP_REPEAT
done

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiment_linear_model_w_outside_data \
    -p ./data/fromraw_cid_inchi_smiles_fp_labels_onehots.csv \
    --outside-path ./data/DrugBank_smiles_fp.csv \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/linear/drug_bank \
    --repeat $REPEAT
done

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiment_linear_drugbank_mixup \
    -p ./data/fromraw_cid_inchi_smiles_fp_labels_onehots.csv \
    --outside-path ./data/DrugBank_smiles_fp.csv \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/linear/drugbank_mixup \
    --repeat $REPEAT \
    --mixup $MIXUP \
    --mixup-repeat $MIXUP_REPEAT
done

