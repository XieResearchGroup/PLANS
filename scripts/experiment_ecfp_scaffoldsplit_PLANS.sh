#!/bin/bash

EPOCHS=1000
BATCH_SIZE=256
ES_PATIENCE=20
REPEAT=3
MIXUP=0.4
MIXUP_REPEAT=3


# no mixup, no partial
for i in 0 1029 812 204 6987
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ecfp_linearModel_PLANS_scaffoldSplitting.experiment_linear_exploit_partial \
    -p ./data/cyp450_smiles_ECFP_labels.json \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ecfp/ns_NoMixup_scaffoldSplitting \
    --repeat $REPEAT \
    --rand-seed $i \
    --learning-rate 1e-5
done


# no mixup, use partial
for i in 0 1029 812 204 6987
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ecfp_linearModel_PLANS_scaffoldSplitting.experiment_linear_chembl_balance_explit_partial \
    -p ./data/cyp450_smiles_ECFP_labels.json \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ecfp/ns_NoMixup_balanced_scaffoldSplitting \
    --repeat $REPEAT \
    --rand-seed $i \
    --learning-rate 1e-5
done


# mixup, no partial
for i in 0 1029 812 204 6987
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ecfp_linearModel_PLANS_scaffoldSplitting.experiment_linear_exploit_partial \
    -p ./data/cyp450_smiles_ECFP_labels.json \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ecfp/ns_Mixup_scaffoldSplitting \
    --repeat $REPEAT \
    --mixup $MIXUP \
    --mixup-repeat $MIXUP_REPEAT \
    --rand-seed $i
done


# mixup, use partial
for i in 0 1029 812 204 6987
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ecfp_linearModel_PLANS_scaffoldSplitting.experiment_linear_chembl_balance_explit_partial \
    -p ./data/cyp450_smiles_ECFP_labels.json \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ecfp/ns_Mixup_balanced_scaffoldSplitting \
    --repeat $REPEAT \
    --mixup $MIXUP \
    --mixup-repeat $MIXUP_REPEAT \
    --rand-seed $i
done

