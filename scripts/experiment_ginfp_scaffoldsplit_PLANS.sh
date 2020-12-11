#!/bin/bash

EPOCHS=1000
BATCH_SIZE=256
ES_PATIENCE=20
REPEAT=3
MIXUP=0.4
MIXUP_REPEAT=10

for i in 0 1029 812 204 6987
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ginfp_linearModel_PLANS_scaffoldSplitting.experiment_ginfp_scaffoldsplit_mixup \
    -p ./data/cyp450_smiles_GINfp_labels.json \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ginfp/ns_NoMixup_scaffoldSplitting \
    --repeat $REPEAT \
    --rand-seed $i \
    --learning-rate 1e-5
done

for i in 0 1029 812 204 6987
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ginfp_linearModel_PLANS_scaffoldSplitting.experiment_ginfp_scaffoldsplit_mixup \
    -p ./data/cyp450_smiles_GINfp_labels.json \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ginfp/ns_Mixup_scaffoldSplitting \
    --repeat $REPEAT \
    --mixup $MIXUP \
    --mixup-repeat $MIXUP_REPEAT \
    --rand-seed $i
done

for i in 0 1029 812 204 6987
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ginfp_linearModel_PLANS_scaffoldSplitting.experiment_ginfp_scaffoldsplit_balance \
    -p ./data/cyp450_smiles_GINfp_labels.json \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ginfp/ns_NoMixup_balanced_scaffoldSplitting \
    --repeat $REPEAT \
    --rand-seed $i \
    --learning-rate 1e-5
done

for i in 0 1029 812 204 6987
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ginfp_linearModel_PLANS_scaffoldSplitting.experiment_ginfp_scaffoldsplit_balance \
    -p ./data/cyp450_smiles_GINfp_labels.json \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ginfp/ns_Mixup_balanced_scaffoldSplitting \
    --repeat $REPEAT \
    --mixup $MIXUP \
    --mixup-repeat $MIXUP_REPEAT \
    --rand-seed $i \
done

