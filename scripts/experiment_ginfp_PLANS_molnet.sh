#!/bin/bash

EPOCHS=1000
BATCH_SIZE=128
ES_PATIENCE=20
REPEAT=3
MIXUP=0.4
MIXUP_REPEAT=10

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ginfp_linearModel_PLANS.experiment_ginfp_mixup_molnet \
    -p ./data/MolNet/tox21_ginfp.json \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ginfp/tox21_exploit_partial/no_mixup \
    --repeat $REPEAT \
    --rand-seed $i \
    --learning-rate 1e-5
done


for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ginfp_linearModel_PLANS.experiment_ginfp_mixup_molnet \
    -p ./data/MolNet/tox21_ginfp.json \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ginfp/tox21_exploit_partial/mixup \
    --repeat $REPEAT \
    --mixup $MIXUP \
    --mixup-repeat $MIXUP_REPEAT \
    --rand-seed $i
done

# for i in 0 1029 812 204 6987
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m src.training.experiments_ginfp_linearModel_PLANS.experiment_ginfp_balance_molnet \
#     -p ./data/MolNet/tox21_ginfp.json \
#     --outside-path ./data/ChEMBL24_ginfp.hdf5 \
#     -e $EPOCHS \
#     -b $BATCH_SIZE \
#     --es-patience $ES_PATIENCE \
#     --log-path ./logs/ginfp/tox21_chembl24_balanced_partial_no_mixup \
#     --repeat $REPEAT \
#     --rand-seed $i \
#     --learning-rate 1e-5
# done

# for i in 0 1029 812 204 6987
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m src.training.experiments_ginfp_linearModel_PLANS.experiment_ginfp_balance_molnet \
#     -p ./data/MolNet/tox21_ginfp.json \
#     --outside-path ./data/ChEMBL24_ginfp.hdf5 \
#     -e $EPOCHS \
#     -b $BATCH_SIZE \
#     --es-patience $ES_PATIENCE \
#     --log-path ./logs/ginfp/tox21_chembl24_balanced_partial_mixup \
#     --repeat $REPEAT \
#     --rand-seed $i \
#     --mixup $MIXUP \
#     --mixup-repeat $MIXUP_REPEAT
# done

# for i in 0 1029 812 204 6987
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m src.training.experiment_ginfp_scaffold_splitting_mixup \
#     -p ./data/cyp450_smiles_GINfp_labels.json \
#     -e $EPOCHS \
#     -b $BATCH_SIZE \
#     --es-patience $ES_PATIENCE \
#     --log-path ./logs/ginfp/ns_mixup_scaffold_splitting \
#     --repeat $REPEAT \
#     --mixup $MIXUP \
#     --mixup-repeat $MIXUP_REPEAT \
#     --rand-seed $i
# done
