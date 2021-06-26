#!/bin/bash

EPOCHS=1000
BATCH_SIZE=128
ES_PATIENCE=20
REPEAT=3
MIXUP=0.4
MIXUP_REPEAT=10
DATAPATH=./data/MolNet/tox21_ecfp.csv

# for i in {1..5}
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m src.training.experiments_ecfp_linearModel_PLANS.experiment_linear_exploit_partial_molnet \
#     -p ${DATAPATH} \
#     -e $EPOCHS \
#     -b $BATCH_SIZE \
#     --es-patience $ES_PATIENCE \
#     --log-path ./logs/ecfp/tox21/exploit_partial/no_mixup \
#     --repeat $REPEAT
# done

# for i in {1..5}
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m src.training.experiments_ecfp_linearModel_PLANS.experiment_linear_exploit_partial_molnet \
#     -p ${DATAPATH} \
#     -e $EPOCHS \
#     -b $BATCH_SIZE \
#     --es-patience $ES_PATIENCE \
#     --log-path ./logs/ecfp/tox21/exploit_partial/mixup \
#     --repeat $REPEAT \
#     --mixup $MIXUP \
#     --mixup-repeat $MIXUP_REPEAT \
#     --rand-seed $i
# done

# for i in {1..5}
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m src.training.experiments_ecfp_linearModel_PLANS.experiment_linear_chembl_balance_exploit_partial_molnet \
#     -p ${DATAPATH} \
#     --outside-path ./data/ChEMBL24.hdf5 \
#     -e $EPOCHS \
#     -b $BATCH_SIZE \
#     --es-patience $ES_PATIENCE \
#     --log-path ./logs/ecfp/tox21/balance_partial/no_mixup \
#     --repeat 2 \
#     --rand-seed $i
# done

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ecfp_linearModel_PLANS.experiment_linear_chembl_balance_exploit_partial_molnet \
    -p ${DATAPATH} \
    --outside-path ./data/ChEMBL24.hdf5 \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ecfp/tox21/balance_partial/mixup \
    --repeat 2 \
    --rand-seed $i \
    --mixup 0.4 \
    --mixup-repeat 10
done
