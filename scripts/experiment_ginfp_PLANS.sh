#!/bin/bash

EPOCHS=1000
BATCH_SIZE=256
ES_PATIENCE=20
REPEAT=3
MIXUP=0.4
MIXUP_REPEAT=10

# for i in 0 1029 812 204 6987
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m src.training.experiment_ginfp_mixup \
#     -p ./data/cyp450_smiles_GINfp_labels.json \
#     -e $EPOCHS \
#     -b $BATCH_SIZE \
#     --es-patience $ES_PATIENCE \
#     --log-path ./logs/ginfp/ns_only_no_mixup/new_seeds \
#     --repeat $REPEAT \
#     --rand-seed $i \
#     --learning-rate 1e-5
# done


# for i in 0 1029 812 204 6987
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m src.training.experiment_ginfp_mixup \
#     -p ./data/cyp450_smiles_GINfp_labels.json \
#     -e $EPOCHS \
#     -b $BATCH_SIZE \
#     --es-patience $ES_PATIENCE \
#     --log-path ./logs/ginfp/ns_mixup/new_seeds \
#     --repeat $REPEAT \
#     --mixup $MIXUP \
#     --mixup-repeat $MIXUP_REPEAT \
#     --rand-seed $i
# done

# for i in 0 1029 812 204 6987
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m src.training.experiment_ginfp_balance \
#     -p ./data/cyp450_smiles_GINfp_labels.json \
#     --outside-path ./data/ChEMBL24_ginfp.hdf5 \
#     -e $EPOCHS \
#     -b $BATCH_SIZE \
#     --es-patience $ES_PATIENCE \
#     --log-path ./logs/ginfp/chembl24_balanced_partial_no_mixup/new_seeds \
#     --repeat $REPEAT \
#     --rand-seed $i \
#     --learning-rate 1e-5
# done

# for i in 0 1029 812 204 6987
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m src.training.experiment_ginfp_balance \
#     -p ./data/cyp450_smiles_GINfp_labels.json \
#     --outside-path ./data/ChEMBL24_ginfp.hdf5 \
#     -e $EPOCHS \
#     -b $BATCH_SIZE \
#     --es-patience $ES_PATIENCE \
#     --log-path ./logs/ginfp/chembl24_balanced_partial_mixup/new_seeds \
#     --repeat $REPEAT \
#     --rand-seed $i \
#     --mixup $MIXUP \
#     --mixup-repeat $MIXUP_REPEAT
# done

for i in 0 1029 812 204 6987
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiment_ginfp_scaffold_splitting_mixup \
    -p ./data/cyp450_smiles_GINfp_labels.json \
    -e $EPOCHS \
    -b $BATCH_SIZE \
    --es-patience $ES_PATIENCE \
    --log-path ./logs/ginfp/ns_mixup_scaffold_splitting \
    --repeat $REPEAT \
    --mixup $MIXUP \
    --mixup-repeat $MIXUP_REPEAT \
    --rand-seed $i
done
