DATAPATH=./data/MolNet/tox21_ecfp.csv

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ecfp_linearModel_MLP_only.experiments_ecfp_linearModel_MLP_molnet \
    -p ${DATAPATH} \
    -e 1000 \
    -b 256 \
    --es-patience 20 \
    --log-path ./logs/linear/MLP_only/tox21_no_mixup \
    --repeat 0 \
    --rand-seed $i
done

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ecfp_linearModel_MLP_only.experiments_ecfp_linearModel_MLP_molnet \
    -p ${DATAPATH} \
    -e 1000 \
    -b 256 \
    --es-patience 20 \
    --log-path ./logs/linear/MLP_only/tox21_mixup \
    --repeat 0 \
    --mixup 0.4 \
    --mixup-repeat 10 \
    --rand-seed $i
done
