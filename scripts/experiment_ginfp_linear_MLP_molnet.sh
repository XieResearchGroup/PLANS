for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ginfp_linearModel_MLP_only.experiment_ginfp_linearModel_MLP_molnet \
    -p ./data/MolNet/tox21_ginfp.json \
    -e 1000 \
    -b 256 \
    --es-patience 20 \
    --log-path ./logs/ginfp/MLP_only_tox21_no_ns_ginfp/no_mixup \
    --repeat 0 \
    --rand-seed $i \
    --learning-rate 1e-5
done

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiments_ginfp_linearModel_MLP_only.experiment_ginfp_linearModel_MLP_molnet \
    -p ./data/MolNet/tox21_ginfp.json \
    -e 1000 \
    -b 256 \
    --es-patience 20 \
    --log-path ./logs/ginfp/MLP_only_tox21_no_ns_ginfp/mixup \
    --repeat 0 \
    --mixup 0.4 \
    --mixup-repeat 10 \
    --rand-seed $i
done
