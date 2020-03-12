for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiment_linear_opt_mixup \
    -p ./data/fromraw_cid_inchi_smiles_fp_labels_onehots.csv \
    -e 1000 \
    -b 256 \
    --es-patience 20 \
    --log-path ./logs/linear/opt_mixup \
    --repeat 3 \
    --mixup 0.4 \
    --mixup-repeat 10 \
    --rand-seed $i
done