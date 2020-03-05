for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m src.training.experiment_linear_balance_classes \
    -p ./data/fromraw_cid_inchi_smiles_fp_labels_onehots.csv \
    --outside-path ./data/DrugBank_smiles_fp.csv \
    -e 1000 \
    -b 256 \
    --es-patience 20 \
    --log-path ./logs/linear/drugbank_mixup \
    --repeat 5 \
    --rand-seed $i
done