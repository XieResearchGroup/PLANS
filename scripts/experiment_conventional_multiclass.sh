for i in {1..5}
do
    python -m src.training.experiment_conventional_multiclass \
    -p ./data/fromraw_cid_inchi_smiles_fp_labels_onehots.csv \
    --log-path ./logs/convention \
    --n-estimators 1000
done
