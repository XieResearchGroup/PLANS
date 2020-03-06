python -m src.training.experiment_conventional_multiclass ^
-p .\data\fromraw_cid_inchi_smiles_fp_labels_onehots.csv ^
--log-path .\logs\convention ^
--max-depth 5 ^
--n-estimators 500 ^
--rand-seed 0
